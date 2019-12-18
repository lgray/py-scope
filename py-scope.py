import select, socket
import sys
from time import sleep
import binascii
import yaml
import numpy as np
from concurrent.futures import ProcessPoolExecutor

DEFAULT_LONG_READ = 1 << 24
SCOPE_NCHANNELS = 4

class scope(object):
    def __init__(self, conf):
        self.sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.verbose = conf['verbose']        
        self.timeout = conf['timeout']
        self.sck.connect((conf['hostname'], conf['port']))
        self.ro_query = {'horizontal': 'HORizontal:ACQLENGTH?;:WFMOutpre:XINcr?;:WFMOutpre:PT_Off?',
                         'fastframe': 'HORizontal:FASTframe:STATE?;:HORizontal:FASTframe:COUNt?',
                         'channels': 'SELECT:CH1?;:SELECT:CH2?;:SELECT:CH3?;:SELECT:CH4?',
                         'vertical': 'DATA:SOURCE CH%d;:WFMOutpre:YMUlt?;:WFMOutpre:YOFf?;:WFMOutpre:YZEro?;:WFMOutpre:BYT_Nr?'}

    def shutdown(self):
        self.sck.shutdown(socket.SHUT_RDWR)
        self.sck.close()

    def send_cmd(self, cmd, readsize=DEFAULT_LONG_READ):
        if self.verbose: 
            print('sending ->',cmd)        
        padded = cmd + '\n'    
        self.sck.send(padded.encode())
        if 'CURVE' in cmd:
            chunks = []
            self.sck.settimeout(self.timeout)
            rcd = b''
            while True:
                try:
                    rcd = self.sck.recv(readsize)
                    if len(rcd) > 0:
                        break
                except Exception:
                    pass
            while len(rcd) > 0:
                chunks.append(rcd)                
                try:
                    rcd = self.sck.recv(readsize)
                except Exception:
                    rcd = b''
            return b''.join(chunks)
        elif '?' in cmd:
            return self.sck.recv(4096).decode().strip()
        elif '*RST' == cmd:
            sleep(3)
            return
        else:
            return ''
    
    def describe_readout(self):
        out = {}
        for key, query in self.ro_query.items():            
            if 'channels' in key:
                ret = self.send_cmd(query)
                out['chmask'] = [bool(int(ich)) for ich in ret.split(';')]
                out['nch'] = sum([int(ich) for ich in ret.split(';')])
            if 'vertical' in key:
                for i in range(SCOPE_NCHANNELS):
                    ret = self.send_cmd(query % (i+1))
                    #labels = ['ymult', 'yoffset', 'yzero', 'nbytes']
                    out['vertical%d' % (i+1)] = [float(x) if i < 3 else int(x) for i, x in enumerate(ret.split(';'))]
            if 'horizontal' in key:
                ret = self.send_cmd(query).split(';')
                out['nPt'] = int(ret[0])
                out['dt'] = float(ret[1])
                out['t0'] = float(ret[2])
            if 'fastframe' in key:
                ret = self.send_cmd(query).split(';')
                out['fastframe'] = True if ret[0] != '0' else False
                out['nFrames'] = 0 if not out['fastframe'] else int(ret[1])
        sum_bytes = 0
        for i in range(SCOPE_NCHANNELS):
            sum_bytes += out['vertical%d' % (i+1)][3]*out['chmask'][i]
        out['readout_size'] = out['nPt']
        out['readout_size_bytes'] = out['nPt'] * sum_bytes
        if out['fastframe'] and out['nFrames'] > 0:
            out['readout_size'] = out['nPt'] * out['nFrames']
            out['readout_size_bytes'] = out['nPt'] * out['nFrames'] * sum_bytes
        ro_ch = []
        for i, read in enumerate(out['chmask']):
            if read:
                ro_ch.append(i+1)
        source_cmd = 'DATA:SOURCE ' + ','.join(['CH%d' % i for i in ro_ch])
        self.send_cmd(source_cmd)
        return out
        

out_file='test.hdf5'
myscope = None
states = {}
tasks = None
print('opening', sys.argv[1])
with open(sys.argv[1]) as f:    
    config = yaml.load_all(f, Loader=yaml.SafeLoader)
    for conf in config:
        if len(conf.keys()) == 1 and 'scope' in conf.keys():
            print('scope:', conf['scope'])
            myscope = scope(conf['scope'])
        elif len(conf.keys()) == 1 and 'task' in conf.keys():
            tasks = conf['task']
        else:
           print('adding states:', list(conf.keys()))
           states.update(conf)

def acq(thescope, loop_spec, header_info):
    import time
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind('tcp://0.0.0.0:33373')
    nevents = 0
    # need to calculate expected read size based on header info
    readsize = DEFAULT_LONG_READ
    while True:
        try:
            for step in loop_spec:
                data = thescope.send_cmd(step, readsize)
                socket.send(data, zmq.NOBLOCK)
            nevents += 1
        except KeyboardInterrupt:
            return nevents

def receiver():
    import zmq
    context = zmq.Context()
    rsck = context.socket(zmq.PULL)
    rsck.connect('tcp://0.0.0.0:33373')
    ssck = context.socket(zmq.PUSH)
    ssck.connect('tcp://0.0.0.0:33374')
    while True:
        try:
            ssck.send(rsck.recv())
        except KeyboardInterrupt:
            return

def unpack_buffers(data, header_info):
    assert data[-1] == 10 #newline in ascii
    assert data[0] == 35 #hash in ascii
    unpack = data[:-1] # pop the trailing newline

    def consume_header(data):
        if len(data) == 0:
            return 0, data
        assert data[0] == 35
        nbytes_len = int(data[1:2], 16)
        nbytes_payload = int(data[2:2+nbytes_len])
        out = data[2+nbytes_len:]
        return nbytes_payload, out
    
    bufsize, unpack = consume_header(unpack)
    out = {}
    for i in range(SCOPE_NCHANNELS):
        if header_info['chmask'][i]:
            npbuf = np.frombuffer(unpack[:bufsize], dtype=np.int8)
            out[i+1] = npbuf
            unpack = unpack[bufsize:]
            bufsize, unpack = consume_header(unpack)
    #print('received -> data of size', len(data), 'bytes and', len(out.keys()), 'sub-buffers')
    return out

def writer(out_file,header_info):
    import zmq
    import h5py
    context = zmq.Context()
    rsck = context.socket(zmq.PULL)
    rsck.bind('tcp://0.0.0.0:33374')
    outf = h5py.File('myfile.hdf5','w')
    dset = outf.create_dataset("waveform",
                               (header_info['nch'], 0),
                               maxshape=(header_info['nch'], None),
                               compression="gzip",
                               dtype = np.int8)
    for k, v in header_info.items():
        dset.attrs.create(k, v)
    #print(dset)
    nevents = 0
    while True:
        try:
            data = rsck.recv()
            bufs = unpack_buffers(data, header_info)
            stacked = np.stack(([bufs[i] for i in bufs.keys()]))            
            nevents += 1
            dset.resize((header_info['nch'], nevents*header_info['readout_size']))
            dset[:,(nevents-1)*header_info['readout_size']:nevents*header_info['readout_size']] = stacked
            print('saved %d events' % nevents, end= '\r')
        except KeyboardInterrupt:
            break
    outf.close()
    return

def run_acq_loop(thescope, loop_spec, out_file):
    header_info = thescope.describe_readout()
    with ProcessPoolExecutor(max_workers=3) as pexec:
        acq_loop = pexec.submit(acq, thescope, loop_spec, header_info)
        write_loop = pexec.submit(writer, out_file, header_info)
        rec_loop = pexec.submit(receiver)
        while acq_loop.running():
            try:
                # initialize zmq and do data-writing
                time.sleep(0.5)
            except KeyboardInterrupt:            
                break
        print('\nevent_loop sent', acq_loop.result(), 'events')
       

import time
if tasks is not None:
    for task in tasks:
        if task not in states.keys():
            raise Exception('%s is not an available task!' % task)
        if task == 'acquisition_loop':
            run_acq_loop(myscope, states[task], out_file)
        else:
            for command in states[task]:
                out = myscope.send_cmd(command)                
                if 'CURVE' in command or '?' in command:
                    print(out)

myscope.shutdown()

