import select, socket
import sys
from time import sleep
import binascii
import yaml
import numpy as np
import h5py
import zmq
from concurrent.futures import ProcessPoolExecutor

DEFAULT_LONG_READ = 1 << 24
SCOPE_NCHANNELS = 4

class scope(object):
    def __init__(self, conf):
        self.sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.verbose = conf['verbose']        
        self.timeout = conf['timeout']
        if self.timeout == 'None': 
            self.timeout = None
        self.file_split = conf['file_split'] if 'file_split' in conf.keys() else -1
        self.daq_timeout = conf['daq_timeout'] if 'daq_timeout' in conf.keys() else self.timeout
        self.sck.connect((conf['hostname'], conf['port']))
        self.ro_query = {'horizontal': 'HORizontal:ACQLENGTH?;:WFMOutpre:XINcr?;:WFMOutpre:PT_Off?',
                         'fastframe': 'HORizontal:FASTframe:STATE?;:HORizontal:FASTframe:COUNt?',
                         'channels': 'SELECT:CH1?;:SELECT:CH2?;:SELECT:CH3?;:SELECT:CH4?',
                         'vertical': 'DATA:SOURCE CH%d;:WFMOutpre:YMUlt?;:WFMOutpre:YOFf?;:WFMOutpre:YZEro?;:WFMOutpre:BYT_Nr?'}

    def shutdown(self):
        self.sck.shutdown(socket.SHUT_RDWR)
        self.sck.close()

    def send_cmd(self, cmd, readsize=DEFAULT_LONG_READ, zmq_socket=None):
        if self.verbose: 
            print('sending ->',cmd)        
        padded = cmd + '\n'
        self.sck.send(padded.encode())
        if 'CURVE' in cmd:
            outbuf = b''
            self.sck.settimeout(self.daq_timeout)
            rcd = b''
            
            while True:
                try:
                    rcd = self.sck.recv(readsize)
                    if len(rcd) > 0:
                        break
                except Exception:
                    pass
            total_len = len(rcd)
            rcd_len = len(rcd)
            zmq_socket.send(rcd, zmq.NOBLOCK)
            #print('rcd ->', rcd_len, total_len, rcd[:min(20, len(rcd))], '...', rcd[max(0,rcd_len-20):])
            
            outbuf += rcd
            while outbuf[-1] != 10 or zmq_socket != None:
                try:
                    rcd = self.sck.recv(readsize)                
                except Exception:
                    rcd = b''
                rcd_len = len(rcd)
                #print('rcd', rcd)
                if rcd_len > 0 and rcd != b'\n':
                    total_len += rcd_len
                    #print('rcd ->', rcd_len, total_len, rcd[:min(20, len(rcd))], '...', rcd[max(0,rcd_len-20):])
                    #outbuf += rcd
                    #outbuf = b''
                    zmq_socket.send(rcd, zmq.NOBLOCK)
            print('ended!')
            self.sck.settimeout(self.timeout)          
            return outbuf
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
            channel_bytes = (1 if not out['fastframe'] else out['nFrames']) * out['nPt'] * out['vertical%d' % (i+1)][3]
            str_ch_bytes = str(channel_bytes)
            header = '#'+hex(len(str_ch_bytes)).strip('0x').upper()+str_ch_bytes
            print('expected header:',header)
            sum_bytes += out['chmask'][i]*(channel_bytes + len(header))
        sum_bytes += 1 # for the '\n' at the end
        out['readout_size'] = out['nPt']
        out['readout_size_bytes'] = sum_bytes
        if out['fastframe'] and out['nFrames'] > 0:
            out['readout_size'] = out['nPt'] * out['nFrames']            
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
if len(sys.argv) > 2:
    print('saving to', sys.argv[2])
    out_file = sys.argv[2]
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
    readsize = header_info['readout_size_bytes']
    while True:
        try:
            for step in loop_spec:
                data = thescope.send_cmd(step, readsize, zmq_socket=socket)
                if thescope.verbose:
                    print('sending data ->', data[:min(80,len(data))])
                socket.send(data, zmq.NOBLOCK)
            nevents += 1
        except KeyboardInterrupt:
            try:
                thescope.send_cmd('BUSY?')
            except:
                pass

            return nevents

def receiver():
    import zmq
    context = zmq.Context()
    rsck = context.socket(zmq.PULL)
    rsck.connect('tcp://0.0.0.0:33373')
    ssck = context.socket(zmq.PUSH)
    ssck.connect('tcp://0.0.0.0:33374')
    outbuf = b''
    while True:
        try:
            data = rsck.recv()
            if data != b';\n':
                outbuf += data
            #print(len(data), len(outbuf), data[:min(len(data), 20)], '...', data[max(0, len(data) - 20):])
            has_terminator = outbuf.find(b';\n#')
            while has_terminator > -1:
                #print('found terminator->',outbuf[has_terminator-5:has_terminator+2])
                tosend = outbuf[:has_terminator+2]
                #print('sending buf:', tosend)
                ssck.send(tosend)
                outbuf = outbuf[has_terminator+2:]
                has_terminator = outbuf.find(b';\n#')
                print(outbuf[:min(len(outbuf),10)], len(outbuf), outbuf[max(len(outbuf)-10, 0):], has_terminator)
                
            if len(outbuf) > 2 and (outbuf[-2:] == b';\n' or outbuf[-1:] == b';'):                
                if outbuf[-2:] == b';\n':
                    tosend = outbuf
                elif outbuf[-1:] == b';':
                    tosend = outbuf + b'\n'
                #print('tosend ->', len(tosend), tosend[:20], '...', tosend[-20:])
                ssck.send(tosend)                
                outbuf = b''
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e)

def unpack_buffers(data, header_info):
    assert data[-1] == 10 #newline in ascii
    assert data[-2] == 59 #semicolon in ascii
    assert data[0] == 35 #hash in ascii
    unpack = data[:len(data)-2] # pop the trailing semicolon and newline

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

def writer(out_file, header_info, file_split):
    import zmq
    import h5py
    context = zmq.Context()
    rsck = context.socket(zmq.PULL)
    rsck.bind('tcp://0.0.0.0:33374')
    ifile = 0
    out_file_parts = out_file.split('.')
    fname = out_file_parts[0] + '_{0}'.format(ifile) + '.' + '.'.join(out_file_parts[1:])
    outf = h5py.File(fname,'w')
    dset = outf.create_dataset("waveform",
                               (header_info['nch'], 0),
                               maxshape=(header_info['nch'], None),
                               compression="gzip",
                               dtype = np.int8)
    for k, v in header_info.items():
        dset.attrs.create(k, v)
    #print(dset)
    nevents_tot = 0
    nevents = 0
    while True:
        try:
            data = rsck.recv()
            bufs = unpack_buffers(data, header_info)
            stacked = np.stack(([bufs[i] for i in bufs.keys()]))            
            nevents += 1
            #print('datashape ->',stacked.shape)
            data_len = stacked.shape[-1]
            dset.resize((header_info['nch'], nevents*data_len))
            dset[:,(nevents-1)*data_len:nevents*data_len] = stacked
            print('saved %d events' % ((nevents + nevents_tot) * ( header_info['nFrames'] if header_info['fastframe'] else 1 )) , end= '\r')
            if file_split > 0 and nevents > file_split:
                outf.close()
                nevents_tot += nevents
                nevents = 0
                ifile += 1
                fname = out_file_parts[0] + '_{0}'.format(ifile) + '.' + '.'.join(out_file_parts[1:])
                print('opening new file:', fname)
                outf = h5py.File(fname,'w')
                dset = outf.create_dataset("waveform",
                                           (header_info['nch'], 0),
                                           maxshape=(header_info['nch'], None),
                                           compression="gzip",
                                           dtype = np.int8)
                for k, v in header_info.items():
                    dset.attrs.create(k, v)
        except KeyboardInterrupt:
            break
    #print('\n')
    outf.close()
    return

def run_acq_loop(thescope, loop_spec, out_file):
    header_info = thescope.describe_readout()
    with ProcessPoolExecutor(max_workers=3) as pexec:
        acq_loop = pexec.submit(acq, thescope, loop_spec, header_info)
        write_loop = pexec.submit(writer, out_file, header_info, thescope.file_split)
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
                try:
                    out = myscope.send_cmd(command)                
                    if 'CURVE' in command or '?' in command:
                        print(out)
                except KeyboardInterrupt:
                    pass

myscope.shutdown()

