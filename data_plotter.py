import os, sys
import h5py
import numpy as np
import pandas as pd
import os, sys
import h5py
import numpy as np
import pandas as pd

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import integrate

import scipy.stats

import numba

from scipy.optimize import curve_fit
from scipy.stats import norm

import matplotlib.font_manager as font_manager
from tqdm.autonotebook import tqdm

tdc_bin = 5e-3  ## in ns

def extract_dataset(filename, dataset_name='waveform'):
    f = h5py.File(filename,'r')
    dataset = f[dataset_name]
    attrs_out = dict(dataset.attrs)
    ymults = [dataset.attrs['vertical{0}'.format(i+1)][0] for i in range(4)]
    yzeros = [dataset.attrs['vertical{0}'.format(i+1)][1] for i in range(4)]
    npoints = dataset.attrs['nPt']
    events = dataset.shape[1]//npoints
    chmask = dataset.attrs['chmask']  
    data_out = np.zeros(4*dataset.shape[1]).reshape((4, events, npoints))
    ich = 0
    for i in range(4):
        if chmask[i]:
            data_out[i] = (dataset[ich].reshape(events, npoints) - yzeros[i])*ymults[i]
            ich += 1
    f.close()
    return data_out, attrs_out

def const(x, c):
    return c

def calculate_voltages_raw(v_in, pedestal_length=350):
    v_preamp_adjusted = np.copy(v_in)
    for i in range(v_in.shape[0]):
        ped = 0.0
        try:
            popt, pcov = curve_fit(const, np.arange(v_in.shape[1]), v_in[i, :pedestal_length], p0=[ped])
            ped = popt[0]
        except RuntimeError:
            ped = np.mean(v_in[i, :pedestal_length])
        v_preamp_adjusted[i] = v_preamp_adjusted[i] - ped
    return v_preamp_adjusted

def calculate_voltages(v_in, gain_post=-10, pedestal_length=400):
    gain_post_inv = 1.0/gain_post
    v_preamp_adjusted = gain_post_inv * calculate_voltages_raw(v_in, pedestal_length=pedestal_length)
    return v_preamp_adjusted

def calculate_tcross(v_in, percent_thresh, dt, gain_post=-10, pedestal_length=400):
    v_preamp_pedsub = calculate_voltages(v_in, gain_post=gain_post, 
                                         pedestal_length=pedestal_length)
    time = np.arange(v_in.shape[1])*dt
    idx_max = np.argmax(v_preamp_pedsub, axis=-1)
    threshold = np.max(v_preamp_pedsub, axis=-1)*percent_thresh
    t0s = np.zeros(v_in.shape[0])
    
    for evt in range(v_in.shape[0]):
        t0 = -1.0
        if idx_max[evt] > 0:
            spline = InterpolatedUnivariateSpline(time, v_preamp_pedsub[evt])        
            start = time[np.argmin(abs(spline(time[:idx_max[evt]])-threshold[evt]))]
            try:
                t0 = optimize.newton(lambda x: spline(x)-threshold[evt], x0=start, maxiter=500)
            except RuntimeError:
                t0 = np.nan
        t0s[evt] = t0
    
    return t0s

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def moyal(x, norm, loc, scale):
    return norm*scipy.stats.moyal.pdf(x, loc, scale)

def plot_tcross(ich, t0s, trigger_t0s=None, ax=None, dofit=False, num_bins=20):
    mask = t0s > -0.5
    if trigger_t0s is not None:
        mask = (t0s > -0.5) & (trigger_t0s > -0.5)
    clean_t0s = t0s[mask]
    if trigger_t0s is not None:
        clean_trigger_t0s = trigger_t0s[mask]

    to_plot = clean_t0s    
    if trigger_t0s is not None:
        to_plot = clean_t0s - clean_trigger_t0s
    
    mean = np.mean(to_plot)
    sigma = np.std(to_plot, ddof=1)
    
    range_hist = (np.min(to_plot),np.max(to_plot))
    if ax is None:
        fig, ax = plt.subplots(dpi=400)
    
    if dofit:
        bins, edges = np.histogram(to_plot, num_bins, density=False)
        centers = 0.5*(edges[1:] + edges[:-1])
        try:
            popt, pcov = curve_fit(gaus,centers,bins,p0=[1,mean,sigma])
            ax.plot(centers, gaus(centers,popt[0], popt[1], popt[2]))
            mean = popt[1]
            sigma = popt[2]
        except:
            pass        
    
    ax.hist(to_plot, num_bins, range=range_hist, 
            density=False,
            label='mean = %.3g s\nsigma = %.3g\n#event = %d\n#bin = %d'%(mean,sigma,to_plot.size,num_bins))
    ax.legend()
    ax.grid(which='both')
    if trigger_t0s is None:
        ax.set(xlabel='Time (s)', ylabel='Occurance',
               title='Channel {0}'.format(ich+1))
    else:
        ax.set(xlabel=r'Time_{Trigger} - Time_{Channel} (s)', ylabel='Occurance',
               title='Channel {0}'.format(ich+1))

def plot_time_quan(ich, t0s, trigger_t0s=None, ax=None, dofit=False, num_bins=20):
    mask = t0s > -0.5
    if trigger_t0s is not None:
        mask = (t0s > -0.5) & (trigger_t0s > -0.5)
    clean_t0s = t0s[mask]
    if trigger_t0s is not None:
        clean_trigger_t0s = trigger_t0s[mask]

    to_plot = clean_t0s    
    if trigger_t0s is not None:
        to_plot = clean_t0s - clean_trigger_t0s
    
    mean = np.mean(to_plot)
    sigma = np.std(to_plot, ddof=1)
    
    range_hist = (np.min(to_plot),np.max(to_plot))
    if ax is None:
        fig, ax = plt.subplots(dpi=400)
    
    if dofit:
        bins, edges = np.histogram(to_plot, num_bins, density=False)
        centers = 0.5*(edges[1:] + edges[:-1])
        try:
            popt, pcov = curve_fit(gaus,centers,bins,p0=[1,mean,sigma])
            ax.plot(centers, gaus(centers,popt[0], popt[1], popt[2]))
            mean = popt[1]
            sigma = popt[2]
        except:
            pass        
    
    ax.hist(to_plot, num_bins, range=range_hist, 
            density=False,
            label='mean = %.3g s\nsigma = %.3g\n#event = %d\n#bin = %d'%(mean,sigma,to_plot.size,num_bins))
    ax.legend()
    ax.grid(which='both')
    if trigger_t0s is None:
        ax.set(xlabel='Time (s)', ylabel='Occurance',
               title='Channel {0}'.format(ich+1))
    else:
        ax.set(xlabel=r'Time_{Trigger} - Time_{Channel} (s)', ylabel='Occurance',
               title='Channel {0}'.format(ich+1))


@numba.jit(nopython=True)
def get_time_index(vals, thresh):
    istep = 0
    while vals[istep] > thresh:
        istep += 1
    return istep

def calculate_time(v_in, dt, th_cfd=0.5, tdc_bin=0.005, tdc_start = 40):
    eventLen = v_in.shape[0]
    time = np.arange(v_in.shape[1])*dt*1e9 #nanoseconds
    
    t_cfd = np.zeros(eventLen)
    for itrace, trace in enumerate(v_in):
        v_pk = np.argmin(trace)
        v_pk_range = np.linspace(time[max(0, v_pk - 20)], time[min(v_in.shape[1]-1, v_pk+20)], 10000)
        tdc_start = time[max(0, v_pk - 20)]
        tdc_max = time[v_in.shape[1]-1]
        interp_signal = InterpolatedUnivariateSpline(time, trace)
        v_pk = np.min(interp_signal(v_pk_range))        
        v_th_cfd = 0 - th_cfd*(0 - v_pk)
        
        binned_signal = interp_signal(np.linspace(start=tdc_start, stop=tdc_max, num=int((tdc_max-tdc_start)/0.005) + 1))
        t_cfd[itrace] = tdc_start + get_time_index(binned_signal, v_th_cfd)*0.005
        
    std_t_cfd = np.std(t_cfd, ddof=1)
    std_t_cfd = std_t_cfd * 1e3
    mean_t_cfd = np.mean(t_cfd)
    return t_cfd, std_t_cfd, mean_t_cfd
    
def calculate_charge(v_in, dt, transCond, gain_post=-10, pedestal_length=400, charge_norm=1e15):
    v_preamp_pedsub = calculate_voltages(v_in, gain_post=gain_post, 
                                         pedestal_length=pedestal_length)
    time = np.arange(v_in.shape[1])*dt
    norm = charge_norm/transCond
    
    return integrate.simps(norm*v_preamp_pedsub, time, axis=-1)


def plot_charge(dataset, ich, dt, transCond, mask=None, ax=None, gain_post=-10, 
                pedestal_length=400, charge_norm=1e15, num_bins=200, dofit=True):
    charges = calculate_charge(dataset[ich], dt, transCond, 
                               gain_post=gain_post,
                               pedestal_length=pedestal_length,                               
                               charge_norm=charge_norm)
    Q_avg = np.mean(charges)
    minmax = (np.min(charges),np.max(charges))
    range_hist = (0,20)#(minmax[0]*(0.5 if minmax[0] > 0 else 2.0),minmax[1]*(2.0 if minmax[1] > 0 else 0.5))
    if ax is None:
        fig, ax = plt.subplots(dpi=400)

    scale = np.std(charges, ddof=1)
    loc = np.mean(charges)
    norm = charges.size
    if dofit:
        bins, edges = np.histogram(charges, num_bins, range=range_hist, density=False)
        centers = 0.5*(edges[1:] + edges[:-1])
        try:
            popt, pcov = curve_fit(moyal, centers, bins, p0=[norm, loc,scale])
            ax.plot(centers, moyal(centers, popt[0], popt[1], popt[2]))
            norm = popt[0]
            loc = popt[1]
            scale = popt[2]
        except:
            pass 
    #print('landau fit ->', norm, loc, scale)    
    
    #ax.set_yscale('log')
    ax.hist(charges, num_bins, range=range_hist, 
             density=False,
             label='peak = %.2f fC\n#event = %d\n#bin = %d'%(loc,dataset.shape[1],num_bins))
    ax.legend()
    ax.grid(which='both')
    ax.set(xlabel='Charge (fC)', ylabel='Occurance',
           title='Channel {0}'.format(ich+1))
    return ax


def plot_amplitude(dataset, ich, ax=None, gain_post=-10, 
                   pedestal_length=400, num_bins = 100):
    v_preamp_pedsub = calculate_voltages(dataset[ich], gain_post=gain_post, 
                                         pedestal_length=pedestal_length)
    v_pk = np.max(v_preamp_pedsub, axis=-1)
    
    range_ampl = (0, 0.3)#(np.min(v_pk)*0.5,np.max(v_pk)*2)
    if ax is None:
        fig, ax = plt.subplots(dpi=400)
    ax.hist(v_pk, num_bins, range=range_ampl, density=False,
            label='#event = %d\n#bin = %d'%(dataset.shape[1],num_bins))
    ax.legend()
    ax.grid()
    ax.set(xlabel='Amplitude (V)', ylabel='Occurance',
           title='Channel {0}'.format(ich+1))
    return ax

def plot_waveform(time, voltage, pp, xlable="Time(ns)", ylable="Voltage(V)", title="Raw Data ch1", pdf=False, pic=False):
    fig, ax1 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(voltage))):
        ax1.plot(time, voltage[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax1.grid()
    ax1.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch1')
    if pdf:
        pp.savefig(fig)
        pp.close()
    if pic:
        plt.show()
    plt.close(fig)

def plot_waveforms(time, v_ch1, v_ch2, v_ch3, v_ch4, pp, xlable="Time(ns)", ylable="Voltage(V)", 
                   title="Raw Data ch1", pdf=False, pic=False):
    fig, ax1 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(v_ch1))):
        ax1.plot(time, v_ch1[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax1.grid()
    ax1.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch1')
    if pdf==True:
        pp.savefig(fig)
    if pic==True:
        plt.show()
    plt.close(fig)
    
    fig, ax2 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(v_ch2))):
        ax2.plot(x*dt*1e9, v_ch2[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax2.grid()
    ax2.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch2')
    if pdf:
        pp.savefig(fig)
    if pic:
        plt.show()
    plt.close(fig)
    
    fig, ax3 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(v_ch3))):
        ax3.plot(x*dt*1e9, v_ch3[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax3.grid()
    ax3.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch3')
    if pdf:
        pp.savefig(fig)
    if pic:
        plt.show()
    plt.close(fig)
    
    fig, ax4 = plt.subplots(dpi=200)
    for ab in tqdm(range(0,len(v_ch4))):
        ax4.plot(x*dt*1e9, v_ch4[ab])
    # ax1.set_xlim(left=0,right=30)
    # ax1.set_ylim(bottom=0.20,top=0.50)
    ax4.grid()
    ax4.set(xlabel='Time(ns)', ylabel='Voltage(V)',
           title='Raw Data ch4')
    if pdf:
        pp.savefig(fig)
    if pic:
        plt.show()
    plt.close(fig)

##### signal processing for all channels #############
def plotting_job(afile, scope_config, outfile):
    from matplotlib.backends.backend_pdf import PdfPages
    tc = scope_config['transcond']['lowgain']
    tcucsc = scope_config['transcond']['UCSC']
    data, attrs = extract_dataset(afile)    
    pp = PdfPages(outfile)
    trigger_t0s = None
    measure_t0s = {}
    
    maxv_ch1 = np.max(calculate_voltages(data[0], gain_post=np.sign(scope_config['gains'][0])), axis=-1)
    maxv_ch2 = np.max(calculate_voltages(data[1], gain_post=np.sign(scope_config['gains'][1])), axis=-1)
    maxv_ch3 = np.max(calculate_voltages(data[2], gain_post=np.sign(scope_config['gains'][2])), axis=-1)
    maxv_ch4 = np.max(calculate_voltages(data[3], gain_post=np.sign(scope_config['gains'][3])), axis=-1)
    
    ch1 = (maxv_ch1 > 0.050) & (maxv_ch1 < 0.272)
    ch2 = (maxv_ch2 > 0.010) & (maxv_ch2 < 0.272)
    ch3 = (maxv_ch3 > 0.030) & (maxv_ch3 < 0.272)
    ch4 = (maxv_ch4 > 0.030) & (maxv_ch3 < 0.272)

    mask = (ch1 & ch2 & ch3 & ch4)
    t0s_simple = []
    for ch in range(4):        
        if attrs['chmask'][ch]:            
            fig, ax = plt.subplots(dpi=400)
            thegain = scope_config['gains'][ch]
            plot_amplitude(data[:,mask,:], ch, ax=ax, gain_post=np.sign(thegain))
            pp.savefig(fig)
            plt.close(fig)

            fig, ax = plt.subplots(dpi=400)
            plot_charge(data[:,mask,:], ch, attrs['dt'], tcucsc if ch != 1 else tc, ax=ax, gain_post=thegain)
            pp.savefig(fig)
            plt.close(fig)
            
            t0s_simple.append(calculate_time(data[ch], 
                                             attrs['dt'], 
                                             th_cfd=scope_config['thresholds'][ch], 
                                             tdc_bin=0.005, 
                                             tdc_start = 40))
            
            t0s = calculate_tcross(data[ch], scope_config['thresholds'][ch], 
                                    attrs['dt'], gain_post=thegain)
            #fig, ax = plt.subplots(dpi=400)
            #plot_tcross(ch, t0s, ax=ax)            
            #pp.savefig(fig)
            #plt.close(fig)
            if ch == scope_config['trigger']:
                trigger_t0s = t0s
            else:
                measure_t0s[ch] = t0s
        
    fit_plot_range = (-1.500, 1.500)
    nbins = 250

    ################ t1 (reference UCSC-21), t2 (ETROC0 board), t3 (reference UCSC-24) ################
    ## t0s_simple[0][0]: ch1, t0s_simple[1][0]: ch2, t0s_simple[2][0]: ch3
    fig, ax = plt.subplots(1, 1, dpi=400)
    tch21_avg = (t0s_simple[1][0] - t0s_simple[0][0])[mask]
    tch21_mean = np.mean(tch21_avg)
    tch21_sigma = np.std(tch21_avg, ddof=1)
    bins, edges = np.histogram(tch21_avg, nbins, range=fit_plot_range,  density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    try:
        popt, pcov = curve_fit(gaus,centers,bins,p0=[1,tch21_mean,tch21_sigma])
        ax.plot(centers, gaus(centers,popt[0], popt[1], popt[2]))
        tch21_mean = popt[1]
        tch21_sigma = abs(popt[2])
    except:
        pass
    ax.hist(tch21_avg, nbins, range=fit_plot_range, 
            density=False,
            label='mean = %.3g ns\nsigma = %.3g ns\n#event = %d\n#bin = %d'%(tch21_mean,tch21_sigma,tch21_avg.size, nbins))
    ax.legend()
    ax.set(xlabel='t_2 - t_1  (ns)', ylabel='Counts',
           title='TOA for CH2 vs CH1')
    pp.savefig(fig)
    plt.close(fig)

    print('21 from fit', tch21_sigma*1e12)
    #tch21_sigma = norm.fit(tch21_avg)[1]
    print('21 from refit', tch21_sigma*1e12)

    fig, ax = plt.subplots(1, 1, dpi=400)
    tch32_avg = (t0s_simple[2][0] - t0s_simple[1][0])[mask]
    tch32_mean = np.mean(tch32_avg)
    tch32_sigma = norm.fit(tch32_avg)[1]
    bins, edges = np.histogram(tch32_avg, nbins, range=fit_plot_range, density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    try:
        popt, pcov = curve_fit(gaus,centers,bins,p0=[1,tch32_mean,tch32_sigma])
        ax.plot(centers, gaus(centers,popt[0], popt[1], popt[2]))
        tch32_mean = popt[1]
        tch32_sigma = abs(popt[2])
    
    except:
        pass
    ax.hist(tch32_avg, nbins, range=fit_plot_range, 
            density=False,
            label='mean = %.3g ns\nsigma = %.3g ns\n#event = %d\n#bin = %d'%(tch32_mean,tch32_sigma,tch32_avg.size,nbins))
    ax.legend()
    ax.set(xlabel='t_3 - t_2 (ns)', ylabel='Counts',
           title='TOA for CH2 vs CH3')
    pp.savefig(fig)
    plt.close(fig)

    print('32 from fit', tch32_sigma*1e12)
    print('32 from refit', tch32_sigma*1e12)

    fig, ax = plt.subplots(1, 1, dpi=400)
    tch31_avg = (t0s_simple[2][0] - t0s_simple[0][0])[mask]
    tch31_mean = np.mean(tch31_avg)
    tch31_sigma = np.std(tch31_avg, ddof=1)
    bins, edges = np.histogram(tch31_avg, nbins, range=fit_plot_range, density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    try:
        popt, pcov = curve_fit(gaus,centers,bins,p0=[1,tch31_mean,tch31_sigma])
        ax.plot(centers, gaus(centers,popt[0], popt[1], popt[2]))
        tch31_mean = popt[1]
        tch31_sigma = abs(popt[2])
    except:
        pass
    ax.hist(tch31_avg, nbins, range=fit_plot_range, 
            density=False,
            label='mean = %.3g ns\nsigma = %.3g ns\n#event = %d\n#bin = %d'%(tch31_mean,tch31_sigma,tch31_avg.size,nbins))
    ax.legend()
    ax.set(xlabel='t_3 - t_1 (ns)', ylabel='Counts',
           title='TOA for CH1 vs CH3')
    pp.savefig(fig)
    plt.close(fig)
    
    print('31 from fit',tch31_sigma*1e12)
    print('31 from refit', tch31_sigma*1e12)

    fig, ax = plt.subplots(1, 1, dpi=400)
    tch2_avg = (0.5*(t0s_simple[0][0]+t0s_simple[2][0]) - t0s_simple[1][0])[mask]
    tch2_mean = np.mean(tch2_avg)
    tch2_sigma = np.std(tch2_avg, ddof=1)
    print('sigma_231 basic', tch2_sigma*1e12)
    bins, edges = np.histogram(tch2_avg, nbins, range=fit_plot_range,  density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    try:
        popt, pcov = curve_fit(gaus,centers,bins,p0=[1,tch2_mean,tch2_sigma])
        ax.plot(centers, gaus(centers,popt[0], popt[1], popt[2]))
        tch2_mean = popt[1]
        tch2_sigma = abs(popt[2])
    except:
        pass

    print('sigma_231 after fit', tch2_sigma*1e12)

    bayes_info = scipy.stats.bayes_mvs(tch2_avg, alpha=0.68)
    print('231', np.std(tch2_avg, ddof=1), bayes_info[2][0])
    print('231\'', scipy.stats.bayes_mvs(tch2_avg[np.abs(tch2_avg - bayes_info[0][0]) < 5*bayes_info[2][0]]))
    

    sigma_sens_2 = np.sqrt(0.5*(tch21_sigma**2 - tch31_sigma**2 + tch32_sigma**2))

    ax.hist(tch2_avg, nbins, range=fit_plot_range, 
            density=False,
            label='sigma = %.3g ns\nCH2 Jitter = %.3g ns\n#event = %d\n#bin = %d'%(tch2_sigma, sigma_sens_2, tch2_avg.size,nbins))
    ax.legend()
    ax.set(xlabel='0.5*(t_1 + t_3) - t_2 (ns)', ylabel='Counts',
           title='TOA for CH2 vs average of CH1+CH3')
    pp.savefig(fig)
    plt.close(fig)

    ################ t1 (reference UCSC-21), t4 (UCSC), t3 (reference UCSC-24) ################
    ## t0s_simple[0][0]: ch1, t0s_simple[3][0]: ch4, t0s_simple[2][0]: ch3
    fig, ax = plt.subplots(1, 1, dpi=400)
    tch41_avg = (t0s_simple[3][0] - t0s_simple[0][0])[mask]
    tch41_mean = np.mean(tch41_avg)
    tch41_sigma = np.std(tch41_avg, ddof=1)
    bins, edges = np.histogram(tch41_avg, nbins, range=fit_plot_range,  density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    try:
        popt, pcov = curve_fit(gaus,centers,bins,p0=[1,tch41_mean,tch41_sigma])
        ax.plot(centers, gaus(centers,popt[0], popt[1], popt[2]))
        tch41_mean = popt[1]
        tch41_sigma = abs(popt[2])
    except:
        pass
    ax.hist(tch41_avg, nbins, range=fit_plot_range, 
            density=False,
            label='mean = %.3g ns\nsigma = %.3g ns\n#event = %d\n#bin = %d'%(tch41_mean,tch41_sigma,tch41_avg.size, nbins))
    ax.legend()
    ax.set(xlabel='t_4 - t_1  (ns)', ylabel='Counts',
           title='TOA for CH4 vs CH1')
    pp.savefig(fig)
    plt.close(fig)

    print('41 from fit', tch41_sigma*1e12)
    print('41 from refit', tch41_sigma*1e12)

    fig, ax = plt.subplots(1, 1, dpi=400)
    tch34_avg = (t0s_simple[2][0] - t0s_simple[3][0])[mask]
    tch34_mean = np.mean(tch34_avg)
    tch34_sigma = norm.fit(tch34_avg)[1]
    bins, edges = np.histogram(tch34_avg, nbins, range=fit_plot_range, density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    try:
        popt, pcov = curve_fit(gaus,centers,bins,p0=[1,tch34_mean,tch34_sigma])
        ax.plot(centers, gaus(centers,popt[0], popt[1], popt[2]))
        tch34_mean = popt[1]
        tch34_sigma = abs(popt[2])
    
    except:
        pass
    ax.hist(tch34_avg, nbins, range=fit_plot_range, 
            density=False,
            label='mean = %.3g ns\nsigma = %.3g ns\n#event = %d\n#bin = %d'%(tch34_mean,tch34_sigma,tch34_avg.size,nbins))
    ax.legend()
    ax.set(xlabel='t_3 - t_4 (ns)', ylabel='Counts',
           title='TOA for CH4 vs CH3')
    pp.savefig(fig)
    plt.close(fig)

    print('34 from fit', tch34_sigma*1e12)
    print('34 from refit', tch34_sigma*1e12)

    fig, ax = plt.subplots(1, 1, dpi=400)
    tch4_avg = (0.5*(t0s_simple[0][0]+t0s_simple[2][0]) - t0s_simple[3][0])[mask]
    tch4_mean = np.mean(tch4_avg)
    tch4_sigma = np.std(tch4_avg, ddof=1)
    print('sigma_431 basic', tch4_sigma*1e12)
    bins, edges = np.histogram(tch4_avg, nbins, range=fit_plot_range,  density=False)
    centers = 0.5*(edges[1:] + edges[:-1])
    try:
        popt, pcov = curve_fit(gaus,centers,bins,p0=[1,tch4_mean,tch4_sigma])
        ax.plot(centers, gaus(centers,popt[0], popt[1], popt[2]))
        tch4_mean = popt[1]
        tch4_sigma = abs(popt[2])
    except:
        pass

    print('sigma_431 after fit', tch4_sigma*1e12)

    bayes_info = scipy.stats.bayes_mvs(tch4_avg, alpha=0.68)
    print('431', np.std(tch4_avg, ddof=1), bayes_info[2][0])
    print('431\'', scipy.stats.bayes_mvs(tch4_avg[np.abs(tch4_avg - bayes_info[0][0]) < 5*bayes_info[2][0]]))

    sigma_sens_4 = np.sqrt(0.5*(tch41_sigma**2 - tch31_sigma**2 + tch34_sigma**2))

    ax.hist(tch4_avg, nbins, range=fit_plot_range,
            density=False,
            label='sigma = %.3g ns\nCH4 Jitter = %.3g ns\n#event = %d\n#bin = %d'%(tch4_sigma, sigma_sens_4, tch4_avg.size,nbins))
    ax.legend()
    ax.set(xlabel='0.5*(t_1 + t_3) - t_4 (ns)', ylabel='Counts',
           title='TOA for CH4 vs average of CH1+CH3')
    pp.savefig(fig)
    plt.close(fig)

    nch = len(measure_t0s.keys())
    iax = 0
    for ch, t0s in measure_t0s.items():
        fig, ax = plt.subplots(1, 1, dpi=400)
        plot_tcross(ch, t0s, trigger_t0s=trigger_t0s, ax=ax)
        pp.savefig(fig)
        plt.close(fig)
        iax += 1
    pp.close()

gain_post = -10.0
scope_config = {'trigger': 2,
                'transcond':{'highgain': 15.7e3*0.75, 'lowgain': 4.4e3*0.75, 'UCSC': 4.7e2}, ## 0.75 is a scale factor for ETROC0 v2 sensor board
                'gains': [gain_post, gain_post, gain_post, gain_post],
                'thresholds': [0.5, 0.5, 0.5, 0.5]}

import sys
import glob
import time

data_path = sys.argv[1]
print(data_path.split('/'))
out_path = '/'.join(data_path.split('/')[:-1])
if len(sys.argv) > 2:
    out_path = sys.argv[2]
print('out path ->', out_path)
processed_files = set()

while True:
    files = set(glob.glob(os.path.join(data_path,'*.hdf5')))
    files_to_process = files - processed_files
    if len(files_to_process):
        for afile in tqdm(files_to_process):            
            outfile = afile[:afile.rfind('.')].split('/')[-1] + '.pdf'
            fname = os.path.join(out_path, outfile)
            plotting_job(afile, scope_config, fname)
            processed_files.add(afile)
    else:
        time.sleep(1)
