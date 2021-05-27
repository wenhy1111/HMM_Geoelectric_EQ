#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------- NEW VRESION! --------
Created at 27 Feb 2019
EP (Earthquake Project) module for data preprocessing
Load data format: D_LIOQ_mid_ti60s_cln5%%.pickle
feature from old verison: 
    data cleaning obsoleted in module
    cross_corr, half_gauss
    
Copyed from GEMS_GD to GEMS_GD_new on 26 Jul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import datetime
#from PyEMD import EEMD
import matplotlib.dates as pdate
from scipy.stats import skew
from scipy.stats import kurtosis

dft_t_quake = datetime.datetime(2016, 2, 6, 3, 57)
dft_num_quake = pdate.date2num(dft_t_quake)
dft_paras = {'RW_len': 240, 
         'RW_step': 120,
         'exam_W_UB':3000}


class eqk():
    
    def __init__(self, name='tweqk'):
        with open(name+'.pickle', 'rb') as f:
            data_pack = pickle.load(f)
        self.mat = data_pack['mat']
        self.lld = data_pack['lld']
        self.position = dict() # lattitude, longitude
        self.position['LIOQ'] = np.array([23.0321, 120.6632])
        self.position['CHCH'] = np.array([23.2197, 120.1618])
        self.position['SHRL'] = np.array([25.1559, 121.5619])
        self.position['DABA'] = np.array([23.4544, 120.7494])
        
    def select(self, name, mag, radius = 0.8, verbose = 'yes'):
        self.selected = self.mat.copy()
        mags = self.selected[:,1]
        pos = self.position[name]
        ds = np.sqrt((self.lld[:,0]-pos[0])**2 + (self.lld[0,1]-pos[1])**2)
#        self.selected = self.selected[mags>mag and ds<radius, :]
        self.selected = self.selected[mags>mag, :]
        num_total = self.selected.shape[0]
        ds = ds[mags>mag]
        self.selected = self.selected[ds<radius, :]
#        self.selected = self.selected[, :]
        self.radius = radius
        self.mag = mag
        if verbose == 'yes':
            print('Selected near events %s out of %s total events.' 
                  %(self.selected.shape[0], num_total))
        
    def plot(self):
        for i in range(self.selected.shape[0]):
            plt.axvline(self.selected[i,0], color='k')


# In[] for financial data use
def max_spread(seq):
    # returns maximum spread: the percentage of largest deviaiton in future W
    x_ini = seq[0]
    spread_abs = np.abs(seq-x_ini)
    ind_ms = np.argmax(spread_abs)
    return spread_abs[ind_ms]/x_ini, ind_ms
    
# In[]
class TS_fft():
    def __init__(self, data_full_name, col, t_start=0., t_end=1.):
        self.data_full_name = data_full_name
        self.col = col
        self.t_start = t_start
        self.t_end = t_end
        self.data_name = data_full_name[2:6]
        with open(data_full_name+'.pickle', 'rb') as f:
            data_pack = pickle.load(f)
        try:
            Mat = data_pack['Mat_raw']
        except:
            Mat = data_pack['Mat']
        print('Time format adjusted.') # matlab datenum is 366 ahead of python's
        Mat[:,0] -= 366
        ra = range(int(len(Mat)*t_start), int(len(Mat)*t_end))
        self.series = Mat[ra,col]
        self.t = Mat[ra,0]
        self.t_interval = data_pack['t_interval']
        print('TS length = %s'%len(self.t))
        
    def detrend(self, sigmas):
        if hasattr(sigmas, "__len__"): # check if sigmas is an array
            pass
        else:
            self.sigmas = np.array([sigmas])
        seq = self.series
        self.Ress = np.zeros((len(sigmas), len(seq))) # to store trends 'Ts'
        self.sigmas = sigmas
        for i in range(len(sigmas)):
            Tr = half_gauss(seq, sigmas[i]) # trend 'T'
#            Trs[i, :] = np.reshape(Tr, (1,len(seq)))
            Res = seq - Tr
            Res[np.isnan(Res)] = 0
            self.Ress[i, :] = Res

    def compute_fft(self, paras):
        # in this function, paras['RW_len'] must be single value
        for para in paras:
            exec('self.'+ para +'= paras[para]')
        # compute the indices in the time series as the RW's endtimes, 'inds'
        inds = np.array(range(0,len(self.series)-self.exam_W_UB))
        inds = inds[self.exam_W_UB:len(inds):self.RW_step]
        self.inds = inds
        self.ffts = np.zeros((len(self.sigmas), len(self.inds), int(self.RW_len/2-1)))
        self.t_inds = self.t[inds]
        for j in range(len(self.sigmas)):
            seq = self.Ress[j, :]
            for i in range(len(self.inds)):
                ind = self.inds[i]
                RW = seq[ind-self.RW_len:ind]
                fftv = np.fft.fft(RW)
                fftv = np.abs(fftv[1:int(self.RW_len/2)])
                fftv = fftv/np.sum(fftv)
                self.ffts[j, i, :] = fftv
        
    def plot_fft(self, mag, radius, ratio = 1/4):
        fft_plot = self.ffts.copy()
        fft_plot = fft_plot[:, :, 0:int(ratio*self.ffts.shape[2])]
        fig = plt.figure()
        ax1 = plt.subplot(2,1,1)
        if mag > 0:
            eq = eqk()
            eq.select(self.data_name, mag, radius)
            eq.plot()
            self.mag = mag
            self.eq = eq
            print('%s events of magnitude >= %.1f, radius <= %.1f' 
                      %(eq.selected.shape[0], mag, radius))
#        plt.yticks(np.arange(len(self.RW_lens))+0.5, self.RW_len.astype(str))
        X, Y = np.meshgrid(range(fft_plot.shape[2]), pdate.num2date(self.t_inds))
        p1 = plt.pcolor(Y, X, fft_plot[0,:,:], cmap='jet')
        plt.title('%s'%self.sigmas[0])
        fig.colorbar(p1)
        plt.subplot(2,1,2, sharex = ax1)
        if mag > 0:
            eq.plot()
        X, Y = np.meshgrid(range(fft_plot.shape[2]), pdate.num2date(self.t_inds))
        p2 = plt.pcolor(Y, X, fft_plot[1,:,:], cmap='jet')
        plt.title('%s'%self.sigmas[1])
        fig.colorbar(p2)
        
    def compute_fs(self, FW_lens, mag = 0, radius = 2, plot = 'no'):
        # FW for Future Window, fs for future seismicity
        # function compute fs with various FW along TS marked by self.inds
        self.eq = eqk()
        self.eq.select(self.data_name, mag, radius)
        self.radius = radius
        eqs = self.eq.selected
        self.FW_lens = FW_lens
        self.fs = np.zeros((len(FW_lens), len(self.inds)))
        for i in range(len(FW_lens)):
            FW_len = FW_lens[i]
            for j in range(len(self.inds)):
                FW_t_start = self.t[self.inds[j]]
                try:
                    FW_t_end = self.t[self.inds[j]+FW_len]
                except:
                    FW_t_end = self.t[-1]
                    print('index %s exceeds that of t, %s' 
                          %(self.inds[j]+FW_len, len(self.t)))
                eqs_s = eqs[eqs[:,0]>FW_t_start, :]
                eqs_e = eqs_s[eqs_s[:,0]<FW_t_end, :]
                if len(eqs_e) > 0:
                    self.fs[i,j] = np.max(eqs_e[:,1])
                    
        if plot == 'yes':
            print('plotting fs')
            plot_num = 2
            fig = plt.figure(figsize=(17,9))
            ax1 = plt.subplot(plot_num,1,1)
            ax1.plot(pdate.num2date(self.eq.selected[:,0]), self.eq.selected[:,1])            
            ax2 = plt.subplot(plot_num,1,2, sharex = ax1)
            plt.yticks(np.arange(len(self.FW_lens))+0.5, self.FW_lens.astype(str))
            X, Y = np.meshgrid((self.t_inds), range(len(self.FW_lens)+1))
            p = plt.pcolor(X, Y, self.fs, cmap='jet')
            fig.colorbar(p)#,orientation="horizontal")
            plt.ylabel('fs')
            pos1 = ax1.get_position()
            pos2 = ax2.get_position()
            ax1.set_position([pos1.x0, pos1.y0, pos2.width, pos1.height])
#            ax2.set_position([pos2.x0, pos2.y0, pos3.width, pos2.height])
            
# In[]

def prep4ml(fft1, fft2, fs, feature_len=20, ratio=1/5, p=0.1, eq=None, t_inds=None, 
            info = None, savename = 'no'):
    # function prepares training and testing data for eq ml
    # first, construct flatten feature
    fft1_ratio = fft1[:, :, 0:int(ratio*fft1.shape[2])]
    fft2_ratio = fft2[:, :, 0:int(ratio*fft2.shape[2])]
    fft_all = np.append(fft1_ratio, fft2_ratio, axis=0)
    for i in range(fft_all.shape[0]):
        if i == 0:
            fft_combine = fft_all[0,:,:]
        else:
            fft_combine = np.append(fft_combine, fft_all[i,:,:], axis = 1)
    sample_len = fft_combine.shape[0]
    inds = np.arange(feature_len, sample_len, 1) 
    feature_flat = np.zeros((len(inds), feature_len*fft_combine.shape[1]))
    for i in range(len(inds)):
        ind = inds[i]
        feature_flat[i,:] = np.reshape(fft_combine[ind-feature_len:ind, :], 
                    (feature_len*fft_combine.shape[1], ))
    feature_flat_normalized = feature_flat/np.max(feature_flat)
    # next, prepare future seismicity
    fs_inds = fs[:,inds]
    fs_mean = np.mean(fs_inds, axis=0)
    fs_mean_sorted = np.sort(fs_mean)
    fs_threshold = fs_mean_sorted[int(len(fs_mean)*(1-p))]
    above_threshold = np.zeros(len(fs_mean))
    above_threshold[fs_mean>fs_threshold] = 1
    t_inds_Y = None
    if hasattr(t_inds, "__len__"):
        t_inds_Y = t_inds[inds]
    prepared = {'X': feature_flat_normalized,
                'Y': above_threshold,
                'fft1': fft1, 
                'fft2': fft2, 
                'fs': fs,
                'feature_len': feature_len, 
                'ratio': ratio, 
                'p': p,
                'eq': eq, 
                't_inds_Y': t_inds_Y, 
                't_inds': t_inds,
                'fs_threshold': fs_threshold,
                'info': info}
    if savename != 'no':
        with open(savename+'_%sfts_ratio%.3f_p%s.pickle'%(int(feature_len), ratio, p),
                  'wb') as f:
            pickle.dump(prepared, f)
    return prepared

def preprocess(data, p, feature_len, ratio, X_scale=1.5, verbose=True):
    new = prep4ml(data['fft1'], data['fft2'], data['fs'], feature_len=feature_len, 
                  ratio=ratio, p=p, t_inds = data['t_inds'], info = data['info'], 
                  eq = data['eq'])
    X = new['X']*X_scale
    Y = new['Y']
    if verbose == True:
        plt.plot(X[0,:])
    data_info = new['info']
    data_info['p'] = p
    data_info['feature_len'] = feature_len
    data_info['ratio'] = ratio
    data_info['eq'] = new['eq']
    data_info['t_inds_Y'] = new['t_inds_Y']
    return X, Y, data_info

# In[] maximum spread, future variability

class TS_ms():
    
    def __init__(self, data_full_name, t_start=0., t_end=1.):
        self.data_full_name = data_full_name
        self.t_start = t_start
        self.t_end = t_end
        self.data_name = data_full_name[2:6]
        with open(data_full_name+'.pickle', 'rb') as f:
            data_pack = pickle.load(f)
        try:
            Mat = data_pack['Mat_raw']
        except:
            Mat = data_pack['Mat']
        print('Time format adjusted.') # matlab datenum is 366 ahead of python's
        Mat[:,0] -= 366
        ra = range(int(len(Mat)*t_start), int(len(Mat)*t_end))
        self.series = Mat[ra,1]
        self.t = Mat[ra,0]
        self.t_interval = data_pack['t_interval']
        print('TS length = %s'%len(self.t))
    
    def compute_ms(self, paras):
        # in this function, paras['RW_lens'] must be np array
        for para in paras:
            exec('self.'+ para +'= paras[para]')
        # compute the indices in the time series as the RW's endtimes, 'inds'
        inds = np.array(range(0,len(self.series)-self.exam_W_UB))
        inds = inds[self.exam_W_UB:len(inds):self.RW_step]
        self.inds = inds
        ms = np.zeros((len(self.RW_lens), len(self.inds)))
        self.t_inds = self.t[inds]
        for i in range(len(self.inds)):
            ind = self.inds[i]
            for j in range(len(self.RW_lens)):
                RW_len = self.RW_lens[j]
                RW = self.series[ind:ind+RW_len] # changed from [ind-RW_len:ind]
                ms[j, i], a = max_spread(RW)
        self.ms = ms
        
    def plot_ms(self):
        plot_num = 2
        fig = plt.figure(figsize=(17,9))
        ax1 = plt.subplot(plot_num,1,1)
        ax1.plot(pdate.num2date(self.t), self.series)
        ax2 = plt.subplot(plot_num,1,2, sharex = ax1)
        plt.yticks(np.arange(len(self.RW_lens))+0.5, self.RW_lens.astype(str))
        X, Y = np.meshgrid((self.t_inds), range(len(self.RW_lens)+1))
        p = plt.pcolor(X, Y, self.ms, cmap='jet')
        fig.colorbar(p)#,orientation="horizontal")
        plt.ylabel('ms')
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax1.set_position([pos1.x0, pos1.y0, pos2.width, pos1.height])
        
    def plot_ms_hist(self, bins=np.arange(50)*0.05, plot = 'pdf'):
        plt.figure(figsize=(17,9))
        total_len = self.ms.shape[1]
        bins_plot = bins[1:len(bins)]
        self.bins_plot = bins_plot
        self.pdfs = np.zeros((self.ms.shape[0], len(bins_plot)))
        self.cdfs = np.zeros((self.ms.shape[0], len(bins_plot)))
#        one_s = np.ones(total_len)
        for i in range(self.ms.shape[0]):
            ms_seq = self.ms[i,:]
            pdf, b = np.histogram(ms_seq, bins)
            pdf = pdf/total_len
            cdf = np.cumsum(pdf)
#            cdf = cdf[1:len(cdf)]
#            cdf = np.append(cdf, cdf[-1]+np.sum(one_s[ms_seq>bins_plot[-1]])/total_len)
            self.cdfs[i,:] = cdf
            self.pdfs[i, :] = pdf
            if plot == 'pdf':
                plt.plot(bins_plot*100, pdf, label= '%s'%self.RW_lens[i])
            elif plot == 'cdf':
                plt.plot(bins_plot*100, cdf, label= '%s'%self.RW_lens[i])
            plt.legend()
            plt.xlabel('ms (%)')
            plt.ylabel('p')
        
    def compute_p(self, plot = 'yes'):
        '''
        compute p values values at all RW_lens (ps), and summarize them by 
        with one p value (their mean, in this case). 
        '''
        num_RWs = len(self.RW_lens)
        self.ps = np.zeros((num_RWs, self.ms.shape[1]))
        for i in range(num_RWs):
            cdfi = self.cdfs[i,:]
            msi = self.ms[i,:]
            for j in range(self.ms.shape[1]):
                self.ps[i,j] = 1-cdfi[np.sum(self.bins_plot < msi[j])]
        if plot == 'yes':
            print('plotting')
            plot_num = 3
            fig = plt.figure(figsize=(17,9))
            ax1 = plt.subplot(plot_num,1,1)
            ax1.plot(pdate.num2date(self.t), self.series)
            plt.subplot(plot_num, 1, 2, sharex = ax1)
            ax2 = plt.plot(pdate.num2date(self.t_inds), np.sum(self.ps, axis = 0)/num_RWs)
            plt.ylabel('average p')
            ax3 = plt.subplot(plot_num,1,3, sharex = ax1)
            plt.yticks(np.arange(len(self.RW_lens))+0.5, self.RW_lens.astype(str))
            X, Y = np.meshgrid((self.t_inds), range(len(self.RW_lens)+1))
            p = plt.pcolor(X, Y, self.ps, cmap='jet')
            fig.colorbar(p)#,orientation="horizontal")
            plt.ylabel('p_m_s')
            pos1 = ax1.get_position()
            pos2 = ax2.get_position()
            pos3 = ax3.get_position()
            ax1.set_position([pos1.x0, pos1.y0, pos3.width, pos1.height])
            ax2.set_position([pos2.x0, pos2.y0, pos3.width, pos2.height])
    

# In[] utility function
def conv_digits(num):
    """
    convert num in (0,1) to string-formatted up-to-3 significant digits for saving
    such as 0.31 to '31', 0.589 to '589', 0.1 to '10', 0.001 to 001, 0.01 to '01'
    """
    N = num*1000
    if N%10 == 0:
        return str(int((N-N%100)/100))+str(int((N%100-N%10)/10))
#    str(int(num*100))
    else:
        return str(int((N-N%100)/100))+str(int((N%100-N%10)/10))+str(int(N%10))
    
# In[]
'''
------------------------ old ones below ------------------------
'''
# In[]
def pct_ylim(y, pct=1, expand=0.12):
    y_sorted = np.sort(y)
    threshold_ind = int(np.round(pct*len(y)/100))
    lower = y_sorted[threshold_ind]
    upper = y_sorted[len(y)-threshold_ind]
    height = upper-lower
    ylims = [lower-height*expand, upper+height*expand]
    return ylims

# In[] extract memory decay by 2 points: first (global) min, and next (local) max

def analize_arf(ars):
    inds = np.array(range(len(ars)))
    A_min = np.min(ars)
    t_min = inds[ars == A_min]
    t_max = 0
    for i in range(int(t_min[0]),len(ars)-1):
        if ars[i+1] <= ars[i]:
            t_max = i+1
            break
    if t_max != 0:
        A_max = ars[t_max]
        memory = (np.log(-A_min)-np.log(A_max))/(t_max-t_min[0])
    else:
        A_max = np.nan
        memory = np.nan
        t_max = np.nan
#    print('tmin=%.1f, Amin=%.4f, tamx=%.1f, Amax=%.4f.' %(t_min, A_min, t_max, A_max))
    return memory, (t_min, A_min, t_max, A_max)

# In[] extract memory decay be fitting with oscillatory decay formula    
from scipy.optimize import least_squares

def osci_decay2(p, t, y):
    res = p[0]*np.exp(-p[1]*t)*np.cos(p[2]*t+p[3])-y
    return res

def osci_decay0(p, t):
    ar = p[0]*np.exp(-p[1]*t)*np.cos(p[2]*t+p[3])
    return ar

def fit_arf2(ars):
    inds = np.array(range(len(ars)))
    A_min = np.min(ars)
    t_min = inds[ars == A_min]
    t_max = 0
    for i in range(int(t_min[0]),len(ars)-1):
        if ars[i+1] <= ars[i]:
            t_max = i+1
            break
    if t_max == 0:
        t_max = len(ars)
    t_bound = int(min(max(5, t_max)+1, len(ars)))
    fit_ars = ars[0:t_bound]
    fit_inds = inds[0:t_bound]
    x0 = np.ones(4) 
    fit = least_squares(osci_decay2, x0,ftol=1e-09, args = (fit_inds, fit_ars))
    plt.plot(inds, ars)
    plt.plot(fit_inds, osci_decay0(fit.x, fit_inds), 'r-',
             label='fit: lambda = %5.3f, A = %5.3f, omega = %5.3f' 
             % (fit.x[1], fit.x[0], fit.x[2]))
    plt.legend()
    return fit

# In[] class Res_TS()
class Res_TS(): # features: AR, Var, Skw, Krt, fft(optional)
    
    def __init__(self, Res, paras = dft_paras):
        self.Res = Res
        for para in paras:
            exec('self.'+ para +'= paras[para]')
        # compute the indices in the time series as the RW's endtimes, 'inds'
        inds = np.array(range(0,len(Res)-self.exam_W_UB))
        inds = inds[self.RW_len:len(inds):self.RW_step]
        self.inds = inds
        
    def compute_EWI(self, ar = 2, var = 'yes', skw = 'no', krt = 'no',
                    fft = 'no', fft_ratio = 1/4):
        """
        function computes EWIs: ar, var, skw, krt, fft from 1D residue time series 
        default setting: only 3 EWIs: ar1, ar2, var
        input format:
            ar: integer 1, 2, 3, ...
            other EWIs: 'yes' or 'no' only
            fft_ratio: number in (0, 1)
        """
        Res = pd.Series(self.Res)
        for num in range(ar):
            exec('AR'+ str(num+1) +'s = np.zeros(len(self.inds))')
        if var == 'yes':
            Vars = np.zeros(len(self.inds))
        elif var != 'no':
            raise ValueError("Input var: 'yes' or 'no' only!")
        if skw == 'yes':
            Skws = np.zeros(len(self.inds))
        elif skw != 'no':
            raise ValueError("Input skw: 'yes' or 'no' only!")
        if krt == 'yes':
            Krts = np.zeros(len(self.inds))
        elif krt != 'no':
            raise ValueError("Input krt: 'yes' or 'no' only!")
        if fft == 'yes':
            self.fft_ratio = fft_ratio
            ffts = np.zeros((len(self.inds), int(np.floor(self.RW_len*self.fft_ratio))))
        elif fft != 'no':
            raise ValueError("Input fft: 'yes' or 'no' only!")
        # compute
        count = 0
        for i in self.inds:
            Res_RW = Res[i-self.RW_len:i]
            for num in range(ar):
                exec('AR'+ str(num+1) +'s[count] = Res_RW.autocorr(lag=' +str(num+1)+')')
            if var == 'yes':
                Vars[count] = Res_RW.var()
            if skw == 'yes':
                Skws[count] = Res_RW.skew()
            if krt == 'yes':
                Krts[count] = Res_RW.kurtosis()
            if fft == 'yes':
                fftv = np.fft.fft(Res_RW)
                ffts[count, :] = fftv[0:int(np.floor(len(fftv)*self.fft_ratio))].real
            count = count + 1
        # save
        for num in range(ar):
            exec('self.AR'+ str(num+1) +'s = AR'+ str(num+1) +'s')
        if var == 'yes':
            self.Vars = Vars
        if skw == 'yes':
            self.Skws = Skws
        if krt == 'yes':
            self.Krts = Krts
        if fft == 'yes':
            self.ffts = ffts
            
    def plot_EWS(self, t_ind, quake = np.nan, ar = 1, var = 'no', skw = 'no', 
                 krt = 'no', newfig = 'yes'):
        if newfig == 'yes':
            plt.figure()
        plot_num = 1
        if var == 'yes':
            plot_num += 1
            var_done = 0
        if skw == 'yes':
            plot_num += 1
            skw_done = 0
        if krt == 'yes':
            plot_num += 1
            krt_done = 0
        # start plotting
        for i in range(plot_num):
            if i == 0: # special case for ar
                exec('ax1 = plt.subplot('+ str(plot_num) +',1,1)')
                for j in range(ar):
                    exec('ax1.plot(t_ind, self.AR'+str(j+1)+'s, label = "AR'+str(j+1)+'")')
                    exec('ax1.legend()')
                exec('ax'+ str(i+1) +'.set_ylabel("Ar")')
                if np.isnan(quake) == 0:
                    exec('ax'+ str(i+1) +'.axvline(quake, color="k")')
            else:
                exec('ax'+ str(i+1) +' = plt.subplot('+ 
                     str(plot_num) +',1,'+str(i+1)+', sharex=ax1)')
                if var == 'yes' and var_done == 0:
                    exec('ax'+ str(i+1) +'.plot(t_ind, self.Vars)')
                    exec('ax'+ str(i+1) +'.set_ylabel("Var")')
                    var_done += 1
                elif skw == 'yes' and skw_done == 0:
                    exec('ax'+ str(i+1) +'.plot(t_ind, self.Skws)')
                    exec('ax'+ str(i+1) +'.set_ylabel("Skew")')
                    skw_done += 1
                elif krt == 'yes' and krt_done == 0:
                    exec('ax'+ str(i+1) +'.plot(t_ind, self.Krts)')
                    exec('ax'+ str(i+1) +'.set_ylabel("Kurtosis")')
                    krt_done += 1
                if np.isnan(quake) == 0:
                    exec('ax'+ str(i+1) +'.axvline(quake, color="k")')

# In[] class IMFs()
def plot_TS(IMFs):
    t = pdate.num2date(IMFs.t)
    plt.figure(figsize=(15,9))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t, IMFs.raw_series)
    ax1.set_title('original TS')
    plt.subplot(2,1,2, sharex = ax1)
    plt.plot(t, IMFs.series)
    plt.title('cleaned TS')


# In[] class IMFs()
class IMFs(): # features
    
    def __init__(self, data_full_name, col, t_start, t_end, n_IMF=6, method='EMD'):
        """
        col: column in Mat
        n_IMF: maximum number of IMFs to compute
        methods: choose between EEMD and EMD
        """
        self.data_full_name = data_full_name
        self.col = col
        self.t_start = t_start
        self.t_end = t_end
        self.n_IMF = n_IMF
        self.method = method
        with open(data_full_name+'.pickle', 'rb') as f:
            data_pack = pickle.load(f)
        Mat = data_pack['Mat']
        print('Time format adjusted.') # matlab datenum is 366 ahead of python's
        Mat[:,0] -= 366
        ra = range(int(len(Mat)*t_start), int(len(Mat)*t_end))
        self.series = Mat[ra,col]
        self.t = Mat[ra,0]
        self.t_interval = data_pack['t_interval']
        self.EWI_config = None
        self.EWI_config = dict()
    
    def decompose(self, save='no'):
        """
        function to do EMD systematically
        """
        if save == 'no':
            print("warning: not saving results")
        tic = time.time()
        series = self.series # choose data in range
        t = self.t
        method = self.method
        if method == 'EEMD':
            IMF = EEMD().eemd(series,t,max_imf=self.n_IMF-1) # gives max_imf+1 IMFs
        elif method == 'EMD':
            IMF = EEMD().emd(series,t,max_imf=self.n_IMF-1) # gives max_imf+1 IMFs
        else:
            raise ValueError("Input method: 'EMD' or 'EEMD' only!")
        self.elapsed_decompose = time.time() - tic
        self.IMF = IMF
        info_IMF = {'IMFs': self}
        if save == 'yes':
            with open(self.data_full_name+'_'+method+'_'+conv_digits(self.t_start)+'_'
                      +conv_digits(self.t_end)+'%_'+str(IMF.shape[0])+
                      'IMFs_col'+str(self.col)+'.pickle','wb') as f:
                pickle.dump(info_IMF, f)
        
    def compute_EWIs(self,para,n_skip=0,ar=1,var='no',skw='no',krt='no',cumu='no'):
        self.EWI_config["para"] = para # parameter for RWs
        self.EWI_config["n_skip"] = n_skip # number of IMF skipped for computing EWI
        self.EWI_config["ar"] = ar
        self.EWI_config["var"] = var
        self.EWI_config["skw"] = skw
        self.EWI_config["krt"] = krt
        self.EWI_config["cumu"] = cumu
        EWIs = list()
        tic = time.time()
        for i in range(max(4, self.IMF.shape[0]-n_skip)):
            if cumu == 'no':
                EWIs.append(Res_TS(self.IMF[i, :], paras = para))
            elif cumu == 'yes':
                EWIs.append(Res_TS(np.sum(self.IMF[0:i+1,:], axis=0), paras = para))
            else:
                raise ValueError("Input cumu: 'yes' or 'no' only!")
            EWIs[i].compute_EWI(ar = ar, var = var, skw = skw, krt = krt)
            compute_EWI_elapsed = time.time() - tic
            print(str(int(compute_EWI_elapsed))+'s elapsed, completed EWI for IMF'
                  +str(i+1)+'/'+str(max(4, self.IMF.shape[0]-n_skip)))
#        self.t_EWI = self.t[np.reshape(EWIs[0].inds, (len(EWIs[0].inds), 1))]
        self.t_EWI = self.t[EWIs[0].inds]
        self.EWIs = EWIs
        
    def plot_EWIs(self, t_quake = 0, EWI = 'AR1s', save_fig = 'no'):
        #EWI can be 'AR1s', 'Vars', 'Skws', 'Krts'
        plt.figure(figsize=(15,9))
        EWIs = self.EWIs
        t = pdate.num2date(self.t_EWI)
        for i in range(len(self.EWIs)):
            if i == 0:
                ax1 = plt.subplot(len(EWIs), 1, i+1)
                exec('ax1.plot(t, EWIs[i].'+ EWI +')')
                ax1.set_title(EWI)
                ax1.set_ylabel('IMF'+str(i+1))
            else:
                plt.subplot(len(EWIs), 1, i+1, sharex = ax1)
                exec('plt.plot(t, EWIs[i].'+ EWI +')')
                plt.ylabel('IMF'+str(i+1))
            if t_quake != 0:
                plt.axvline(t_quake, color="k")
        plt.show()
        if save_fig == 'yes':
            if self.EWI_config["cumu"] == 'no':
                plt.savefig(self.data_full_name+'_'+self.method+'_'+
                            conv_digits(self.t_start)+'_'+conv_digits(self.t_end)+
                            '%_'+str(self.IMF.shape[0])
                            +'IMFs_col'+str(self.col)+'_EWIplot.png', dpi=220)
            elif self.EWI_config["cumu"] == 'yes':
                plt.savefig(self.data_full_name+'_'+self.method+'_'+
                            conv_digits(self.t_start)+'_'+conv_digits(self.t_end)+
                            '%_'+str(self.IMF.shape[0])
                            +'IMFs_col'+str(self.col)+'_EWI(cumuIMF)plot.png', dpi=220)
        
    def plot_IMFs(self, n_ar=50, t_quake = 0, save_fig='no', cumu='no'):
        all_ars = np.zeros((self.IMF.shape[0],n_ar))
        t = pdate.num2date(self.t)
        plt.figure(figsize=(16,9))
        ax1 = plt.subplot(self.IMF.shape[0]+1,2,1) 
        ax1.plot(t, self.series)
        if cumu == 'no':
            ax1.set_title('Individual IMFs column'+
                          str(self.col)+', TI = %.1f min' %(self.t_interval/60))
        elif cumu == 'yes':
            ax1.set_title('Cumulative IMFs column'+str(self.col)+
                          ', TI = %.1f min' %(self.t_interval/60))
        else:
            ValueError("Input cumu: 'yes' or 'no' only!")
        for j in range(self.IMF.shape[0]):
            if cumu == 'no':
                seq = pd.Series(self.IMF[j,:])
            elif cumu == 'yes':
                seq = pd.Series(np.sum(self.IMF[0:j+1,:], axis=0))
            ars = np.zeros(n_ar)
#            lag_time = (np.array(range(n_ar))+1)*self.t_interval/60
            for i in range(n_ar):
                ars[i] = seq.autocorr(lag=i+1)
#            memory, info = analize_arf(ars)
            plt.subplot(self.IMF.shape[0]+1,2,j*2+3, sharex = ax1) 
            plt.plot(t, seq)
            if t_quake != 0:
                plt.axvline(t_quake, color="k")
            plt.subplot(self.IMF.shape[0]+1,2,j*2+4)
#            plt.plot(ars) 
            fit_result = fit_arf2(ars)
            plt.ylim(-0.2, 1)
#            plt.xlim(1, 30)
            all_ars[j,:] = np.reshape(ars, (1, n_ar))
        self.all_ars = all_ars
        self.fit_result = fit_result
        plt.xlabel('lag')
        if 'para' in self.EWI_config.keys():
            para = self.EWI_config['para']
            print('RW length = %.1f hr' %(para['RW_len']*self.t_interval/60/60))
        if save_fig == 'yes':
            if cumu == 'no':
                plt.savefig(self.data_full_name+'_'+self.method+'_'+
                            conv_digits(self.t_start)+'_'+conv_digits(self.t_end)+
                            '%_'+str(self.IMF.shape[0])
                            +'IMFs_col'+str(self.col)+'_IMFplots.png', dpi=220)
            elif cumu == 'yes':
                plt.savefig(self.data_full_name+'_'+self.method+'_'+
                            conv_digits(self.t_start)+'_'+conv_digits(self.t_end)+
                            '%_'+str(self.IMF.shape[0])
                            +'IMFs_col'+str(self.col)+'_cumuIMFplots.png', dpi=220)

# In[]
def cross_corr(seq, sigmas):
    '''
    if sigmas == 0:
        computes trend-free memory: cross correlation between {D_i+1} and {D_i} 
    else:
        (sigmas is positive np array or single float/integer)
        function computes the cross correlation between {D_i+1} and {R_i}, which is  
        the residue TS detrended with gaussian bandwidth sigma, inputed by 'sigmas'
    '''
    if hasattr(sigmas, "__len__"): # check if sigmas is an array
        pass
    else:
        sigmas = np.array([sigmas])
    if sigmas[0] == 0:
        Cs = np.zeros(1)
        D =  pd.Series(np.diff(seq))
        Cs[0] = D.autocorr(lag=1)
    else:
        Trs = np.zeros((len(sigmas), len(seq))) # to store trends 'Ts'
        Cs = np.zeros((len(sigmas)))
        for i in range(len(sigmas)):
            Tr = half_gauss(seq, sigmas[i]) # trend 'T'
            Trs[i, :] = np.reshape(Tr, (1,len(seq)))
            Res = seq - Tr
            Res[np.isnan(Res)] = 0
            Res = pd.Series(Res[0:len(Res)-2]) # remove its last element
            D =  pd.Series(np.diff(seq))
            Cs[i] = Res.corr(D, method='pearson')
    return Cs

# In[]
def half_gauss(seq, bandwidth, plot = 'no'):
    '''
    function applies a one-sided 3-sigma gaussian filter to seq
    returns the trend, T
    bandwidth: the len the filter
    '''
    sigma = bandwidth/3 # so that the filter range : [-3sigma, 0]
    bandwidth = int(bandwidth)
    x = np.arange(-bandwidth,0)+1
    y = 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-x**2/(2*sigma**2))
    y = y/np.sum(y)
    T = np.zeros((len(seq)))
    for i in range(len(T)):
        if i+1 <= bandwidth:
            if i == 0:
                T[i] = seq[i]
            else: # i+1 is the length of filter window
                inds = np.array(range(i+1))
                weights_inds = inds - inds[len(inds)-1]
                weights = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-weights_inds**2/(2*sigma**2))
                weights = weights/np.sum(weights)
                T[i] = sum(weights*seq[inds])
        else: # i+1 > bandwidth
            inds = np.array(range(bandwidth))+i-bandwidth+1
            weights_inds = inds - inds[len(inds)-1]
            weights = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-weights_inds**2/(2*sigma**2))
            weights = weights/np.sum(weights)
            T[i] = sum(weights*seq[inds])
    if plot == 'yes':
        plt.plot(seq)
        plt.plot(T, 'r')
    return T

# In[]
def vola(seq):
    x_f = seq[1:len(seq)]
    x_i = seq[0:len(seq)-1]
    log_return = np.log(x_f/x_i)
    vola = np.var(log_return)
    return vola

# In[] class TS_raw()
class TS_raw():
    
    def __init__(self, TS, paras = dft_paras):
        self.TS = TS
        for para in paras:
            exec('self.'+ para +'= paras[para]')
        # compute the indices in the time series as the RW's endtimes, 'inds'
        inds = np.array(range(0,len(TS)-self.exam_W_UB))
        inds = inds[self.RW_len:len(inds):self.RW_step]
        self.inds = inds
        
    def compute_C(self, sigmas, plot='no'):
        # compute
        if hasattr(sigmas, "__len__"): # check if sigmas is an array
            pass
        else:
            sigmas = np.array([sigmas])
        all_Cs = np.zeros((len(sigmas), len(self.inds)))
        count = 0
        for i in self.inds:
            RW = self.TS[i-self.RW_len:i]
            Cs = cross_corr(RW, sigmas)
            all_Cs[:, count] = Cs
            count = count+1
        if plot == 'yes':
#            plt.figure(figsize=(16,9))
            for j in range(len(sigmas)):
                plt.plot(self.inds, all_Cs[j,:], label = 'bandwidth = %s'%(sigmas[j]))
                plt.legend()
                plt.xlabel('t end')
                plt.ylabel('C')
        self.all_Cs = all_Cs
        self.sigmas = sigmas
        
# In[] class TS()

def truncate(y, l_pct=0, u_pct=0, expand=0.05):
    yshape = y.shape
    y = y.flatten()
    y_sorted = np.sort(y)
    lower = y_sorted[int(np.round(l_pct*len(y)/100))]
    upper = y_sorted[len(y)-int(np.round(u_pct*len(y)/100))]
    height = upper-lower
    if np.min(y)<lower-height*expand:
        y[y<lower-height*expand] = lower-height*expand
    if np.max(y)>upper+height*expand:
        y[y>upper+height*expand] = upper+height*expand
    return y.reshape(yshape)

class TS():
    
    def __init__(self, data_full_name, col, t_start=0., t_end=1.):
        self.data_full_name = data_full_name
        self.col = col
        self.t_start = t_start
        self.t_end = t_end
        self.data_name = data_full_name[2:6]
        with open(data_full_name+'.pickle', 'rb') as f:
            data_pack = pickle.load(f)
        try:
            Mat = data_pack['Mat_raw']
        except:
            Mat = data_pack['Mat']
        print('Time format adjusted.') # matlab datenum is 366 ahead of python's
        Mat[:,0] -= 366
        ra = range(int(len(Mat)*t_start), int(len(Mat)*t_end))
        self.series = Mat[ra,col]
        self.t = Mat[ra,0]
#        self.t_interval = data_pack['t_interval']
        print('TS length = %s'%len(self.t))
        
    def compute_C(self, sigmas, paras, volas = 'no'):
        if hasattr(sigmas, "__len__"): # check if sigmas is an array
            pass
        else:
            sigmas = np.array([sigmas])
        for para in paras:
            exec('self.'+ para +'= paras[para]')
        # compute the indices in the time series as the RW's endtimes, 'inds'
        inds = np.array(range(0,len(self.series)-self.exam_W_UB))
        inds = inds[self.RW_len:len(inds):self.RW_step]
        self.inds = inds
        self.t_inds = self.t[inds]
        all_Cs = np.zeros((len(sigmas), len(self.inds)))
        count = 0
        for i in self.inds:
            RW = self.series[i-self.RW_len:i]
            Cs = cross_corr(RW, sigmas)
            all_Cs[:, count] = Cs
            count = count+1
        self.all_Cs = all_Cs
        self.sigmas = sigmas
        if volas == 'yes':
            volas = np.zeros(len(self.inds))
            count = 0
            for i in self.inds:
                volas[count] = vola(RW)
                count = count+1
            self.volas = volas
    
    def plot_C(self, sig_C = 0, mag = 0, radius = 0.8, save_fig='no'):
#        plt.figure(figsize=(16,9))
        t_plot = self.t_inds 
        for j in range(len(self.sigmas)):
            plt.plot(pdate.num2date(t_plot), self.all_Cs[j,:], 
                     label = 'bandwidth = %s'%(self.sigmas[j]))
            if sig_C > 0:
                T = half_gauss(self.all_Cs[j,:], sig_C)
            plt.plot(pdate.num2date(t_plot), T, 'r', 
                     label = 'smooth bandwidth = %s'%(sig_C))
            plt.legend()
            plt.xlabel('t end')
            plt.ylabel('C')
        if (t_plot[0]<dft_num_quake) & (t_plot[-1]>dft_num_quake):
            plt.axvline(dft_t_quake, color='k')
        if mag > 0:
            eq = eqk()
            eq.select(self.data_name, mag, radius)
            eq.plot()
            self.mag = mag
            self.eq = eq
            plt.title('%s events of magnitude >= %.1f, radius <= %.1f' 
                      %(eq.selected.shape[0], mag, radius))
        if save_fig == 'yes':
            plt.savefig(self.data_full_name+'_'+conv_digits(self.t_start)+
                        '_'+conv_digits(self.t_end)+'%_col'+str(self.col)+
                        '_RW%s_sig%s_Cplot.png'%(self.RW_len, self.sigmas[0]), dpi=220)
            
    def compute_Cmap(self, paras, var = 'no', skw = 'no', krt = 'no'):
        # in this function, paras['RW_lens'] must be np array
        for para in paras:
            exec('self.'+ para +'= paras[para]')
        # compute the indices in the time series as the RW's endtimes, 'inds'
        inds = np.array(range(0,len(self.series)-self.exam_W_UB))
        inds = inds[self.exam_W_UB:len(inds):self.RW_step]
        self.inds = inds
        Cs = np.zeros((len(self.RW_lens), len(self.inds)))
        self.t_inds = self.t[inds]
        for i in range(len(self.inds)):
            ind = self.inds[i]
            for j in range(len(self.RW_lens)):
                RW_len = self.RW_lens[j]
                RW = self.series[ind-RW_len:ind]
                Cs[j, i] = cross_corr(RW, 0)
        self.Cs = Cs
        if var == 'yes':
            Vs = np.zeros((len(self.RW_lens), len(self.inds)))
            for i in range(len(self.inds)):
                ind = self.inds[i]
                for j in range(len(self.RW_lens)):
                    RW_len = self.RW_lens[j]
                    RW = self.series[ind-RW_len:ind]
                    Vs[j, i] = np.var(RW)
            self.Vs = Vs
        if skw == 'yes':
            Ss = np.zeros((len(self.RW_lens), len(self.inds)))
            for i in range(len(self.inds)):
                ind = self.inds[i]
                for j in range(len(self.RW_lens)):
                    RW_len = self.RW_lens[j]
                    RW = self.series[ind-RW_len:ind]
                    Ss[j, i] = skew(RW)
            self.Ss = Ss
        
    def plot_Cmap(self, mag = 0, radius = 0.8, save='no'):
        plot_num = 2
        if hasattr(self, 'Vs'):
            plot_num += 1
        if hasattr(self, 'Ss'):
            plot_num += 1
        fig = plt.figure(figsize=(17,9))
        ax1 = plt.subplot(plot_num,1,1)
        ax1.plot(pdate.num2date(self.t), self.series)
        if mag > 0:
            eq = eqk()
            eq.select(self.data_name, mag, radius)
            eq.plot()
            self.mag = mag
            self.eq = eq
            plt.title('%s events of magnitude >= %.1f, radius <= %.1f' 
                      %(eq.selected.shape[0], mag, radius))
        ax2 = plt.subplot(plot_num,1,2, sharex = ax1)
        plt.yticks(np.arange(len(self.RW_lens))+0.5, self.RW_lens.astype(str))
        X, Y = np.meshgrid((self.t_inds), range(len(self.RW_lens)+1))
        p = plt.pcolor(X, Y, self.Cs, cmap='RdBu')
        fig.colorbar(p)#,orientation="horizontal")
        plt.ylabel('C')
        eq.plot()
        if hasattr(self, 'Vs'):
            plt.subplot(plot_num,1,3, sharex = ax1)
            plt.yticks(np.arange(len(self.RW_lens))+0.5, self.RW_lens.astype(str))
            X, Y = np.meshgrid((self.t_inds), range(len(self.RW_lens)+1))
            p = plt.pcolor(X, Y, truncate(self.Vs, 10, 10), cmap='RdBu')
            fig.colorbar(p)#,orientation="horizontal")
            plt.ylabel('variance')
            eq.plot()
            if hasattr(self, 'Ss'):
                plt.subplot(plot_num,1,4, sharex = ax1)
                plt.yticks(np.arange(len(self.RW_lens))+0.5, self.RW_lens.astype(str))
                X, Y = np.meshgrid((self.t_inds), range(len(self.RW_lens)+1))
                p = plt.pcolor(X, Y, truncate(self.Ss, 10, 10), cmap='RdBu')
                fig.colorbar(p)#,orientation="horizontal")
                plt.ylabel('skewness')
                eq.plot()
        elif hasattr(self, 'Ss'):
            plt.subplot(plot_num,1,3, sharex = ax1)
            plt.yticks(np.arange(len(self.RW_lens))+0.5, self.RW_lens.astype(str))
            X, Y = np.meshgrid((self.t_inds), range(len(self.RW_lens)+1))
            p = plt.pcolor(X, Y, truncate(self.Ss, 10, 10), cmap='RdBu')
            fig.colorbar(p)#,orientation="horizontal")
            plt.ylabel('skewness')
            eq.plot()
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax1.set_position([pos1.x0, pos1.y0, pos2.width, pos1.height] )
        if save == 'yes':
            plt.savefig(self.data_full_name+'_col'+str(self.col)+
                        '_mag%s_rad%s_Cmapplot.png'%(mag, radius), dpi=220)
# In[]

