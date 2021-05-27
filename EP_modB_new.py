"""
Created on Tue May 28 15:52:59 2019

@author: haoyuwen

Copyed from GEMS_GD to GEMS_GD_new on 26 Jul, diverge since then

module contains new EP functions, including:
    1. indicator_map(series, paras, t = np.array([1]), C = 'yes', var = 'yes', 
                  skw = 'yes', krt = 'yes')
    2. indicator_map_eqdata(dataname, paras, C = 'yes', var = 'yes', skw = 'yes', 
                         krt = 'yes', savename = 0)
    3. plot_indimap(fig, col, indi)
    4. normalize(mat)
    5. combine_indi(data, indi_list)
    6. time2failure(t, failures) and time_after_failure(t, failures)
    7. taf_filter(indi, taf, lower_bound = 0, upper_bound = 0)
    8. ttf_scatplot(col, t_eqs, taf_LB = 2, taf_UB = 0, save = 'no') and
       taf_scatplot(col, t_eqs, save = 'no')
    9. indi_plot(col, save = 'no', mag = 5., radius = 1.)
    10. get_dist(indi, ttf, taf, bounds = [-np.inf, np.inf, -np.inf, np.inf])
    11. plot_pmf(seq, bins, label = 0)
    12. create_Blist(incre = 1, taf_LB = 3, max_ttf = 15) and
        compute_MVS_indi(indi_mat, ttf, taf, bounds_list) and 
        compute_pcts_inds(indi_mat, ttf, taf, bounds_list)
    13. plot_MVS(dataname, col_name, bounds_list = create_Blist(1), save='no') and
        plot_pcts(dataname, col_name, taf_LB = 2, mag = 4, mag_UB = 0, radius = 1., 
              pcts = [0.1, 0.5, 0.9], save='no')
    14. convert_PCTmat(indi_seq, t_indi, dist_WL, R_step = 0) and
        compute_tau(PCT_TSs, t_PCT, Tau_WL, R_step = 1)
    15. scatplot_Taus(dataname, col_name = 'col1', indi_name = 'Cs', dist_WL = 72,
                  Tau_WL = 42, mag = 4, mag_UB = 0, radius = 1., taf_LB=2, save='no')
    16. compute_fs(t, eq_mat, WL_fs=10) and
        scatplot_Taus_fs(dataname, col_name = 'col1', indi_name = 'Cs', RW_ind = 0, 
                     dist_WL = 72, Tau_WL = 42, radius = 1., WL_fs=10, save='no')
    17. fs_scatplot(dataname, col_name, radius, WL_fs, save='no')
    18. mktrend_n_save(name, bandwidth) and
        mkres_n_save(name, bandwidth)
    19. seq2score(seq, threshold), 
        compute_GWMtau(score_window, pre_ignore_rate = 0.5), and
        syscompute_GWMtau(indi_seq, indi_t, threshold, Tau_WL = 72*7, R_step = 1, 
                      pre_ignore_rate = 0.5)
    20. count_GWMTaus(Taus, verbose = 'no')
        count_eqs_GWMTau(t_eqs, t_Taus, Taus, verbose = 'no'), and 
        compute_EBratio(Taus, t_Taus, t_eqs, verbose = 'no')
    21. crop_tod(col, tod_LB = 7/24, tod_UB = 17/24):
    22. tod_scatplot(col, save_info = str(), save = 'no')
    23. << KS matrix analysis with reject_rate >>
    24. << KS matrix analysis with modularity >>
    25. get_seg_pts(t_start, t_end, seglen_max = 5, seglen_min = 4, max_overlap = 1)
    26. cut_distends(dist, pct = 0.5)
    27. ensemble_analysis(dataname, mag = 4.2, rad = 0.6, col_name = 'col1', 
                          indi_name = 'Cs', posi_ttf_UB = 5, nega_ttf_LB = 7,
                          nega_taf_UB = 2, seglen = 50, max_overlap = 2, plot ='yes', 
                          save = 'no')
    28. generate_TRnTE(dataname, id_div, total_div = 10)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from EP_modA import cross_corr
#from EP_modA import eqk
from scipy.stats import skew
from scipy.stats import kurtosis
import time
from scipy.stats import kendalltau
from scipy.ndimage import gaussian_filter
from scipy import stats
import seaborn as sns
import random

# In[] 0
def ind_inbetween(num, seq):
    # function finds the indices in seq for which num is in between numerically
    inds = np.arange(1, len(seq))
    greaters = seq > num
    ind = inds[np.diff(greaters) == True]
    if len(ind) == 0:
        ind = len(seq)-1
    return int(ind-1), int(ind)

class eqk():
    
    def __init__(self, name='tweqk28'):
        with open(name+'.pickle', 'rb') as f:
            data_pack = pickle.load(f)
        self.mat = data_pack['mat']
        self.mat[:,0] = self.mat[:,0]
        self.lld = data_pack['lld']
        self.position = dict() # lattitude, longitude
        self.position['LIOQ'] = np.array([23.0321, 120.6632])
        self.position['CHCH'] = np.array([23.2197, 120.1618])
        self.position['SHRL'] = np.array([25.1559, 121.5619])
        self.position['DABA'] = np.array([23.4544, 120.7494])
        self.position['FENL'] = np.array([23.7156, 121.4112])
        self.position['HERM'] = np.array([24.1088, 120.5015])
        self.position['PULI'] = np.array([23.9208, 120.9788])
        self.position['YULI'] = np.array([23.3247, 121.3181])
        self.position['SIHU'] = np.array([23.6370, 120.2293])
        self.position['RUEY'] = np.array([22.9732, 121.1557])
        self.position['KUOL'] = np.array([24.9629, 121.1420])
        self.position['TOCH'] = np.array([24.8435, 121.8052])
        self.position['HUAL'] = np.array([24.6745, 121.3677])
        self.position['ENAN'] = np.array([24.4758, 121.7849])
        self.position['DAHU'] = np.array([24.4106, 120.9024])
        self.position['LISH'] = np.array([24.2495, 121.2524])
        self.position['SHCH'] = np.array([24.1183, 121.6250])
        self.position['KAOH'] = np.array([22.6577, 120.2893])
        self.position['WANL'] = np.array([22.5909, 120.5937])
        self.position['FENG'] = np.array([22.2043, 120.7007])
    
    def NS_divide(self, keep_eq = ''):
        position = self.position
        south_stns = dict()
        north_stns = dict()
        latitudes = []
        for stn in position:
            latitudes.append(position[stn][0])
        latitudes = np.array(latitudes)
        latitudes = np.sort(latitudes)
        mid_latitude = (latitudes[9] + latitudes[10])/2
        for stn in position:
            if position[stn][0] > mid_latitude:
                north_stns[stn] = position[stn]
            else:
                south_stns[stn] = position[stn]
        self.north_stns = north_stns
        self.south_stns = south_stns
        self.mid_latitude = mid_latitude
        if len(keep_eq) == 1: # either 'N' or 'S'
            eq_lats = self.lld[:,0]
            if keep_eq == 'N':
                eq_keep = eq_lats >= mid_latitude
            elif keep_eq == 'S':
                eq_keep = eq_lats < mid_latitude
            self.mat = self.mat[eq_keep, :]
            self.lld = self.lld[eq_keep, :]
        eq_lats = self.lld[:,0]
        self.div_y_N_rate = (np.max(eq_lats)-mid_latitude)/(np.max(eq_lats)-np.min(eq_lats))
        self.div_y_S_rate = 1 - self.div_y_N_rate
        return
    
    def get_mag_divs(self, num_div = 10):
        # function divides magnitudes evenly, and record the division points
        mags_sorted = np.sort(self.mat[:,1])
        distance_div = np.int(np.floor(len(mags_sorted)/(num_div-1)))
        mag_divs = np.zeros((num_div, ))
        ind = 0
        for i in range(num_div):
            mag_divs[i] = mags_sorted[ind]
            ind += distance_div
        self.mag_divs = mag_divs
    
    def select(self, name, mag, radius = 0.8, verbose = 'yes'):
        self.selected = self.mat.copy()
        self.selected_lld = self.lld.copy()
        mags = self.selected[:,1]
        pos = self.position[name[0:4]]
        ds = np.sqrt((self.lld[:,0]-pos[0])**2 + (self.lld[:,1]-pos[1])**2)
        self.selected = self.selected[mags>mag, :]
        self.selected_lld = self.selected_lld[mags>mag, :]
        num_total = self.selected.shape[0]
        ds = ds[mags>mag]
        self.selected = self.selected[ds<radius, :]
        self.selected_lld = self.selected_lld[ds<radius, :]
        self.radius = radius
        self.mag = mag
        if verbose == 'yes':
            print('Selected near events %s out of %s total events.' 
                  %(self.selected.shape[0], num_total))
    
    def select_ULB(self, name, mag_UB, mag_LB, rad_UB, rad_LB, verbose = 'yes'):
        self.selected = self.mat.copy()
        self.selected_lld = self.lld.copy()
        mags = self.selected[:,1]
        pos = self.position[name[0:4]]
        ds = np.sqrt((self.lld[:,0]-pos[0])**2 + (self.lld[:,1]-pos[1])**2)
        pass_mag = np.logical_and(mags > mag_LB, mags < mag_UB)
        pass_rad = np.logical_and(ds < rad_UB, ds > rad_LB)
        to_select = np.logical_and(pass_mag, pass_rad)
        self.selected = self.selected[to_select, :]
        self.selected_lld = self.selected_lld[to_select, :]
        self.mag_LB = mag_LB
        self.mag_UB = mag_UB
        self.rad_LB = rad_LB
        self.rad_UB = rad_UB
        if verbose == 'yes':
            print('Selected near events %s with ULB.' %(self.selected.shape[0]))
    
    def remove_aftshock(self, distance, t_lag, verbose = 'yes'):
        '''
        remove after shocks satisfying both:
            (1) time since last eq less than 't_lag'
            (2) epicenter distance from last eq less than 'distance'
        '''
        selected_eq_info = np.concatenate((self.selected, self.selected_lld), axis=1)
        to_remove = np.zeros((selected_eq_info.shape[0], ))
        for i in np.arange(1, selected_eq_info.shape[0]):
            if selected_eq_info[i,0] - selected_eq_info[i-1,0] < t_lag:
                d1sq = (selected_eq_info[i,2] - selected_eq_info[i-1,2])**2
                d2sq = (selected_eq_info[i,3] - selected_eq_info[i-1,3])**2
                if np.sqrt(d1sq + d2sq) < distance:
                    to_remove[i] = 1
        if verbose == 'yes':
            print('Removed %s aftershocks.'%np.sum(to_remove))
        to_keep = (1 - to_remove).astype('bool')
        selected_eq_info = selected_eq_info[to_keep, :]
        self.selected_no_aftshock = selected_eq_info
        
    def plot(self, no_aftshock = False):
        if no_aftshock == False:
            for i in range(self.selected.shape[0]):
                plt.axvline(self.selected[i,0], color='k')
        else:
            for i in range(self.selected_no_aftshock.shape[0]):
                plt.axvline(self.selected_no_aftshock[i,0], color='b')
                
    def makeGM(self, div_y = 20, div_x = 0, plot = 'no'):
        lld = self.lld.copy()
        mat = self.mat.copy()
        y_min, y_max = np.min(lld[:,0]), np.max(lld[:,0]) 
        x_min, x_max = np.min(lld[:,1]), np.max(lld[:,1])
        eqs_keep = np.ones((len(lld[:,0])), dtype = bool)
        eqs_keep[(lld[:,0]==y_min)|(lld[:,0]==y_max)] = 0
        eqs_keep[(lld[:,1]==x_min)|(lld[:,1]==x_max)] = 0
        lld_keep = lld[eqs_keep]
        mat_keep = mat[eqs_keep]
        y_min, y_max = np.min(lld_keep[:,0]), np.max(lld_keep[:,0])
        x_min, x_max = np.min(lld_keep[:,1]), np.max(lld_keep[:,1])
        box_y = (y_max - y_min)/div_y
        if div_x == 0: # 0 for unassigned
            div_x = np.int(np.round((x_max - x_min)/box_y))
        y_cuts = np.linspace(y_min, y_max, div_y+1) 
        x_cuts = np.linspace(x_min, x_max, div_x+1)
        if plot == 'yes': 
            fig = plt.figure(figsize=(16,16))
            f_title = 'EQ Grid Map with %s x-divisions and %s y-divisions'%(div_x, div_y)
            f_title += ', labeled with (ix, iy)'
            plt.title(f_title)
            for iy in range(len(y_cuts)):
                plt.plot([x_min, x_max], [y_cuts[iy], y_cuts[iy]], 'b--', linewidth=0.5)
            for ix in range(len(x_cuts)):
                plt.plot([x_cuts[ix], x_cuts[ix]], [y_min, y_max], 'b--', linewidth=0.5)
            plt.scatter(lld_keep[:,1], lld_keep[:,0], s = (np.exp(mat_keep[:,1])/20))
            position = self.position
            for stn in position:
                plt.scatter([position[stn][1]], [position[stn][0]], marker = '*', s = 120, 
                            label = stn)
            for ix in range(div_x):
                for iy in range(div_y):
                    loc_x = (x_cuts[ix] + x_cuts[ix+1])/2
                    loc_y = (y_cuts[iy] + y_cuts[iy+1])/2
                    plt.text(loc_x, loc_y, '(%s,%s)'%(ix, iy), 
                             horizontalalignment='center', verticalalignment='center')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            figure_name = 'EQGM_divx%s_divy%s'%(div_x, div_y)
            plt.savefig(figure_name +'.tiff', dpi = 300, bbox_inches='tight')
            plt.close(fig)
        GMyx = np.zeros((len(mat_keep), 2), dtype = int) 
        # GMyx gives the box index of GM, box index ranges from 0, and has cutnum-1 values
        for i in range(len(GMyx)):
            GMyx[i,1], _ = ind_inbetween(lld_keep[i,1], x_cuts)
            GMyx[i,0], _ = ind_inbetween(lld_keep[i,0], y_cuts)
        self.div_x = div_x
        self.div_y = div_y
        self.eqs_keep = eqs_keep
        self.lld_keep = lld_keep
        self.mat_keep = mat_keep
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_cuts = x_cuts
        self.y_cuts = y_cuts
        self.GMyx = GMyx
        
    def GM_heatmap(self, GMyx = [], plot = 'no'):
        if len(GMyx) == 0:
            print('Using all EQs for GM heatmap')
            GMyx = self.GMyx # otherwise, use input GMyx n-by-2 mat
        x_cuts = self.x_cuts
        y_cuts = self.y_cuts
        numEQyx = np.zeros((len(y_cuts)-1,len(x_cuts)-1))
        for ix in range(len(x_cuts)-1):
            for iy in range(len(y_cuts)-1):
                numEQyx[iy, ix] = np.sum(np.logical_and(GMyx[:,0]==iy, GMyx[:,1]==ix))
        if plot == 'yes':
            plt.figure()
            ax = sns.heatmap(np.flipud(numEQyx), cmap= "BuPu")
            ax.set_xticks(np.arange(0, len(x_cuts),1))
            ax.set_xticklabels(np.round(x_cuts*100)/100, rotation = 270)
            ax.set_yticks(np.flipud(np.arange(0, len(y_cuts),1)))
            ax.set_yticklabels(np.round(y_cuts*100)/100, rotation = 0)
            ax.scatter(self.lld_keep[:,1], self.lld_keep[:,0], 
                       s = (np.exp(self.mat_keep[:,1])/20))
            ax.set_title('EQ heatmap')
        return numEQyx
    
    def GM_select(self, iy, ix, plot = 'no'):# --- dev ---
        if plot == 'yes':
            fig2 = plt.figure(figsize=(14,14))
            plt.plot([self.x_min, self.x_max], [self.y_cuts[iy], self.y_cuts[iy]], 
                     'b--', linewidth=0.5)
            plt.plot([self.x_min, self.x_max], [self.y_cuts[iy+1], self.y_cuts[iy+1]], 
                     'b--', linewidth=0.5)
            plt.plot([self.x_cuts[ix], self.x_cuts[ix]], [self.y_min, self.y_max], 
                     'b--', linewidth=0.5)
            plt.plot([self.x_cuts[ix+1], self.x_cuts[ix+1]], [self.y_min, self.y_max], 
                     'b--', linewidth=0.5)
            plt.scatter(self.lld_keep[:,1], self.lld_keep[:,0], 
                        s = (np.exp(self.mat_keep[:,1])/20))
            position = self.position
            for stn in position:
                plt.scatter([position[stn][1]], [position[stn][0]], marker = '*', s = 120, 
                            label = stn)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('EQs, stations, and the select area with ix%s_iy%s_divx%s_divy%s'%(ix, 
                                                                iy, self.div_x, self.div_y))
            figure_name = 'EQGM_selected_ix%s_iy%s_divx%s_divy%s'%(ix, iy, self.div_x, self.div_y)
            plt.savefig(figure_name +'.png', dpi = 500)
            plt.close(fig2)
#            plt.close()
        return self.mat_keep[np.logical_and(self.GMyx[:,0]==iy, self.GMyx[:,1]==ix)]
        
# In[] 1
def indicator_map(series, paras, t = np.array([1]), C = 'yes', var = 'yes', 
                  skw = 'yes', krt = 'yes'):
    '''
    function computes indicator map, adapted from Class TS() in EP_modA.py
    in this function, paras['RW_lens'] must be np array
    '''
    results = dict() # the result dictionary to return
    results['paras'] = paras
    RW_lens = paras['RW_lens']
    exam_W_UB = paras['exam_W_UB']
    RW_step = paras['RW_step']
    # compute the indices in the time series as the RW's endtimes, 'inds'
    inds = np.array(range(0,len(series)-exam_W_UB))
    inds = inds[exam_W_UB:len(inds):RW_step]
    results['inds'] = inds
    if len(t) > 1:
        t_ind = t[inds]
        results['t_ind'] = t_ind
    if C == 'yes':
        tic = time.time()
        Cs = np.zeros((len(RW_lens), len(inds)))
        for i in range(len(inds)):
            ind = inds[i]
            for j in range(len(RW_lens)):
                RW_len = RW_lens[j]
                RW = series[ind-RW_len:ind]
                Cs[j, i] = cross_corr(RW, 0)
        results['Cs'] = Cs
        print('Correlation computed, elapsed %.0fs.'%(time.time()-tic))
    if var == 'yes':
        tic = time.time()
        Vs = np.zeros((len(RW_lens), len(inds)))
        for i in range(len(inds)):
            ind = inds[i]
            for j in range(len(RW_lens)):
                RW_len = RW_lens[j]
                RW = series[ind-RW_len:ind]
                Vs[j, i] = np.var(RW)
        results['Vs'] = Vs
        print('Variance computed, elapsed %.0fs.'%(time.time()-tic))
    if skw == 'yes':
        tic = time.time()
        Ss = np.zeros((len(RW_lens), len(inds)))
        for i in range(len(inds)):
            ind = inds[i]
            for j in range(len(RW_lens)):
                RW_len = RW_lens[j]
                RW = series[ind-RW_len:ind]
                Ss[j, i] = skew(RW)
        results['Ss'] = Ss
        print('Skewness computed, elapsed %.0fs.'%(time.time()-tic))
    if krt == 'yes':
        tic = time.time()
        Ks = np.zeros((len(RW_lens), len(inds)))
        for i in range(len(inds)):
            ind = inds[i]
            for j in range(len(RW_lens)):
                RW_len = RW_lens[j]
                RW = series[ind-RW_len:ind]
                Ks[j, i] = kurtosis(RW)
        results['Ks'] = Ks
        print('Kurtosis computed, elapsed %.0fs.'%(time.time()-tic))
    return results


def indicator_map_day(series, t, RW_day = 1):
    '''
    modification of indicator_map
    '''
    results = dict() # the result dictionary to return
    results['paras'] = RW_day
    results['RW_day'] = RW_day
    t = t/RW_day
    # compute the indices in the time series as the RW's endtimes, 'inds'
    max_duration = np.int(np.floor(t[-1] - t[0]+1))
    inds = np.zeros((max_duration, ))
    init_step = 1
    i_t = 0
    i_inds = 0
    mod_ts = np.mod(t, 1)
    while i_t < len(t):
        if init_step == 1:
            if mod_ts[i_t] > 0.1:
                i_t += 1
                continue
            else:
                init_step = 0
                inds[i_inds] = i_t
                i_t += 1
                i_inds += 1
        else:
            if mod_ts[i_t] < mod_ts[i_t-1]: # t[i_t] starts a new day
                if i_t - inds[i_inds-1] > 200: # at least 200 readings per day
                    inds[i_inds] = i_t-1
                    i_inds += 1
                else:
                    print('not enough ready per day at %.2f'%t[i_t])
            i_t += 1
    inds = inds[0:i_inds]
    inds = np.int64(inds)
    t = t*RW_day
    results['inds'] = inds
    if len(t) > 1:
        t_ind = t[inds]
        results['t_ind'] = t_ind
    # C
    tic = time.time()
    Cs = np.zeros((1, len(inds)-1))
    for i in range(len(inds)-1):
        RW = series[inds[i]:inds[i+1]]
        Cs[0, i] = cross_corr(RW, 0)
    results['Cs'] = Cs
    print('Correlation computed, elapsed %.0fs.'%(time.time()-tic))
    # V
    tic = time.time()
    Vs = np.zeros((1, len(inds)-1))
    for i in range(len(inds)-1):
        RW = series[inds[i]:inds[i+1]]
        Vs[0, i] = np.var(RW)
    results['Vs'] = Vs
    print('Variance computed, elapsed %.0fs.'%(time.time()-tic))
    # S
    tic = time.time()
    Ss = np.zeros((1, len(inds)-1))
    for i in range(len(inds)-1):
        RW = series[inds[i]:inds[i+1]]
        Ss[0, i] = skew(RW)
    results['Ss'] = Ss
    print('Skewness computed, elapsed %.0fs.'%(time.time()-tic))
    # K
    tic = time.time()
    Ks = np.zeros((1, len(inds)-1))
    for i in range(len(inds)-1):
        RW = series[inds[i]:inds[i+1]]
        Ks[0, i] = kurtosis(RW)
    results['Ks'] = Ks
    print('Kurtosis computed, elapsed %.0fs.'%(time.time()-tic))
    return results

def indicator_map_eqdata_day(dataname, extra_savename = '', RW_day = 1):
    '''
    systematically call 'indicator_map' function on eq data
    e.g. D_LIOQ_ti2s_v524_th50_n30_trimmed
    '''
    with open(dataname + '.pickle', 'rb') as f:
        data = pickle.load(f)
    Mat = data['Mat']
    col1 = indicator_map_day(Mat[:,1], Mat[:,0], RW_day=RW_day)
    col1['dataname'] = dataname+'_col1'
    print('Column 1 done')
    col2 = indicator_map_day(Mat[:,2], Mat[:,0], RW_day=RW_day)
    col2['dataname'] = dataname+'_col2'
    print('Column 2 done')
    result = {'col1': col1, 'col2': col2}
    with open(dataname + '_RWday%s'%RW_day + extra_savename + '.pickle', 'wb') as f:
        pickle.dump(result, f)
    return result

# In[] 2
dft_paras = {'RW_lens': np.array([1800, 7200, 28800]), 
             'RW_step': 600, 
             'exam_W_UB': 28800}

def indicator_map_eqdata(dataname, paras = dft_paras, C = 'yes', var = 'yes', 
                         skw = 'yes', krt = 'yes', savename = 0,
                         scramble = 'no', scramble_id = 0):
    '''
    systematically call 'indicator_map' function on eq data
    e.g. D_LIOQ_ti2s_v524_th50_n30_trimmed
    '''
    with open(dataname + '.pickle', 'rb') as f:
        data = pickle.load(f)
    Mat = data['Mat']
    if scramble == 'yes':
        ind = np.arange(Mat.shape[0])
        random.shuffle(ind)
        cols_rand = Mat[:, 1:3]
        cols_rand = cols_rand[ind, :]
        Mat_rand = Mat.copy()
        Mat_rand[:, 1:3] = cols_rand
        Mat = Mat_rand
        scramble_txt = '%s'%scramble_id
        while len(scramble_txt) < 2:
            scramble_txt = '0'+scramble_txt
        scramble_txt = '_SCR'+scramble_txt
    else:
        scramble_txt = ''
    col1 = indicator_map(Mat[:,1], paras, t=Mat[:,0], var=var, skw=skw, krt=krt)
    col1['dataname'] = dataname+'_col1'
    print('Column 1 done')
    col2 = indicator_map(Mat[:,2], paras, t=Mat[:,0], var=var, skw=skw, krt=krt)
    col2['dataname'] = dataname+'_col2'
    print('Column 2 done')
    result = {'col1': col1, 'col2': col2}
    if savename != 0:
        with open(dataname + '_' + savename + scramble_txt + '.pickle', 'wb') as f:
            pickle.dump(result, f)
    return result

# In[] 3
def plot_indimap(fig, col, indi):
    '''
    col is in the format of D_CHCH_ti2s_v524_th50_n30_trimmed_CVSK.pickle
    '''
    RW_lens = col['paras']['RW_lens']
    X, Y = np.meshgrid((col['t_ind']), range(col[indi].shape[0]+1))
    plt.yticks(np.arange(len(RW_lens))+0.5, RW_lens.astype(str))
    p = plt.pcolor(X, Y, col[indi], cmap='RdBu')
    fig.colorbar(p)#,orientation="horizontal")
    plt.ylabel('C')

# In[] 4
def normalize(mat):
    '''
    normalize a matrix by subtracting mean and dividing by std
    '''
    matc = mat.copy()
    matc = (matc - np.nanmean(mat))/np.nanstd(mat)
    return matc

# In[] 5
def combine_indi(data, indi_list):
    '''
    e.g. combine_indi(data, ['Cs', 'Vs', 'Ss', 'Ks'])
    data is in the format of D_CHCH_ti2s_v524_th50_n30_trimmed_CVSK.pickle
    '''
    count = 1
    for item in indi_list:
        new = normalize([data['col1'][item]])[0, :, :]
        if count == 1:
            combined1 = new
            names = item[0]
        else:
            combined1 = np.concatenate((combined1, new))
            names = names+'_'+item[0]
        count += 1
    data['col1'][names] = combined1
    print('Have combined '+str(count-1)+' indicator for col1: ' + names)
    count = 1
    for item in indi_list:
        new = normalize([data['col2'][item]])[0, :, :]
        if count == 1:
            combined2 = new
        else:
            combined2 = np.concatenate((combined2, new))
        count += 1
    data['col2'][names] = combined2
    print('Have combined '+str(count-1)+' indicator for col2: ' + names)

# In[] 6
def time2failure(t, failures):
    '''
    function computes time to failure array 'ttf' from time array 't'
    array 'failures' marks failure times
    '''
    pointer = 0 # goes through the length of failures
    ttf = np.empty((len(t),))
    ttf[:] = np.nan
    for i in range(len(t)):
        if pointer < len(failures):
            ttf[i] = failures[pointer] - t[i]
            if failures[pointer] < t[i]:
                pointer += 1
    return ttf

def time_after_failure(t, failures):
    '''
    function computes time after failure array 'taf' from time array 't'
    array 'failures' marks failure times
    '''
    pointer = 0 # goes through the length of failures
    taf = np.empty((len(t),))
    taf[:] = np.nan
    for i in range(len(t)):
        if t[i] < failures[0]:
            pass
        elif pointer < len(failures)-1:
            if failures[pointer+1] < t[i]:
                pointer += 1
            taf[i] = t[i] - failures[pointer]
        elif pointer == len(failures)-1:
            taf[i] = t[i] - failures[pointer]
    return taf

# In[] 7
def taf_filter(indi, taf, lower_bound = 0, upper_bound = 0):
    '''
    indi is a 1-D seq (of indicators) or n-by-time mat
    taf is a 1-D corresponding seq of time_after_failures
    function filters indi_seq such that those whose taf must be within lower_bound 
    and upper_bound, otherwise shall be converted to nan
    '''
    indic = indi.copy()
    if len(indic.shape) == 1:
        if lower_bound == 0:
            pass
        else:
            indic[taf < lower_bound] = np.nan
        if upper_bound == 0:
            pass
        else:
            indic[taf > upper_bound] = np.nan
    else: # indic is n-by-time mat
        if lower_bound == 0:
            pass
        else:
            indic[:, taf < lower_bound] = np.nan
        if upper_bound == 0:
            pass
        else:
            indic[:, taf > upper_bound] = np.nan
    return indic

# In[] 8
def ttf_scatplot(col, t_eqs, taf_LB = 2, taf_UB = 0, save_info = str(), save = 'no',
                 plot_name = ''):
    '''
    in case of saving figure, the figure will not appear, but saved directly
    '''
    fig = plt.figure(figsize=(16,9))
    if len(plot_name) == 0:
        plot_name = 'ttf_scat_'+col['dataname'][2:6] + '_' + save_info
        if taf_LB > 0:
            plot_name = plot_name + '_tafLB%s'%taf_LB
        if taf_UB > 0:
            plot_name = plot_name + '_tafUB%s'%taf_UB
#    plot_name = plot_name + '_' + save_info
    fig.suptitle(plot_name)
    ttf = time2failure(col['t_ind'], t_eqs)
    taf = time_after_failure(col['t_ind'], t_eqs)
#    print('Time axis adjusted by -365')
    C_map = col['Cs']
    RW_lens = col['paras']['RW_lens']
    ttf_notnan = ttf.copy()[~np.isnan(ttf)]
    ttf_notnan_sorted = np.sort(ttf_notnan)
    x_UB = ttf_notnan_sorted[-200]
    print(x_UB)
    plt.subplot(2,2,1)
    indi_C = taf_filter(col['Cs'], taf, lower_bound = taf_LB)
    for i in range(C_map.shape[0]):
        plt.scatter(ttf, indi_C[i,:], label = 'RW len = ' +str(RW_lens[i]), 
                    s = 4, alpha=0.4)
    indi_sorted = np.sort(indi_C[0,:])
    indi_sorted = indi_sorted[~np.isnan(indi_sorted)]
    y_UB = indi_sorted[int(len(indi_sorted)*0.999)]-0.001
    y_LB = indi_sorted[int(len(indi_sorted)*0.001)]
    plt.ylim([y_LB, y_UB])
    plt.xlim([0, x_UB-1])
    plt.legend()
    plt.xlabel('ttf')
    plt.ylabel('C')
    plt.subplot(2,2,2)
    indi_V = taf_filter(col['Vs'], taf, lower_bound = taf_LB)
    for i in range(C_map.shape[0]):
        plt.scatter(ttf, indi_V[i,:], label = 'RW len = ' +str(RW_lens[i]), 
                    s = 4, alpha=0.4)
    indi_sorted = np.sort(indi_V[0,:])
    indi_sorted = indi_sorted[~np.isnan(indi_sorted)]
    y_UB = indi_sorted[int(len(indi_sorted)*0.5)]*500
    if np.max(indi_sorted) < y_UB:
        y_UB = np.max(indi_sorted)
    y_LB = indi_sorted[int(len(indi_sorted)*0.01)]
    plt.ylim([y_LB, y_UB])
    plt.xlim([0, x_UB])
    plt.legend()
    plt.xlabel('ttf')
    plt.ylabel('V')
    plt.subplot(2,2,3)
    indi_S = taf_filter(col['Ss'], taf, lower_bound = taf_LB)
    for i in range(C_map.shape[0]):
        plt.scatter(ttf, indi_S[i,:], label = 'RW len = ' +str(RW_lens[i]), 
                    s = 4, alpha=0.4)
    indi_sorted = np.sort(indi_S[0,:])
    indi_sorted = indi_sorted[~np.isnan(indi_sorted)]
    y_UB = indi_sorted[int(len(indi_sorted)*0.999)]
    y_LB = indi_sorted[int(len(indi_sorted)*0.001)]
    plt.ylim([y_LB, y_UB])
    plt.xlim([0, x_UB])
    plt.legend()
    plt.xlabel('ttf')
    plt.ylabel('S')
    plt.subplot(2,2,4)
    indi_K = taf_filter(col['Ks'], taf, lower_bound = taf_LB)
    for i in range(C_map.shape[0]):
        plt.scatter(ttf, indi_K[i,:], label = 'RW len = ' +str(RW_lens[i]), 
                    s = 4, alpha=0.4)
    indi_sorted = np.sort(indi_K[0,:])
    indi_sorted = indi_sorted[~np.isnan(indi_sorted)]
    y_UB = indi_sorted[int(len(indi_sorted)*0.999)]
    y_LB = indi_sorted[int(len(indi_sorted)*0.001)]
    plt.ylim([y_LB, y_UB])
    plt.xlim([0, x_UB])
    plt.legend()
    plt.xlabel('ttf')
    plt.ylabel('K')
    if save == 'no':
        pass
    else:
        plt.savefig(plot_name+'.png', dpi = 300)
        plt.close(fig)

def taf_scatplot(col, t_eqs, save = 'no'):
    '''
    in case of saving figure, the figure will not appear, but saved directly
    '''
    fig = plt.figure(figsize=(16,9))
    fig.suptitle(col['dataname']+'_taf')
    taf = time_after_failure(col['t_ind']-365, t_eqs)
    print('Time axis adjusted by -365')
    C_map = col['Cs']
    RW_lens = col['paras']['RW_lens']
    plt.subplot(2,2,1)
    for i in range(C_map.shape[0]):
        plt.scatter(taf, col['Cs'][i,:], 
                    label = 'RW len = ' +str(RW_lens[i]), s = 4, alpha=0.4)
    plt.legend()
    plt.xlabel('taf')
    plt.ylabel('C')
    plt.subplot(2,2,2)
    for i in range(C_map.shape[0]):
        plt.scatter(taf, col['Vs'][i,:], 
                    label = 'RW len = ' +str(RW_lens[i]), s = 4, alpha=0.4)
    plt.legend()
    plt.xlabel('taf')
    plt.ylabel('V')
    plt.subplot(2,2,3)
    for i in range(C_map.shape[0]):
        plt.scatter(taf, col['Ss'][i,:], 
                    label = 'RW len = ' +str(RW_lens[i]), s = 4, alpha=0.4)
    plt.legend()
    plt.xlabel('taf')
    plt.ylabel('S')
    plt.subplot(2,2,4)
    for i in range(C_map.shape[0]):
        plt.scatter(taf, col['Ks'][i,:], 
                    label = 'RW len = ' +str(RW_lens[i]), s = 4, alpha=0.4)
    plt.legend()
    plt.xlabel('taf')
    plt.ylabel('K')
    if save == 'yes':
        plt.savefig(col['dataname']+'_taf', dpi = 300)
        plt.close(fig)

# In[] 9
def indi_plot(col, save = 'no', mag = 5., radius = 1., res_reduce = 1):
    '''
    plots indicator TS along earthquake time stamps
    '''
    fig = plt.figure(figsize=(24,12.5))
    fig.suptitle(col['dataname'])
    C_map = col['Cs']
    RW_lens = col['paras']['RW_lens']
    data_name = col['dataname'][2:6]
    if mag > 0:
        eqks = eqk()
        eqks.select(data_name, mag, radius = radius)
    ax1 = plt.subplot(2,2,1)
    for i in range(C_map.shape[0]):
        plt.plot(col['t_ind'][::res_reduce], col['Cs'][i,:][::res_reduce], 
                 label = 'RW len = '+str(RW_lens[i]))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('C')
    if mag > 0:
        eqks.plot()
    plt.subplot(2,2,2, sharex = ax1)
    for i in range(C_map.shape[0]):
        plt.plot(col['t_ind'][::res_reduce], col['Vs'][i,:][::res_reduce], 
                 label = 'RW len = '+str(RW_lens[i]))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('V')
    if mag > 0:
        eqks.plot()
    plt.subplot(2,2,3, sharex = ax1)
    for i in range(C_map.shape[0]):
        plt.plot(col['t_ind'][::res_reduce], col['Ss'][i,:][::res_reduce], 
                 label = 'RW len = '+str(RW_lens[i]))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('S')
    if mag > 0:
        eqks.plot()
    plt.subplot(2,2,4, sharex = ax1)
    for i in range(C_map.shape[0]):
        plt.plot(col['t_ind'][::res_reduce], col['Ks'][i,:][::res_reduce], 
                 label = 'RW len = '+str(RW_lens[i]))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('K')
    if mag > 0:
        eqks.plot()
    if save == 'yes':
        plt.savefig(col['dataname']+'_CVSK_indiplot', dpi = 500)
        plt.close(fig)

# In[] 10
def get_dist(indi, ttf, taf, bounds = [-np.inf, np.inf, -np.inf, np.inf]):
    '''
    bounds = [ttf_LB, ttf_UB, taf_LB, taf_UB]
    function reduces indi and only keeps entries/column whose ttf&taf are in bounds
    and not being nan
    '''
    [ttf_LB, ttf_UB, taf_LB, taf_UB] = bounds
    indic = indi.copy()
    if len(indic.shape) > 1:
        remove = np.zeros((indic.shape[1], ), dtype = 'bool')
        for i in range(indic.shape[1]):
            if (ttf[i]<ttf_LB) | (ttf[i]>ttf_UB) | (taf[i]<taf_LB) | (taf[i]>taf_UB):
                remove[i] = True
        for j in range(indic.shape[0]):
            remove[np.isnan(indic[j,:])] = True
        indic = indic[:, ~remove]
    else:
        remove = np.zeros((len(indic), ), dtype = 'bool')
        for i in range(len(indic)):
            if (ttf[i]<ttf_LB) | (ttf[i]>ttf_UB) | (taf[i]<taf_LB) | (taf[i]>taf_UB):
                remove[i] = True
        remove[np.isnan(indic)] = True
        indic = indic[~remove]
    return indic

# In[] 11
def plot_pmf(seq, bins, label = 0):
    '''
    function plots pmf of 1-D seq; label should be string if input any
    '''
    counts, bins = np.histogram(seq, bins = bins)
    bins = bins[:-1] + (bins[1] - bins[0])/2
    probs = counts/float(counts.sum())
    plt.plot(bins, probs, label = label)
    if label != 0:
        plt.legend()

# In[] 12
def create_Blist(incre = 1, taf_LB = 2, max_ttf = 15):
    '''
    prepares the list of bounds to compare through, in which each entry 
    is also a list: bounds = [ttf_LB, ttf_UB, taf_LB, taf_UB]
    '''
    bounds_list = []
    bounds = [0,incre,taf_LB,np.inf]
    num_list = np.int(max_ttf/incre)
    for i in range(num_list):
        bounds_list.append(bounds.copy())
        bounds[0] += incre
        bounds[1] += incre
    return bounds_list

def compute_MVS_indi(indi_mat, ttf, taf, bounds_list):
    '''
    function computes Mean, Variance, and Skewness for indicator mat
    '''
    len_B = len(bounds_list)
    num_RWs = indi_mat.shape[0]
    Ms = np.zeros((num_RWs, len_B))
    Vs = np.zeros((num_RWs, len_B))
    Ss = np.zeros((num_RWs, len_B))
    for i in range(len_B):
        new_indi = get_dist(indi_mat, ttf, taf, bounds = bounds_list[i])
        for j in range(num_RWs):
            Ms[j,i] = np.mean(new_indi[j,:])
            Vs[j,i] = np.var(new_indi[j,:])
            Ss[j,i] = skew(new_indi[j,:])
    return Ms, Vs, Ss

def compute_pcts_indi(indi_mat, ttf, taf, bounds_list, pcts = [0.1, 0.5, 0.9]):
    '''
    function computes 25, 50, and 75 percentiles, for indicator mat
    '''
    len_B = len(bounds_list)
    num_RWs = indi_mat.shape[0]
    pct_low = np.zeros((num_RWs, len_B))
    pct_mid = np.zeros((num_RWs, len_B))
    pct_hi = np.zeros((num_RWs, len_B))
    for i in range(len_B):
        new_indi = get_dist(indi_mat, ttf, taf, bounds = bounds_list[i])
        for j in range(num_RWs):
            sorted_indi = np.sort(new_indi[j,:])
            dist_len = len(sorted_indi)
            if dist_len > 1:
                pct_low[j,i] = sorted_indi[np.int(dist_len*pcts[0])]
                pct_mid[j,i] = sorted_indi[np.int(dist_len*pcts[1])]
                pct_hi[j,i] = sorted_indi[np.int(dist_len*pcts[2])]
            else:
                pct_low[j,i] = np.nan
                pct_mid[j,i] = np.nan
                pct_hi[j,i] = np.nan
    return pct_low, pct_mid, pct_hi

# In[] 13
def plot_MVS(dataname, col_name, bounds_list = create_Blist(1), mag = 4, 
             mag_UB = 0, radius = 1., save='no'):
    '''
    function plots Mean, Variance, and Skewness of indicators for each data
    for a range of ttf-taf bounds
    col_name has to been either 'col1' or 'col2'
    '''
    fig = plt.figure(figsize=(16,9))
    if mag_UB < mag:
        fig.suptitle('Statistical properties of indicators for ' + 
                     dataname[0:len(dataname)-5]+' '+col_name+
                     ', meansured within 1-day windows along ttf scatter plots' + 
                     '(mag %s, taf_LB %s)'%(mag, bounds_list[0][2]))
    else:
        fig.suptitle('Statistical properties of indicators for ' + 
                     dataname[0:len(dataname)-5]+' '+col_name+
                     ', meansured within 1-day windows along ttf scatter plots' + 
                     '(mag %s to %s, taf_LB %s)'%(mag, mag_UB, bounds_list[0][2]))
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    col = data[col_name]
    RW_lens = col['paras']['RW_lens']
    eqks = eqk()
    eqks.select(dataname, mag, radius = radius)
    t_eqs = eqks.selected[:,0]
    if mag_UB > mag:
        t_eqs = t_eqs[eqks.selected[:,1] < mag_UB]
        print('Mag_UB applied, new #EQs = %s'%len(t_eqs))
    ttf = time2failure(col['t_ind'], t_eqs)
    taf = time_after_failure(col['t_ind'], t_eqs)
    C_M, C_V, C_S = compute_MVS_indi(col['Cs'], ttf, taf, bounds_list)
    V_M, V_V, V_S = compute_MVS_indi(col['Vs'], ttf, taf, bounds_list)
    S_M, S_V, S_S = compute_MVS_indi(col['Ss'], ttf, taf, bounds_list)
    K_M, K_V, K_S = compute_MVS_indi(col['Ks'], ttf, taf, bounds_list)
    summary = [[C_M, C_V, C_S], 
               [V_M, V_V, V_S], 
               [S_M, S_V, S_S], 
               [K_M, K_V, K_S]]
    names = [['Mean(C)', 'Var(C)', 'Skew(C)'], 
             ['Mean(V)', 'Var(V)', 'Skew(V)'], 
             ['Mean(S)', 'Var(S)', 'Skew(S)'], 
             ['Mean(K)', 'Var(K)', 'Skew(K)']]
    for plot_row in range(4):
        for plot_col in range(3):
            plt.subplot(4,3,plot_col+3*plot_row+1)
            for i in range(len(RW_lens)):
                plt.plot(summary[plot_row][plot_col][i,:], label = str(RW_lens[i]))
            plt.ylabel(names[plot_row][plot_col])
            plt.xlabel('LB of ttf (UB=LB+1)')
            plt.legend()
    if save == 'yes':
        if mag_UB > mag:
            plt.savefig('MVSplot_'+dataname[0:len(dataname)-5]+'_'+col_name+
                    '_mag%sto%s_tafLB%s'%(mag, mag_UB, bounds_list[0][2]), dpi = 300)
        else:
            plt.savefig('MVSplot_'+dataname[0:len(dataname)-5]+'_'+col_name+
                        '_mag%s_tafLB%s'%(mag, bounds_list[0][2]), dpi = 300)
        plt.close(fig)
    
def plot_pcts(dataname, col_name, taf_LB = 2, mag = 4, mag_UB = 0, radius = 1., 
              pcts = [0.1, 0.5, 0.9], save='no'):
    '''
    function plots Mean, Variance, and Skewness of indicators for each data
    for a range of ttf-taf bounds
    col_name has to been either 'col1' or 'col2'
    '''
    PCTs = np.int64(np.array(pcts)*100)
    fig = plt.figure(figsize=(16,9))
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    col = data[col_name]
    RW_lens = col['paras']['RW_lens']
    eqks = eqk()
    eqks.select(dataname, mag, radius = 1.)
    t_eqs = eqks.selected[:,0]
    if mag_UB > mag:
        t_eqs = t_eqs[eqks.selected[:,radius] < mag_UB]
        print('Mag_UB applied, new #EQs = %s'%len(t_eqs))
    ttf = time2failure(col['t_ind'], t_eqs)
    taf = time_after_failure(col['t_ind'], t_eqs)
    bounds_list = create_Blist(incre=1, taf_LB=taf_LB, max_ttf=np.ceil(np.nanmax(ttf)))
    C_low, C_mid, C_hi = compute_pcts_indi(col['Cs'], ttf, taf, bounds_list, pcts=pcts)
    V_low, V_mid, V_hi = compute_pcts_indi(col['Vs'], ttf, taf, bounds_list, pcts=pcts)
    S_low, S_mid, S_hi = compute_pcts_indi(col['Ss'], ttf, taf, bounds_list, pcts=pcts)
    K_low, K_mid, K_hi = compute_pcts_indi(col['Ks'], ttf, taf, bounds_list, pcts=pcts)
    summary = [[C_low, C_mid, C_hi], 
               [V_low, V_mid, V_hi], 
               [S_low, S_mid, S_hi], 
               [K_low, K_mid, K_hi]]
    pct_low, pct_mid, pct_hi = '%s'%(PCTs[0]), '%s'%(PCTs[1]), '%s'%(PCTs[2])
    names = [[pct_low+'pct(C)', pct_mid+'pct(C)', pct_hi+'pct(C)'], 
             [pct_low+'pct(V)', pct_mid+'pct(V)', pct_hi+'pct(V)'], 
             [pct_low+'pct(S)', pct_mid+'pct(S)', pct_hi+'pct(S)'], 
             [pct_low+'pct(K)', pct_mid+'pct(K)', pct_hi+'pct(K)']]
    
    if mag_UB < mag:
        fig.suptitle('Percentile (%s,%s,%s) values of indicators for '%(PCTs[0], 
                     PCTs[1], PCTs[2]) + dataname[0:len(dataname)-5] + ' ' + col_name + 
                     ', meansured within 1-day windows along ttf scatter plots' + 
                     '(mag %s, rad %s, taf_LB %s)'%(mag, radius, bounds_list[0][2]))
    else:
        fig.suptitle('Percentile (%s,%s,%s) values of indicators for '%(PCTs[0], 
                     PCTs[1], PCTs[2]) + dataname[0:len(dataname)-5] + ' ' + col_name + 
                     ', meansured within 1-day windows along ttf scatter plots' + 
                     '(mag %s to %s, rad %s, taf_LB %s)'%(mag, mag_UB, radius, 
                      bounds_list[0][2]))
    n_indi = 4 # i.e. C, V, S, K
    n_RW = len(RW_lens)
    n_pcts = 3 # number of percentiles to plot
    for plot_row in range(n_indi):
        for i in range(n_RW):
            plt.subplot(n_indi,n_RW,i+n_RW*plot_row+1)
            for plot_pct in range(n_pcts):
                plt.plot(summary[plot_row][plot_pct][i,:], label=names[plot_row][plot_pct])
            plt.ylabel('RW '+str(np.int(RW_lens[i]/1800))+' hr')
            plt.xlabel('LB of ttf (UB=LB+1)')
            plt.legend()
    pct_name = '%s-%s-%s'%(PCTs[0], PCTs[1], PCTs[2])
    if save == 'yes':
        if mag_UB > mag:
            plt.savefig('PCTplot_'+pct_name+'_'+dataname[0:len(dataname)-5]+'_'+col_name+
                    '_mag%sto%s_rad%s_tafLB%s.png'%(mag, mag_UB, radius, 
                                                bounds_list[0][2]), dpi = 300)
        else:
            plt.savefig('PCTplot_'+pct_name+'_'+dataname[0:len(dataname)-5]+'_'+col_name+
                        '_mag%s_rad%s_tafLB%s.png'%(mag, radius, bounds_list[0][2]), 
                        dpi = 300)
        plt.close(fig)

# In[] 14
def convert_PCTmat(indi_seq, t_indi, dist_WL, R_step = 0):
    '''
    function converts a indicator seq (1D) to a matrix containing 10, 20, ..., 90 
    percentile values of indicator distribution measured in non-overlapping windows 
    of length dist_WL. Note that dist_WL is in #indicators
    '''
    if R_step == 0:
        R_step = np.int(dist_WL/2) # default is Rstep = RW/2
    len_PCTs = np.int(np.floor((len(t_indi)-dist_WL)/R_step))
    t_starts_ExW = np.arange(0, (len_PCTs-1)*R_step+1, R_step)
    t_ends_ExW = t_starts_ExW + dist_WL
    PCT_values = np.int16(np.linspace(10, 90, 9))
    PCT_TSs = np.zeros((9, len_PCTs))    
    PCT_inds_inExW = np.int8(dist_WL*PCT_values/100) # the index for that percentile
    for i in range(len_PCTs):
        ExW = indi_seq[t_starts_ExW[i]:t_ends_ExW[i]] # distribution examine window 
        ExW_sorted = np.sort(ExW)
        ExW_pcts = ExW_sorted[PCT_inds_inExW]
        PCT_TSs[:,i] = ExW_pcts
    t_PCT = t_indi[t_ends_ExW] # the corresponding end time for ExWs
    return PCT_TSs, t_PCT, PCT_values

def compute_tau(PCT_TSs, t_PCT, Tau_WL, R_step = 1):
    '''
    function computes kendau taus of percentiles TSs in matrix 'PCT_TSs'
    with rolling window of length Tau_WL and rolling step 1
    '''
    len_Taus = np.int(np.floor((len(t_PCT)-Tau_WL)/R_step))
    t_starts_ExW = np.arange(0, (len_Taus-1)*R_step+1, R_step)
    t_ends_ExW = t_starts_ExW + Tau_WL
    Taus = np.zeros((PCT_TSs.shape[0], len_Taus))
    x_Tau_WL = np.arange(Tau_WL)
    for i in range(len_Taus):
        for j in range(PCT_TSs.shape[0]):
            ExW = PCT_TSs[j, t_starts_ExW[i]:t_ends_ExW[i]]
            Taus[j, i], _ = kendalltau(x_Tau_WL, ExW)
    t_Taus = t_PCT[t_ends_ExW]
    return Taus, t_Taus

# In[] 15
def scatplot_Taus(dataname, col_name = 'col1', indi_name = 'Cs', RW_ind = 0, dist_WL = 72, 
                  Tau_WL = 42, mag = 4, mag_UB = 0, radius = 1., taf_LB=2, save='no'):
    '''
    assumming indicator RStep 20 min
    RW_ind: index of RW_len in data[col_name]['paras']['RW_lens']
    '''
    with open('D_'+dataname+'_ti2s_v524_th50_n30_trimmed_CVSK.pickle', 'rb') as f:
        data = pickle.load(f)
    indi_mat = data[col_name][indi_name]
    t_indi = data[col_name]['t_ind']-365
    dist_WL = dist_WL # 1 day have 72 indicators
    indi_seq = indi_mat[RW_ind,:]
    RW_name = 'RWlen%dhr'%(data[col_name]['paras']['RW_lens'][RW_ind]/1800)
    PCT_TSs, t_PCT, PCT_values = convert_PCTmat(indi_seq, t_indi, dist_WL)
    Taus, t_Taus = compute_tau(PCT_TSs, t_PCT, Tau_WL)
    eqks = eqk()
    eqks.select(dataname, mag, radius)
    eqs_name = 'mag%s'%mag
    if mag_UB > mag:
        eqks.selected = eqks.selected[eqks.selected[:,1] < mag_UB, :]
        eqs_name += 'to%s'%mag_UB
    eqs_name += '_rad%s'%radius
    ttf = time2failure(t_Taus, eqks.selected[:,0])
    taf = time_after_failure(t_Taus, eqks.selected[:,0])
    Taus_filtered = taf_filter(Taus, taf, lower_bound=taf_LB)
    fig = plt.figure(figsize=(16,9))    
    fig_name = 'ScatTau_'+RW_name+'_'+eqs_name+'_'+dataname+'_'+col_name+'_'+indi_name
    fig_name += '_distWL%dhr_TauWL%dday_tafLB%s'%(dist_WL/3, 
                                                  Tau_WL*dist_WL/144, taf_LB)
    ttf[ttf < -0.1] = np.nan
    fig.suptitle(fig_name)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.scatter(ttf, Taus_filtered[i,:], s = 4, alpha=0.4)
        plt.ylim((-0.95, 0.95))
        if i >= 6:
            plt.xlabel('TTF')
        plt.ylabel('Tau at %s percentile'%PCT_values[i])
    if save == 'yes':
        plt.savefig(fig_name+'.png', dpi = 300)
        plt.close(fig)

# In[] 16
def compute_fs(t, eq_mat, WL_fs=10):
    '''
    function computes future seismicity measured in future window of length WL_fs
    t is 1-D time stamp array
    eq_mat: fist col time, second col magnitude
    WL_fs: the future window to compute FS, in days
    returns fs, in each entries is the log10(sum({mags in window}^10))
    '''
    fs = np.zeros((len(t), ))
    t_eq = eq_mat[:,0]
    mags = eq_mat[:,1]
    energy = np.power(mags, 10)
    for i in range(len(t)):
        if t[i] + WL_fs <= t_eq[-1]:
            a = t_eq>t[i]
            b = t_eq<(t[i]+WL_fs)
            fs[i] = np.log10(np.sum(energy[a & b]))
        else:
            fs[i] = np.nan
    return fs

def scatplot_Taus_fs(dataname, col_name = 'col1', indi_name = 'Cs', RW_ind = 0, 
                     dist_WL = 72, Tau_WL = 42, radius = 1., WL_fs=10, save='no'):
    '''
    assumming indicator RStep 20 min
    RW_ind: index of RW_len in data[col_name]['paras']['RW_lens']
    '''
    with open('D_'+dataname+'_ti2s_v524_th50_n30_trimmed_CVSK.pickle', 'rb') as f:
        data = pickle.load(f)
    indi_mat = data[col_name][indi_name]
    t_indi = data[col_name]['t_ind']-365
    dist_WL = dist_WL # 1 day have 72 indicators
    indi_seq = indi_mat[RW_ind,:]
    RW_name = 'RWlen%dhr'%(data[col_name]['paras']['RW_lens'][RW_ind]/1800)
    PCT_TSs, t_PCT, PCT_values = convert_PCTmat(indi_seq, t_indi, dist_WL)
    Taus, t_Taus = compute_tau(PCT_TSs, t_PCT, Tau_WL)
    eqks = eqk()
    eqks.select(dataname, 0, radius)
    eqs_name = 'rad%s'%radius
    fs = compute_fs(t_Taus, eqks.selected, WL_fs)
    fig = plt.figure(figsize=(16,9))    
    fig_name = 'ScatTau_'+RW_name+'_'+eqs_name+'_'+dataname+'_'+col_name+'_'+indi_name
    fig_name += '_distWL%dhr_TauWL%dday_FsWL%d'%(dist_WL/3, Tau_WL*dist_WL/144, WL_fs)
    fig.suptitle(fig_name)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.scatter(fs, Taus[i,:], s = 4, alpha=0.4)
        plt.ylim((-0.95, 0.95))
        if i >= 6:
            plt.xlabel('Log(Summed Energy)')
        plt.ylabel('Tau at %s percentile'%PCT_values[i])
    if save == 'yes':
        plt.savefig(fig_name+'.png', dpi = 300)
        plt.close(fig)

# In[] 17
def fs_scatplot(dataname, col_name, radius, WL_fs, save='no'):
    with open('D_'+dataname+'_ti2s_v524_th50_n30_trimmed_CVSK.pickle', 'rb') as f:
        data = pickle.load(f)
    eqks = eqk()
    eqks.select(dataname, 0, radius)
    eqs_name = 'rad%s'%radius
    fig = plt.figure(figsize=(16,9))    
    fig_name = 'ScatFS_'+eqs_name+'_'+dataname+'_'+col_name+'_rad%s_FsWL%d'%(radius, 
                                                                             WL_fs)
    fig.suptitle(fig_name)
    t_indi = data[col_name]['t_ind']-365
    fs = compute_fs(t_indi, eqks.selected, WL_fs)
    RW_lens = data[col_name]['paras']['RW_lens']
    subplot_ind = 1
    for indi_name in ['Cs', 'Vs', 'Ss', 'Ks']:
        indi_mat = data[col_name][indi_name]
        plt.subplot(2,2,subplot_ind)
        subplot_ind += 1
        for i in range(indi_mat.shape[0]):
            plt.scatter(fs, indi_mat[i,:], label = 'RW len = ' +str(RW_lens[i]), 
                        s = 4, alpha=0.4)
            plt.legend()
            plt.xlabel('FS')
            plt.ylabel(indi_name)
    if save == 'yes':
        plt.savefig(fig_name+'.png', dpi = 300)
        plt.close(fig)

# In[] 18
def mktrend_n_save(name, bandwidth):
    with open('D_'+name+'.pickle', 'rb') as f:
        data = pickle.load(f)
    Mat = data['Mat']
    tic = time.time()
    trend_col1 = gaussian_filter(Mat[:,1], bandwidth)
    elapsed = time.time() - tic
    tic += elapsed
    print(name+' col 1 done, elapsed = %d s.'%elapsed)
    trend_col2 = gaussian_filter(Mat[:,2], bandwidth)
    elapsed = time.time() - tic
    print(name+' col 2 done, elapsed = %d s.'%elapsed)
    Matc = Mat.copy()
    Matc[:,1] = trend_col1
    Matc[:,2] = trend_col2
    result = {'Mat': Matc}
    with open('D_'+name+'_trendBW%d.pickle'%bandwidth, 'wb') as f:
        pickle.dump(result, f)

def mkres_n_save(name, bandwidth):
    with open('D_'+name+'.pickle', 'rb') as f:
        data = pickle.load(f)
    Mat = data['Mat']
    tic = time.time()
    trend_col1 = gaussian_filter(Mat[:,1], bandwidth)
    elapsed = time.time() - tic
    tic += elapsed
    print(name+' col 1 done, elapsed = %d s.'%elapsed)
    trend_col2 = gaussian_filter(Mat[:,2], bandwidth)
    elapsed = time.time() - tic
    print(name+' col 2 done, elapsed = %d s.'%elapsed)
    Matc = Mat.copy()
    Matc[:,1] = Mat[:,1] - trend_col1
    Matc[:,2] = Mat[:,2] - trend_col2
    result = {'Mat': Matc}
    with open('D_'+name+'_resBW%d.pickle'%bandwidth, 'wb') as f:
        pickle.dump(result, f)
        
# In[] 19
def seq2score(seq, threshold):
    '''
    converts a seq to 0-or-1 score seq, 1 if value above threshold
    '''
    score_seq = np.empty((len(seq), ))
    score_seq[:] = np.nan
    score_seq[seq > threshold] = 1
    score_seq[seq <= threshold] = 0
    return score_seq

def compute_GWMtau(score_window, pre_ignore_rate = 0.5):
    '''
    for a 0-or-1 score seq window, compute growing-windowed mean seq of same length,
    each window starts at time 0, with ending time growing to max
    then compute the kendall-tau of growing-windowed mean seq, and ignoring 
    pre_ignore_rate*100% of starting entries in growing_windowed_mean for stability
    '''
    pre_ignore = np.int(pre_ignore_rate*len(score_window))
    growing_windowed_mean = np.zeros((len(score_window), ))
    for i in range(len(score_window)):
        growing_windowed_mean[i] = np.nanmean(score_window[0:i+1])
    tau, _ = kendalltau(np.arange(pre_ignore, len(score_window)), 
                        growing_windowed_mean[pre_ignore:len(score_window)])
    if np.isnan(tau):
        tau = 0
#    plt.plot(growing_windowed_mean[pre_ignore:len(score_window)])
#    plt.plot(score_window[pre_ignore:len(score_window)])
    return tau

def syscompute_GWMtau(indi_seq, indi_t, threshold, Tau_WL = 72*7, R_step = 1, 
                      pre_ignore_rate = 0.5):
    '''
    systematically compute GWMtau using rolling windows
    '''
    len_Taus = np.int(np.floor((len(indi_t)-Tau_WL)/R_step))
    t_starts_ExW = np.arange(0, (len_Taus-1)*R_step+1, R_step)
    t_ends_ExW = t_starts_ExW + Tau_WL
    Taus = np.zeros((len_Taus, ))
    score_seq = seq2score(indi_seq, threshold)
    for i in range(len_Taus):
        ExW = score_seq[t_starts_ExW[i]:t_ends_ExW[i]]
        Taus[i] = compute_GWMtau(ExW, pre_ignore_rate)
    t_Taus = indi_t[t_ends_ExW]
    return Taus, t_Taus

# In[] 20
def count_GWMTaus(Taus, verbose = 'no'):
    '''
    function counts along Taus, how many are positive, how many are negative
    '''
    positive = np.sum(Taus > 0)
    negative = np.sum(Taus < 0)
    if verbose == 'yes':
        print('Taus count: positive=%s, negative=%s, posi/nega=%.3f'
              %(positive, negative, positive/negative))
    return positive, negative

def count_eqs_GWMTau(t_eqs, t_Taus, Taus, verbose = 'no'):
    '''
    function counts how mamy eqs lie on positive, how many eqs lie on negative
    '''
    positive = 0
    negative = 0
    undecided = 0
    ind_t_Taus = np.arange(len(t_Taus))
    for i in range(len(t_eqs)):
        t_eq = t_eqs[i]
        temp = ind_t_Taus[t_Taus > t_eq]
        if len(temp) == 0:
            continue
        t_eq_round_up = np.min(temp)
        if t_eq_round_up == 0:
            continue
        t_eq_round_down = t_eq_round_up - 1
        if  Taus[t_eq_round_down] > 0:
            positive += 1
        elif Taus[t_eq_round_down] < 0:
            negative += 1
        else:
            undecided += 1 
    if verbose == 'yes':
        print('Eq count: positive=%s, negative=%s, undecided=%s, posi/nega=%.3f'
              %(positive, negative, undecided, positive/negative))
    return positive, negative, undecided

def compute_EBratio(Taus, t_Taus, t_eqs, verbose = 'no'):
    '''
    function computes (eq's PN ratio) over (background's PN ratio)
    where PN is positive/nagitve, as EBratio
    EPratio = 1 for pure random guesses, > 1 for positive predictive power
    '''
    positive, negative, undecided = count_eqs_GWMTau(t_eqs, t_Taus, Taus)
    posi, nega = count_GWMTaus(Taus)
    if negative > 0:
        PNratio_eq = positive/negative
    else:
        PNratio_eq = np.nan
    if nega > 0:
        PNratio_background = posi/nega
    else:
        PNratio_background = np.nan
    EBratio_PNratio = PNratio_eq/PNratio_background
    if verbose == 'yes':
        print("eq/background's PNratio = %.3f. "%EBratio_PNratio)
    return EBratio_PNratio

# In[] 21
def crop_tod(col, tod_LB = 7/24, tod_UB = 17/24):
    '''
    time-of-day
    DABA's GE data behaves strangely during night, and only good for 0700 - 1700 hrs.
    this function crops indicators so that only day-time data are left
    '''
    tod = col['t_ind'] - np.floor(col['t_ind'])
    keep = np.logical_and(tod < tod_UB, tod > tod_LB)
    col_new = col.copy()
    col_new['Cs'] = col_new['Cs'][:, keep]
    col_new['Vs'] = col_new['Vs'][:, keep]
    col_new['Ss'] = col_new['Ss'][:, keep]
    col_new['Ks'] = col_new['Ks'][:, keep]
    col_new['inds'] = col_new['inds'][keep]
    col_new['t_ind'] = col_new['t_ind'][keep]
    return col_new

def tod_scatplot(col, save_info = str(), save = 'no'):
    '''
    in case of saving figure, the figure will not appear, but saved directly
    '''
    fig = plt.figure(figsize=(16,9))
    plot_name = 'tod_scat_'+col['dataname'][2:6] + '_' + save_info
#    plot_name = plot_name + '_' + save_info
    fig.suptitle(plot_name)
    tod = col['t_ind'] - np.floor(col['t_ind'])
#    print('Time axis adjusted by -365')
    C_map = col['Cs']
    RW_lens = col['paras']['RW_lens']
    plt.subplot(2,2,1)
    indi_C = col['Cs']
    for i in range(C_map.shape[0]):
        plt.scatter(tod*24, indi_C[i,:], label = 'RW len = ' +str(RW_lens[i]), 
                    s = 4, alpha=0.4)
    plt.legend()
    plt.xlabel('time of day (hr)')
    plt.ylabel('C')
    plt.subplot(2,2,2)
    indi_V = col['Vs']
    for i in range(C_map.shape[0]):
        plt.scatter(tod*24, indi_V[i,:], label = 'RW len = ' +str(RW_lens[i]), 
                    s = 4, alpha=0.4)
    plt.legend()
    plt.xlabel('time of day (hr)')
    plt.ylabel('V')
    plt.subplot(2,2,3)
    indi_S = col['Ss']
    for i in range(C_map.shape[0]):
        plt.scatter(tod*24, indi_S[i,:], label = 'RW len = ' +str(RW_lens[i]), 
                    s = 4, alpha=0.4)
    plt.legend()
    plt.xlabel('time of day (hr)')
    plt.ylabel('S')
    plt.subplot(2,2,4)
    indi_K = col['Ks']
    for i in range(C_map.shape[0]):
        plt.scatter(tod*24, indi_K[i,:], label = 'RW len = ' +str(RW_lens[i]), 
                    s = 4, alpha=0.4)
    plt.legend()
    plt.xlabel('time of day (hr)')
    plt.ylabel('K')
    if save == 'no':
        pass
    else:
        plt.savefig(plot_name+'.png', dpi = 300)
        plt.close(fig)

# In[22] KS matrix computation and plotting
def compute_KS_matrix(indi, ttf, taf, taf_LB = 2, incre = 0, divide = 0):
    '''
    function computes the KS-p value matrix among different ttf intervals
    p value high (>0.05): cannot reject hypothesis, same distribution
    indi: 1-D indicator array of (len,)
    incre: increment of each ttf interval (but yield very different sample size)
    divide: on sorted ttf, divide (how many times) the sample equally
    1 and only 1 of incre, divide must be 0
    '''
    if incre > 0:
        ttf_UB = np.int(np.floor(np.max(ttf)) - 1)
        bounds_list = create_Blist(incre = incre, taf_LB = taf_LB, max_ttf = ttf_UB)
        mat_dim = np.int(np.floor(ttf_UB/incre))
        indi_dists = list()
        for i in range(mat_dim):
            indi_dists.append(get_dist(indi, ttf, taf, bounds = bounds_list[i]))
        KS_mat = np.zeros((mat_dim, mat_dim))
    elif divide > 0:
        ttf_indi_taf = np.zeros((len(ttf), 3))
        ttf_indi_taf[:,0] = ttf
        ttf_indi_taf[:,1] = indi
        ttf_indi_taf[:,2] = taf
        ttf_indi_taf = ttf_indi_taf[np.argsort(ttf_indi_taf[:,0])]
        ttf_indi_taf = ttf_indi_taf[ttf_indi_taf[:,2] > taf_LB]
        sample_size = np.int(np.floor(len(ttf_indi_taf)/divide)-5)
        indi_dists = list()
        ttf_division_points = np.zeros((divide,))
        for i in range(divide):
            indi_dists.append(ttf_indi_taf[i*sample_size:(i+1)*sample_size, 1])
            ttf_division_points[i] = ttf_indi_taf[(i+1)*sample_size, 0]
        KS_mat = np.zeros((divide, divide))
        mat_dim = divide
    for i in range(mat_dim):
        for j in range(mat_dim):
            KS_mat[i, j] = stats.ks_2samp(indi_dists[i], indi_dists[j])[1]
            # [1] for p-value, [0] for KS-statistics
    if incre > 0:
        return KS_mat
    elif divide > 0:
        return KS_mat, ttf_division_points

def plot_KS_matrix(dataname, mag = 4, radius = 0.8, taf_LB = 2, incre = 0, 
                   divide = 0, tod_LB = 7/24, tod_UB = 17/24, end_crop = 0,
                   savename = str()):
    '''
    function plots KS-distance matrix at varying mag-rad
    '''
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    eqks = eqk()
    eqks.select(dataname, mag, radius = radius)
    t_eqs = eqks.selected[:,0]    
    for col_name in ['col1', 'col2']:
        col = crop_tod(data[col_name], tod_LB, tod_UB)
        ttf = time2failure(col['t_ind'], t_eqs)
        taf = time_after_failure(col['t_ind'], t_eqs)
        fig = plt.figure(figsize = (16.5,9.5))
        fig_name = 'KSmat_'+dataname+'_'+col_name+'_mag%s_rad%s_tafLB%s'%(mag, 
                   radius, taf_LB)+'_%sdivides_endcrop%s_'%(divide,
                   end_crop)+savename
        fig.suptitle(fig_name)
        plot_num = 0
        for indi_name in ['Cs', 'Vs', 'Ss', 'Ks']:
            plot_num += 1
            ax = plt.subplot(2,2,plot_num)
            indi = col[indi_name][0,:]
            if incre > 0:
                KS_mat = compute_KS_matrix(indi, ttf, taf, taf_LB, incre = incre)
                sns.heatmap(KS_mat)
                ax.set_xticks(np.arange(len(KS_mat)))
                ax.set_yticks(np.arange(len(KS_mat)))
            elif divide > 0:
                KS_mat, div_pts = compute_KS_matrix(indi, ttf, taf, taf_LB, 
                                                    divide = divide)
                div_pts = np.round(div_pts*10)/10
                sns.heatmap(KS_mat)
                ax.set_xticks((np.arange(len(KS_mat))+1)[::2])
                ax.set_xticklabels(div_pts[::2], rotation = -90)
                ax.set_yticks((np.arange(len(KS_mat))+1)[::2])
                ax.set_yticklabels(div_pts[::2], rotation = 0)
            threshold, reject_rate = KSmat_analysis(KS_mat, end_crop = end_crop)
            threshold_day = div_pts[np.int(threshold)]
            plt.title(indi_name+'_threshold%s_rejectrate%.3f'%(threshold_day, 
                                                               reject_rate))
        if len(savename) > 0:
            plt.savefig(fig_name+'.png', dpi = 300)
            plt.close(fig)
            
# In[23] KS matrix analysis with reject_rate
def KSmat_analysis(KS_mat, reject_threshold = 0.05, end_crop = 2):
    '''
    function finds out the rectangular section of matrix just under diagonal
    such that number of entries within has max rate of hypothesis rejection
    the two division blocks are [0 to threshold], [threshold+1 to end]
    '''
    reject_rates = np.zeros((len(KS_mat)-1,))
    for i in range(len(reject_rates)):
        mat_selected = KS_mat[i+1:len(KS_mat), 0:i+1]
        reject_rates[i] = np.sum(mat_selected < reject_threshold)/mat_selected.size
    reject_rates = reject_rates[0:len(reject_rates)-end_crop]
    threshold = np.argmax(reject_rates)
    reject_rate = np.max(reject_rates)
    return threshold, reject_rate

def KSmat_analysis_para(dataname, col_name = 'col1', indi_name = 'Cs', 
                        mags = [4, 4.5], radii = [0.8, 1.], taf_LB = 2, divide = 30):
    '''
    function computes optimal reject_rates and corresponding TTF thresholds
    at varying mag-rad
    '''
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    col = data[col_name]
    indi = col[indi_name][0,:]
    thresholds = np.zeros((len(mags), len(radii)))
    reject_rates = np.zeros((len(mags), len(radii)))
    eqks = eqk()
    for i in range(len(mags)):
        for j in range(len(radii)):
            mag = mags[i]
            radius = radii[j]
            eqks.select(dataname, mag, radius = radius)
            t_eqs = eqks.selected[:,0]
            ttf = time2failure(col['t_ind'], t_eqs)
            taf = time_after_failure(col['t_ind'], t_eqs)
            KS_mat, div_pts = compute_KS_matrix(indi, ttf, taf, taf_LB, 
                                                divide = divide)
            threshold, reject_rate = KSmat_analysis(KS_mat)
            threshold_day = div_pts[np.int(threshold)]
            thresholds[i,j] = threshold_day
            reject_rates[i,j] = reject_rate
    return thresholds, reject_rates

# In[24] KS matrix analysis with modularity
def KSmat_modularity(KS_null, end_crop = 1):   
    '''
    (a mini-function for KSmat_modula_1para_plot and KSmat_modulamap_paras)
    (KS_null = np.int64(KS_mat < reject_threshold))
    function computes modularities at different cutting thresholds for KS matrix
    KS_null shoulb be n-by-n connectivity matrix, can be in logical/double/integer
    '''
    KS_null = np.int64(KS_null)
    dim = len(KS_null)
    thresholds = np.arange(end_crop, dim-end_crop-1)
    modulas = np.zeros((len(thresholds), )) # modularities
    degs = -np.ones((dim, )) # degree of each node
    for i in range(dim):
        for j in range(dim):
            degs[i] += KS_null[i,j]
    total_links = np.sum(degs)/2
    for i_th in range(len(thresholds)):
        threshold = thresholds[i_th]
        status = np.arange(dim) <= threshold
        for i in range(dim):
            for j in range(i):
                if status[i] == status[j]:
                    modulas[i_th] += (KS_null[i,j] - degs[i]*degs[j]/(2*total_links))
    modulas = modulas/(2*total_links)
    return thresholds, modulas

def KSmat_modula_1para_plot(dataname, col_name = 'col1', indi_name = 'Cs', mag = 3.8, 
                            radius = 0.6, divide = 30, end_crop = 1, taf_LB = 2):
    '''
    function computes and plots the modularity map for given single mag-radius
    at different KS-matrix spliting thresholds
    '''
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    col = data[col_name]
    indi = col[indi_name][0,:]
    eqks = eqk()
    eqks.select(dataname, mag, radius = radius)
    t_eqs = eqks.selected[:,0]
    ttf = time2failure(col['t_ind'], t_eqs)
    taf = time_after_failure(col['t_ind'], t_eqs)
    KS_mat, div_pts = compute_KS_matrix(indi, ttf, taf, taf_LB, divide = divide)
    KS_connect = KS_mat >= 0.05 # connectivity matrix
    # p value above 0.05 means cannot tell two samples from different distribution
    thresholds, modulas = KSmat_modularity(KS_connect, end_crop = end_crop) 
    # modularities at different cutting thresholds
    threshold_days = div_pts[thresholds]
    # plotting part
    plt.figure(figsize=(18,7))
    div_pts = np.round(div_pts*10)/10
    ax = plt.subplot(1,2,1)
    sns.heatmap(KS_mat)
    ax.set_xticks((np.arange(len(KS_mat))+1)[::2])
    ax.set_xticklabels(div_pts[::2], rotation = -90)
    ax.set_yticks((np.arange(len(KS_mat))+1)[::2])
    ax.set_yticklabels(div_pts[::2], rotation = 0)
    ax = plt.subplot(1,2,2)
    sns.heatmap(KS_connect)
    ax.set_xticks((np.arange(len(KS_mat))+1)[::2])
    ax.set_xticklabels(div_pts[::2], rotation = -90)
    ax.set_yticks((np.arange(len(KS_mat))+1)[::2])
    ax.set_yticklabels(div_pts[::2], rotation = 0)
    optimal_thre = threshold_days[np.argmax(modulas)]
    highest_modu = modulas[np.argmax(modulas)]
    print('Highest modularity of %.4f at threshold %.4f (days)'%(highest_modu, 
                                                                 optimal_thre))
    return threshold_days, modulas

def KSmat_modulamap_paras(dataname, col_name = 'col1', indi_name = 'Cs', 
                          mags = [4, 4.5], radii = [0.8, 1.], end_crop = 1, 
                          taf_LB = 2, divide = 40):
    '''
    function computes optimal-modularity map for at varying mag-radius
    end_crop is for ignored No. of modularity computation at both ends
    optimal-modularity: the highest mod, by cross comparing mods with half-spliting 
    the KS matrix with increasing thresholds
    '''
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    col = data[col_name]
    indi = col[indi_name][0,:]
    thresholds_map = np.zeros((len(mags), len(radii)))
    modulas_map = np.zeros((len(mags), len(radii)))
    eqks = eqk()
    for i in range(len(mags)):
        for j in range(len(radii)):
            mag = mags[i]
            radius = radii[j]
            eqks.select(dataname, mag, radius = radius)
            t_eqs = eqks.selected[:,0]
            ttf = time2failure(col['t_ind'], t_eqs)
            taf = time_after_failure(col['t_ind'], t_eqs)
            KS_mat, div_pts = compute_KS_matrix(indi, ttf, taf, taf_LB, 
                                                divide = divide)
            KS_connect = KS_mat >= 0.05 # connectivity matrix
            # p > 0.05 means cannot tell two samples from different distribution
            thresholds_para, modulas_para = KSmat_modularity(KS_connect, 
                                                             end_crop = end_crop) 
            # modularities at different cutting thresholds
            threshold_days = div_pts[thresholds_para]            
            thresholds_map[i,j] = threshold_days[np.argmax(modulas_para)]
            modulas_map[i,j] = modulas_para[np.argmax(modulas_para)]
    opti_ind = np.unravel_index(np.argmax(modulas_map), modulas_map.shape)
    summary = {'opti_mod': modulas_map[opti_ind],
               'opti_thre': thresholds_map[opti_ind],
               'opti_mag': mags[opti_ind[0]],
               'opti_rad': radii[opti_ind[1]],
               'thresholds_map': thresholds_map, 
               'modulas_map': modulas_map}
    print('Highest modularity of %.4f at threshold %.4f (days)'%(summary['opti_mod'],
          summary['opti_thre']) + ' for mag %.2f, radius %.2f.'%(summary['opti_mag'], 
          summary['opti_rad']))
    return summary

def sys_KSmat_modulamap_paras(dataname='DABA_full_ti2_fcln3_nonan_CVSK600_7to17',
                              mags = np.arange(3.6, 4.7, 0.1), 
                              radii = np.arange(0.3, 1.3, 0.1), 
                              divides = [30, 40, 50],
                              save_name = 'modsum_mmdd'):
    '''
    systematically call KSmat_modulamap_paras to compute and save mag-rad heatmap of 
    highest modularity and corresponding TTF
    '''
    modula_test_result = dict()
    for col_name in ['col1', 'col2']:
        for indi_name in ['Cs', 'Vs', 'Ss', 'Ks']:
            for divide in divides:
                summary = KSmat_modulamap_paras(dataname, col_name = col_name, 
                                                indi_name = indi_name, 
                                                divide = divide, 
                                                mags = mags, radii = radii)
                modula_test_result[col_name+'_'+indi_name+'_divide%s'%divide]=summary
    modula_test_result['mags'] = mags
    modula_test_result['radii'] = radii
    with open(dataname + '_' + save_name + '.pickle', 'wb') as f:
        pickle.dump(modula_test_result, f)
        
def plot_modularity_summary(data_name, divides = [30,40,50], cols = ['col1','col2'],
                            save = 'yes'):
    '''
    data_name have to be the saved dataname from sys_KSmat_modulamap_paras
    function plots mag-rad heatmap of highest modularity and corresponding TTF
    '''
    with open(data_name+'.pickle', 'rb') as f:
        results = pickle.load(f)
    for col_name in cols:
        for divide in divides:
            indi_names = ['Cs', 'Vs', 'Ss', 'Ks']
            fig = plt.figure(figsize=(18,9))
            fig.suptitle('Modularity summary '+col_name+' divide%s'%divide, 
                         fontsize = 16)
            mags = np.round(results['mags']*10)/10
            radii = np.round(results['radii']*10)/10
            plot_num = 1
            for i in range(len(indi_names)):
                result = results[col_name+'_'+indi_names[i]+'_divide%s'%divide]
                plt.subplot(2,4,plot_num)
                ax1 = sns.heatmap(result['modulas_map']*100, annot=True, cbar=False)
                ax1.set_xticklabels(radii)
                ax1.set_yticklabels(mags)
                plt.title('Max modularity (*0.01) mag-rad map '+indi_names[i][0])
                plt.subplot(2,4,plot_num+1)
                ax2 = sns.heatmap(result['thresholds_map'], annot=True, cbar=False)
                ax2.set_xticklabels(radii)
                ax2.set_yticklabels(mags)
                plot_num += 2
                plt.title('TTF thresholds mag-rad map '+indi_names[i][0])
            fig.tight_layout(pad=1.2, rect = (0,0,1,0.96))
            if save == 'yes':
                plt.savefig(data_name+'_'+col_name+'_divide%s'%divide+'.png', 
                            dpi = 300)
                plt.close(fig)

# In[25]
def get_seg_pts(t_start, t_end, seglen_max = 5, seglen_min = 4, max_overlap = 1):
    '''
    function splits indicator TS in indic_TS_nega_admit in between EQs, into segments
    to make negative ensembles. Segments can either overlap or not
    longest segment allowed, seglen_max = 5
    shortest segment allowed, seglen_min = 4
    max overlapping between two consecutive segments, max_overlap = 1
    priority: prefering overlapping than not overlapping (potentially waste data)
    e.g.: ****                             *****
             ****           rather than         *****
                ****                                         (for 10-long-TS_part)
    '''
    len_TSpart = t_end - t_start
    if len_TSpart >= seglen_min:
        num_seg = np.floor((len_TSpart - max_overlap)/(seglen_min - max_overlap))
        if num_seg > 1:
            seglen = seglen_min
            overlap = (num_seg*seglen_min - len_TSpart)/(num_seg-1)
            if overlap <= 0:
                # a negative overlap means consecutive segments are not 'in contact', 
                # therefore segs can be longer (so overlap reach 0) while <= seglen_max 
                seglen = min(len_TSpart/num_seg, seglen_max)
                overlap = (num_seg*seglen - len_TSpart)/(num_seg-1) # update
            seg_pts = np.zeros((np.int(num_seg), 2)) #the segmentation secheme
            for i in range(np.int(num_seg)):
                seg_pts[i,0] = t_start + i*(seglen-overlap)
                seg_pts[i,1] = t_start + i*(seglen-overlap) + seglen
        else: # i.e. num_seg = 1
            seglen = min(len_TSpart, seglen_max)
            mid_pt = (t_end + t_start)/2
            seg_pts = np.zeros((1,2))
            seg_pts[0,0] = mid_pt-seglen/2
            seg_pts[0,1] = mid_pt+seglen/2
    else:
        seg_pts = None
    return seg_pts

# In[26]
def cut_distends(dist, pct = 0.5):
    '''
    function removes 'pct' percent of data points at both ends of a distribution
    given by 1-D array seq
    '''
    seq = dist.copy()
    if len(seq.shape) == 2:
        seq = seq.ravel()
    len_seq = len(seq)
    seq = np.sort(seq)
    n_remove = np.int(np.ceil(len_seq*pct/100))
    seq = seq[n_remove:len_seq-n_remove]
    return seq

# In[27]
def ensemble_analysis(dataname, mag = 4.2, rad = 0.6, col_name = 'col1', 
                      indi_name = 'Cs', posi_ttf_UB = 5, nega_ttf_LB = 7,
                      nega_taf_UB = 2, seglen = 50, max_overlap = 2, plot = 'yes', 
                      save = 'no', mode = 0):
    '''
    function calculates and plots positive (near EQ) and negative (no near EQ)
    ensembles, for given dataname's one col's one indicator
    nega_ttf_LB: the LB of ttf for deciding negative parts of TS (no near EQ)
    nega_taf_UB: 'no near EQ' also require not being immediately after previous EQ
    posi_ttf_UB: the UB of ttf for deciding positive parts of TS (near EQ)
    seglen: the length of the segments for ensemble members, unit: data points
    max_overlap: maximum allowed overlap between segments, unit: data points
    '''
    # ----- load data ----- 
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    eqks = eqk()
    eqks.select(dataname, mag, radius = rad)
    t_eqs = eqks.selected[:,0]
    col = data[col_name]
    ttf = time2failure(col['t_ind'], t_eqs)
    taf = time_after_failure(col['t_ind'], t_eqs)
    # ----- separate positive TS and negative TS ----- 
    indic_TS = col[indi_name]
    indic_TS_posi_admit = np.zeros((len(ttf), ), dtype = bool)
    indic_TS_posi_admit[ttf < posi_ttf_UB] = 1
    indic_TS_nega_admit = np.zeros((len(ttf), ), dtype = bool)
    indic_TS_nega_admit[ttf > nega_ttf_LB] = 1
    indic_TS_nega_admit[taf < nega_taf_UB] = 0
#    indic_TS_posi = indic_TS[0, indic_TS_posi_admit] ------------
    indic_TS_posi = indic_TS[indic_TS_posi_admit]
    t_ind_posi = col['t_ind'][indic_TS_posi_admit]
#    indic_TS_nega = indic_TS[0, indic_TS_nega_admit]
    indic_TS_nega = indic_TS[indic_TS_nega_admit]
    t_ind_nega = col['t_ind'][indic_TS_nega_admit]
    # ----- plot positive TS and negative TS ----- 
    if plot == 'yes':
        fig = plt.figure(figsize=(18,9))
        plt.subplot(2,2,(1,2))
        fig_name = dataname+'_'+col_name+'_'+indi_name+'_mag%s_rad%s_seglen%s'%(mag, 
                                                                    rad, seglen)
        plt.title(fig_name)
        plt.plot(t_ind_posi, indic_TS_posi, 'ro', markersize = 1, label = "near-EQ")
        plt.plot(t_ind_nega, indic_TS_nega, 'o', markersize = 1, label ="no near-EQ")
        plt.legend()
        eqks.plot()
    # ----- create positive & negative ensembles ----- 
    posi_ensembles = list()
    posi_ensembles_tind = list()
    nega_ensembles = list()
    nega_ensembles_tind = list()
    nega_parts = list()
    posi_parts = list()
    for i in range(len(t_eqs)):
        if i == 0:
            posi_TSpart = indic_TS_posi[t_ind_posi < t_eqs[i]]
            nega_TSpart = indic_TS_nega[t_ind_nega < t_eqs[i]]
            posi_TSpart_tind = t_ind_posi[t_ind_posi < t_eqs[i]]
            nega_TSpart_tind = t_ind_nega[t_ind_nega < t_eqs[i]]
        else:
            if mode == 0:
                posi_part = np.logical_and(t_ind_posi < t_eqs[i], 
                                           t_ind_posi > t_eqs[i] - posi_ttf_UB)
            elif mode == 1:
                posi_part = np.logical_and(t_ind_posi < t_eqs[i], 
                                           t_ind_posi > t_eqs[i-1])
            posi_TSpart = indic_TS_posi[posi_part]
            nega_part = np.logical_and(t_ind_nega < t_eqs[i], t_ind_nega >t_eqs[i-1])
            nega_TSpart = indic_TS_nega[nega_part]
            posi_TSpart_tind = t_ind_posi[posi_part]
            nega_TSpart_tind = t_ind_nega[nega_part]
        if len(posi_TSpart) > 0:
            seg_pts = get_seg_pts(0, len(posi_TSpart)-1, seglen_max = seglen, 
                                     seglen_min = seglen, max_overlap=max_overlap)
            if len(posi_TSpart) > 0:
                if hasattr(seg_pts, '__len__'):
                    seg_pts = np.int64(seg_pts)
                    for j in range(seg_pts.shape[0]):
                        posi_ensembles.append(posi_TSpart[seg_pts[j,0]:seg_pts[j,1]])
                        posi_ensembles_tind.append(posi_TSpart_tind[seg_pts[j,1]])
                    posi_parts.append(posi_TSpart)
        if len(nega_TSpart) > 0:
                seg_pts = get_seg_pts(0, len(nega_TSpart)-1, seglen_max = seglen, 
                                         seglen_min = seglen,max_overlap=max_overlap)
                if len(nega_TSpart) > 0:
                    if hasattr(seg_pts, '__len__'):
                        seg_pts = np.int64(seg_pts)
                        for j in range(seg_pts.shape[0]):
                            nega_ensembles.append(nega_TSpart[seg_pts[j,0]:seg_pts[j,1]])
                            nega_ensembles_tind.append(nega_TSpart_tind[seg_pts[j,1]])
                        nega_parts.append(nega_TSpart)
    # ----- plot ensemble's mean ----- 
    posi_means = np.zeros((len(posi_ensembles), ))
    for i in range(len(posi_ensembles)):
        posi_means[i] = np.nanmean(posi_ensembles[i])
    nega_means = np.zeros((len(nega_ensembles), ))
    for i in range(len(nega_ensembles)):
        nega_means[i] = np.nanmean(nega_ensembles[i])
    if plot == 'yes':
        plt.subplot(2,2,3)
        plt.title(col_name + "'s " + indi_name + "' ensemble mean")
        sns.distplot(posi_means, label = "near-EQ", color = 'r')
        sns.distplot(nega_means, label = "no near-EQ")
        plt.legend()
    # ----- plot ensemble's var ----- 
    posi_vars = np.zeros((len(posi_ensembles), ))
    for i in range(len(posi_ensembles)):
        posi_vars[i] = np.nanvar(posi_ensembles[i])
    nega_vars = np.zeros((len(nega_ensembles), ))
    for i in range(len(nega_ensembles)):
        nega_vars[i] = np.nanvar(nega_ensembles[i])
    if plot == 'yes':
        plt.subplot(2,2,4)
        plt.title(col_name + "'s " + indi_name + "' ensemble variance")
        sns.distplot(posi_vars, label = "near-EQ", color = 'r')
        sns.distplot(nega_vars, label = "no near-EQ")
        plt.legend()
    info = dict()
    info['posi_means'] = posi_means
    info['nega_means'] = nega_means
    info['posi_vars'] = posi_vars
    info['nega_vars'] = nega_vars
    info['posi_ensembles'] = posi_ensembles
    info['nega_ensembles'] = nega_ensembles
    info['posi_parts'] = posi_parts
    info['nega_parts'] = nega_parts
    info['posi_over_nega'] = len(indic_TS_posi)/len(indic_TS_nega)
    info['t_ind'] = col['t_ind']
    info['posi_ensembles_tind'] = np.array(posi_ensembles_tind)
    info['nega_ensembles_tind'] = np.array(nega_ensembles_tind)
    if plot == 'yes':
        if save == 'yes':
            plt.savefig('EnsemblePlot_'+fig_name+'.png', dpi = 300)
            plt.close(fig)
    return info

# In[28]
def generate_TRnTE(dataname, id_div, total_div = 10):
    '''
    generate and save training and testing data
    TR for training, TE for testing
    '''  
    for type_div in ['TR', 'TE']:
        with open('D_'+dataname+'.pickle', 'rb') as f:
            data = pickle.load(f)
        code_name = '_%sof%s_'%(id_div, total_div) + type_div
        new_data = data.copy()
        # next, make logical_getdata
        data_len = len(data['col1']['inds'])
        len_to_get = np.int(np.floor(data_len/total_div))
        div_start = (id_div - 1)*len_to_get
        div_end = id_div*len_to_get
        if type_div == 'TR':
            logical_getdata = np.ones((data_len,), dtype = bool)
            if id_div < total_div:
                logical_getdata[div_start:div_end] = 0
            else:
                logical_getdata[div_start:] = 0
        elif type_div == 'TE':
            logical_getdata = np.zeros((data_len,), dtype = bool)
            if id_div < total_div:
                logical_getdata[div_start:div_end] = 1
            else:
                logical_getdata[div_start:] = 1
        for col in new_data:
            new_data[col]['dataname'] = data[col]['dataname'] + code_name
            new_data[col]['Cs'] = data[col]['Cs'][:, logical_getdata]
            new_data[col]['Vs'] = data[col]['Vs'][:, logical_getdata]
            new_data[col]['Ss'] = data[col]['Ss'][:, logical_getdata]
            new_data[col]['Ks'] = data[col]['Ks'][:, logical_getdata]
            new_data[col]['inds'] = data[col]['inds'][logical_getdata]
            new_data[col]['t_ind'] = data[col]['t_ind'][logical_getdata]
        with open('D_'+dataname+code_name+'.pickle', 'wb') as f:
            pickle.dump(new_data, f)
        print('D_'+dataname+code_name+' saved.')
        
# In[29]
def clean_V(dataname_CVSK, mode = 'old', rate = 0.9, naming = '_NcV'):
    '''
    function cleans exploding V values for CVSK files
    input dataname_CVSK without '.pickle', e.g 'D_CHCH_ti2_FCW500S2_CVSK200'
    '''
    with open(dataname_CVSK + '.pickle', 'rb') as f:
        data = pickle.load(f)
    if mode == 'old':
        for col in data:
            Vs = data[col]['Vs'][0,:].copy()
            Vs_sorted = np.sort(Vs)
            Vs_std = np.nanstd(Vs_sorted[0:int(len(Vs)*0.90)])
            data[col]['Vs'][0, Vs>Vs_std*10] = Vs_std
        with open(dataname_CVSK + '_cV.pickle', 'wb') as f:
            pickle.dump(data, f)
    if mode == 'new':
        print('New cV applied')
        for col in data:
            Vs = data[col]['Vs'][0,:].copy()
            Vs_sorted = np.sort(Vs)
            Vs_UB = Vs_sorted[np.int(len(Vs)*rate)]*100
            data[col]['Vs'][0, Vs>Vs_UB] = Vs_UB
        with open(dataname_CVSK + naming +'.pickle', 'wb') as f:
            pickle.dump(data, f)
    elif mode == 'tr':
        print('trV applied')
        for col in data:
            Vs = data[col]['Vs'][0,:].copy()
            Vs_sorted = np.sort(Vs)
            Vs_UB = Vs_sorted[np.int(len(Vs)*0.99)]
            data[col]['Vs'][0, Vs>Vs_UB] = Vs_UB
        with open(dataname_CVSK + '_trV.pickle', 'wb') as f:
            pickle.dump(data, f)
    return

def truncate_CVSK(dataname_CVSK):
    '''
    function truncates CVSK files into same starting point with the EQ catalog file
    input dataname_CVSK without '.pickle', e.g 'D_CHCH_ti2_FCW500S2_CVSK200(_cV)'
    '''
    eqks = eqk()
    t_start = np.floor(eqks.mat[0,0])
    t_end = np.floor(eqks.mat[-1,0]+1) 
    # 't_end' added on 24 Apri, update save name to _4YRs from _4yrs
    with open(dataname_CVSK + '.pickle', 'rb') as f:
        data = pickle.load(f)
    for col in data:
        t_ind = data[col]['t_ind']
        accept = np.ones((len(t_ind),), dtype = bool)
        accept[t_ind < t_start] = 0
        accept[t_ind > t_end] = 0
        data[col]['Cs'] = data[col]['Cs'][:,accept]
        data[col]['Vs'] = data[col]['Vs'][:,accept]
        data[col]['Ss'] = data[col]['Ss'][:,accept]
        data[col]['Ks'] = data[col]['Ks'][:,accept]
        data[col]['inds'] = data[col]['inds'][accept]
        data[col]['t_ind'] = data[col]['t_ind'][accept]
    with open(dataname_CVSK + '_4YRs.pickle', 'wb') as f:
        pickle.dump(data, f)
    return