"""
Created on Wed Feb 26 12:06:59 2020
Mod for Baum-Welch and HMM related works
"""

import numpy as np
import matplotlib.pyplot as plt
import EP_modB_new as mB
from scipy import stats
from scipy.optimize import least_squares
import pickle
import seaborn as sns
from sklearn.cluster import KMeans
import os
import matplotlib.dates as pdate
#from mpl_toolkits.basemap import Basemap
import CM_mod as cm
# import pandas as pd

def dist_fun(para, x, y):
    # para: [norm const, x0, k, theta]
    return para[0] * ((x-para[1])**para[2]) * np.exp(-(x-para[1])/para[3]) - y

def estimate_gamma(dist, cut_pct = 1, n_bins = 100, show = [1,1,1], label = None, c = 'r'):
    dist = mB.cut_distends(dist, pct = cut_pct)
    dist_mean = np.nanmean(dist)
    dist_var = np.nanvar(dist)
    dist_skew = stats.skew(dist)
    # estimate three distribution paras
    x0 = dist_mean - 2*np.sqrt(dist_var)/dist_skew
    theta = np.sqrt(dist_var) * dist_skew / 2
    k = 4/(dist_skew)**2
    # plot from three dist paras
    x_bins = np.histogram(dist, n_bins)[1]
    xs = (x_bins[0:-1] + x_bins[1:])/2
    hist_values = np.histogram(dist, n_bins)[0]
    hist_values = hist_values/(np.nansum(hist_values))
    dist_gamma_pdf = ((xs-x0)**(k-1))*np.exp(-(xs-x0)/theta)
    norm_const = np.nansum(dist_gamma_pdf)
    dist_gamma_pdf = dist_gamma_pdf/norm_const
    #sns.distplot(S1_means)
    plt.plot(xs, hist_values, c+'o', label = label)
#    sns.distplot(hist_values, bins = n_bins, label = label)
#    plt.bar(xs, hist_values, width =(xs[1]-xs[0])*0.3)
    #plt.plot(dist_gamma_pdf*1000)
    if show[0] == 1:
        plt.plot(xs,dist_gamma_pdf, c, label = '3-param gamma')    
    # In estimate gamma from regression
    para0 = np.array([norm_const, x0, k, theta])
    res_lsq = least_squares(dist_fun, para0, args=(xs, hist_values))
    dist_gamma_lsq = dist_fun(res_lsq.x, xs, hist_values) + hist_values
    if show[1] == 1:
        plt.plot(xs,dist_gamma_lsq, c, label = 'lsq gamma')
    # In estimate gamma from robust regression
    res_robustlsq = least_squares(dist_fun, para0, loss='soft_l1', f_scale=0.1, 
                               args=(xs, hist_values))
    dist_gamma_robustlsq = dist_fun(res_robustlsq.x, xs, hist_values) + hist_values
    if show[2] == 1:
        plt.plot(xs,dist_gamma_robustlsq, c, label = 'robust lsq gamma')
    plt.legend()
    return

def estimate_gamma_3para(dist, indic_domain, n_bins = 300, plot = 'yes',
                         label = None, c = 'r'):
    dist_mean = np.nanmean(dist)
    dist_var = np.nanvar(dist)
    dist_skew = stats.skew(dist)
    # estimate three distribution paras
    x0 = dist_mean - 2*np.sqrt(dist_var)/dist_skew
    theta = np.sqrt(dist_var) * dist_skew / 2
    k = 4/(dist_skew)**2
    # plot from three dist paras
    x_bins = np.histogram(indic_domain, n_bins)[1]
    xs = (x_bins[0:-1] + x_bins[1:])/2
    hist_values = np.histogram(dist, x_bins)[0]
    hist_values = hist_values/(np.nansum(hist_values))
    dist_gamma_pdf = ((xs-x0)**(k-1))*np.exp(-(xs-x0)/theta)
    norm_const = np.nansum(dist_gamma_pdf)
    dist_gamma_pdf = dist_gamma_pdf/norm_const
    #sns.distplot(S1_means)
    if plot == 'yes':
        plt.plot(xs, hist_values, c+'o', label = label)
        plt.plot(xs,dist_gamma_pdf, c, label = '3-param gamma')
        plt.legend()
        return
    else:
        return dist_gamma_pdf


def histo_construct_B(n_bins, indic_S1, indic_S3, indic_ts):
    '''
    construct B by discretizing continuous values with histograms
    '''
    B = np.zeros((2, n_bins))
    gamma_S1 = estimate_gamma_3para(indic_S1, indic_ts, n_bins = n_bins, 
                                              plot = 'no')
    gamma_S3 = estimate_gamma_3para(indic_S3, indic_ts, n_bins = n_bins, 
                                              plot = 'no')
    # now, np.nansum(gamma_S1/3) = 1, with nans inside. 
    is_nans = np.zeros((n_bins,), dtype = bool)
    is_nans[np.isnan(gamma_S1)] = 1
    is_nans[np.isnan(gamma_S3)] = 1
    gamma_S1[is_nans] = np.nan
    gamma_S3[is_nans] = np.nan
    gamma_S1 = gamma_S1/np.nansum(gamma_S1)
    gamma_S3 = gamma_S3/np.nansum(gamma_S3)
    
    emi_nans = np.max([np.nanmin(gamma_S1), np.nanmin(gamma_S3)])
    gamma_S1[is_nans] = emi_nans
    gamma_S3[is_nans] = emi_nans
    gamma_S1 = gamma_S1/np.sum(gamma_S1)
    gamma_S3 = gamma_S3/np.sum(gamma_S3)
    B[0,:] = gamma_S1
    B[1,:] = gamma_S3
    return B

def comp_alpha_beta_log(y, A, B, pi):
    '''
    returns log(Alpha) and log(Beta): N-by-T matrix, 
    each row gives log(alpha_i(t)) or log(beta_i(t))
    '''
    N = A.shape[0]
    T = len(y)
    Alpha_L = np.zeros((N, T))
    for i in range(N):
        Alpha_L[i,0] = np.log(pi[i]*B[i,0])
    for t in range(1,T):
        temp_vec = Alpha_L[:,t-1] - Alpha_L[0,t-1]
        for i in range(N):
            Alpha_L[i,t] = np.log(B[i,y[t]]) + np.log(np.sum(np.exp(temp_vec) * A[:,
                         i])) + Alpha_L[0,t-1]
    Beta_L = np.zeros((N, T))
    Beta_L[:,T-1] = 0
    for t_rev in range(1, T):
        t = T - t_rev - 1
        temp_vec = Beta_L[:,t+1] - Beta_L[0,t+1]
        for i in range(N):
            Beta_L[i,t] = np.log(np.sum(np.exp(temp_vec)*A[i,:] * B[:, 
                        y[t+1]])) + Beta_L[0,t+1]
    return Alpha_L, Beta_L

def cut_ends_pmf(pmf, cut_ratio = 0.001):
    remaining = cut_ratio
    i = 0
    while remaining > 0:
        if i > len(pmf)-1:
            break
        if pmf[i] > 0:
            if pmf[i] < remaining:
                remaining -= pmf[i]
                pmf[i] = 0
            if pmf[i] > remaining:
                pmf[i] = pmf[i] - remaining
                remaining = 0
        i += 1
    remaining = cut_ratio
    i = len(pmf) - 1
    while remaining > 0:
        if i < 0:
            break
        if pmf[i] > 0:
            if pmf[i] < remaining:
                remaining -= pmf[i]
                pmf[i] = 0
            if pmf[i] > remaining:
                pmf[i] = pmf[i] - remaining
                remaining = 0
        i -= 1
    return pmf

def gamma_pmf(pmf):
    '''
    function returns a gamma-smoothed pmf given a pmf
    '''
    pmf = cut_ends_pmf(pmf, cut_ratio = 0.001)
    pmf_x = np.arange(len(pmf))
    gamma_mean = np.sum(pmf_x*pmf)
    gamma_var = np.sum((pmf_x*pmf_x)*pmf) - gamma_mean**2
    gamma_skew = np.sum(pmf*(((pmf_x - gamma_mean)/np.sqrt(gamma_var))**3))
    x0 = gamma_mean - 2*np.sqrt(gamma_var)/gamma_skew
    theta = np.sqrt(gamma_var) * gamma_skew / 2
    k = 4/(gamma_skew)**2
    dist_gamma_pmf = ((pmf_x-x0)**(k-1))*np.exp(-(pmf_x-x0)/theta)
    dist_gamma_pmf[dist_gamma_pmf == np.inf] = np.nan
    norm_const = np.nansum(dist_gamma_pmf)
    dist_gamma_pmf = dist_gamma_pmf/norm_const
    return dist_gamma_pmf

def update_AB_log(y, A, B, Alpha_L, Beta_L, gamma_B = 'yes'):
    N = Alpha_L.shape[0]
    T = Alpha_L.shape[1]
    temp_vec = Alpha_L[:,0] + Beta_L[:,0]
    temp_vec = np.exp(temp_vec - temp_vec[0])
    log_P_y_given_model = np.log(np.sum(temp_vec)) + Alpha_L[0,0] + Beta_L[0,0]
    Gamma = np.exp(Alpha_L + Beta_L - log_P_y_given_model)
    Xi = np.zeros((N, N, T-1))
    for i in range(N):
        for j in range(N):
            Xi[i,j,:] = np.exp(Alpha_L[i,0:T-1] + np.log(A[i,j]) + Beta_L[j, 
                           1:T] + np.log(B[j, y[1:T]]) - log_P_y_given_model)
    pi_new = Gamma[:,0]
    A_new = np.zeros((N, N))
    for i in range(N):
        Gamma_sum = np.sum(Gamma[i, 0:T-1])
        for j in range(N):
            A_new[i,j] = np.sum(Xi, axis=2)[i,j] / Gamma_sum
    K = B.shape[1]
    B_new = np.zeros((N, K))
    for j in range(K):
        I_state_k = (y == j)
        for i in range(N):
            B_new[i,j] = np.sum(Gamma[i,I_state_k]) / np.sum(Gamma[i, :])
    if gamma_B == 'yes':
        gamma_S1 = gamma_pmf(B_new[0,:])
        gamma_S3 = gamma_pmf(B_new[1,:])
        is_nans = np.zeros((len(B_new[0,:]),), dtype = bool)
        is_nans[np.isnan(gamma_S1)] = 1
        is_nans[np.isnan(gamma_S3)] = 1
        gamma_S1[is_nans] = np.nan
        gamma_S3[is_nans] = np.nan
        gamma_S1 = gamma_S1/np.nansum(gamma_S1)
        gamma_S3 = gamma_S3/np.nansum(gamma_S3)
        emi_nans = np.max([np.nanmin(gamma_S1), np.nanmin(gamma_S3)])
        gamma_S1[is_nans] = emi_nans
        gamma_S3[is_nans] = emi_nans
        gamma_S1 = gamma_S1/np.sum(gamma_S1)
        gamma_S3 = gamma_S3/np.sum(gamma_S3)
        B_new[0,:] = gamma_S1
        B_new[1,:] = gamma_S3
    return A_new, B_new, pi_new

def BW_HMM_log(y, A, B, pi, max_it = 100, min_it = 50, log_margin = -np.inf, 
               gamma_B = 'no', mutate = 0):
    '''
    N: number of possible states
    K: number of possible observations
    '''
    max_it = min_it + 1
    model_scores = np.zeros((max_it))
    for it in range(max_it):
        Alpha_L, Beta_L = comp_alpha_beta_log(y, A, B, pi)
        temp_vec = Alpha_L[:,0] + Beta_L[:,0]
        temp_vec = np.exp(temp_vec - temp_vec[0])
        log_P_y_given_model = np.log(np.sum(temp_vec)) + Alpha_L[0,0] + Beta_L[0,0]
        model_scores[it] = log_P_y_given_model
        if it > min_it:
            if model_scores[it] - model_scores[it-1] < log_margin:
                print('break at iteration %s'%it)
                model_scores = model_scores[0:it]
                break
        if mutate > 0:
            if it < min_it*0.8:
                N = B.shape[0]
                K = B.shape[1]
                mutate_coef = np.random.rand(N,K)*mutate*2 - mutate + 1
                for j in range(K):
                    for i in range(N):
                        B[i,j] = B[i,j] * mutate_coef[i,j]
        A, B, pi = update_AB_log(y, A, B, Alpha_L, Beta_L, gamma_B = gamma_B)
    return A, B, pi, model_scores

def BW_HMM_log_fixit(y, A, B, pi, num_it = 50, gamma_B = 'yes', mutate = 0):
    '''
    N: number of possible states
    K: number of possible observations
    '''
    model_scores = np.zeros((num_it))
    for it in range(num_it):
        Alpha_L, Beta_L = comp_alpha_beta_log(y, A, B, pi)
        temp_vec = Alpha_L[:,0] + Beta_L[:,0]
        temp_vec = np.exp(temp_vec - temp_vec[0])
        log_P_y_given_model = np.log(np.sum(temp_vec)) + Alpha_L[0,0] + Beta_L[0,0]
        model_scores[it] = log_P_y_given_model
        if mutate > 0:
            if it < num_it*0.8:
                N = B.shape[0]
                K = B.shape[1]
                mutate_coef = np.random.rand(N,K)*mutate*2 - mutate + 1
                for j in range(K):
                    for i in range(N):
                        B[i,j] = B[i,j] * mutate_coef[i,j]
        A, B, pi = update_AB_log(y, A, B, Alpha_L, Beta_L, gamma_B = gamma_B)
    return A, B, pi, model_scores

def comp_gamma(y, A, B, pi):
    Alpha_L, Beta_L = comp_alpha_beta_log(y, A, B, pi)
    temp_vec = Alpha_L[:,0] + Beta_L[:,0]
    temp_vec = np.exp(temp_vec - temp_vec[0])
    log_P_y_given_model = np.log(np.sum(temp_vec)) + Alpha_L[0,0] + Beta_L[0,0]
    Gamma = np.exp(Alpha_L + Beta_L - log_P_y_given_model)
    return Gamma


# In[] for BM_CHMM_utility
def BM_CHMM(dataname, col_name, indi_name, mag, rad, ttf_threshold, seglen, 
            max_overlap, n_bins, gamma_B, min_it, mutate, rand_init = 'no',
            indic = 'mean'):
    ttf_B12, ttf_B23 = ttf_threshold, ttf_threshold
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    eqks = mB.eqk()
    eqks.select(dataname, mag, radius = rad)
    t_eqs = eqks.selected[:,0]
    col = data[col_name]
    ttf = mB.time2failure(col['t_ind'], t_eqs)
    indic_TS = col[indi_name]
    indic_TS = indic_TS[0,:]
    # ----- separate segs for S1/2/3 ----- 
    # create segments
    seg_pts = mB.get_seg_pts(0, len(ttf)-1, seglen_max = seglen, seglen_min = seglen, 
                             max_overlap=max_overlap)
    seg_pts = np.int64(seg_pts)
    t_segs = col['t_ind'][seg_pts[:,1]]
    seg_states = np.zeros((len(seg_pts), ))
    for i in range(len(seg_pts)):
        if ttf[seg_pts[i,1]] > ttf_B12:
            seg_states[i] = 1
        elif ttf[seg_pts[i,1]] > ttf_B23:
            seg_states[i] = 2
        elif ttf[seg_pts[i,1]] > 0:
            seg_states[i] = 3
    seg_pts_S1 = seg_pts[seg_states == 1]
    seg_pts_S2 = seg_pts[seg_states == 2]
    seg_pts_S3 = seg_pts[seg_states == 3]
    S1_means = np.zeros((len(seg_pts_S1), ))
    S1_vars = np.zeros((len(seg_pts_S1), ))
    S2_means = np.zeros((len(seg_pts_S2), ))
    S2_vars = np.zeros((len(seg_pts_S2), ))
    S3_means = np.zeros((len(seg_pts_S3), ))
    S3_vars = np.zeros((len(seg_pts_S3), ))
    for i in range(len(seg_pts_S1)):
        S1_means[i] = np.nanmean(indic_TS[seg_pts_S1[i,0]:seg_pts_S1[i,1]])
        S1_vars[i] = np.nanvar(indic_TS[seg_pts_S1[i,0]:seg_pts_S1[i,1]])
    for i in range(len(seg_pts_S2)):
        S2_means[i] = np.nanmean(indic_TS[seg_pts_S2[i,0]:seg_pts_S2[i,1]])
        S2_vars[i] = np.nanvar(indic_TS[seg_pts_S2[i,0]:seg_pts_S2[i,1]])
    for i in range(len(seg_pts_S3)):
        S3_means[i] = np.nanmean(indic_TS[seg_pts_S3[i,0]:seg_pts_S3[i,1]])
        S3_vars[i] = np.nanvar(indic_TS[seg_pts_S3[i,0]:seg_pts_S3[i,1]])
    means_ts = np.zeros((len(seg_pts), ))
    vars_ts = np.zeros((len(seg_pts), ))
    for i in range(len(seg_pts)):
        means_ts[i] = np.nanmean(indic_TS[seg_pts[i,0]:seg_pts[i,1]])
        vars_ts[i] = np.nanvar(indic_TS[seg_pts[i,0]:seg_pts[i,1]])
    #
    if indic == 'mean':
        indic_ts = means_ts
        indic_S1 = S1_means
        indic_S3 = S3_means
    elif indic == 'var':
        indic_ts = vars_ts
        indic_S1 = S1_vars
        indic_S3 = S3_vars
    # indic_ts being the indicator value time series
    LB_domain = np.nanmin(indic_ts)
    UB_domain = np.nanmax(indic_ts)
    bin_splits = np.linspace(LB_domain, UB_domain, n_bins+1)
    bin_centers = (bin_splits[0:-1] + bin_splits[1:])/2
    if rand_init == 'no':
        pi = np.array([np.sum(seg_states == 1)/len(seg_states), 
                       np.sum(seg_states == 3)/len(seg_states)])
        transitions = np.zeros((len(seg_states)-1, 2))
        transitions[:,0] = seg_states[0:-1]
        transitions[:,1] = seg_states[1:]
        A = np.zeros((2,2))
        for i in range(len(transitions)):
            if transitions[i,0] == 1:
                if transitions[i,1] == 1:
                    A[0,0] += 1
                elif transitions[i,1] == 3:
                    A[0,1] += 1
            elif transitions[i,0] == 3:
                if transitions[i,1] == 1:
                    A[1,0] += 1
                elif transitions[i,1] == 3:
                    A[1,1] += 1
        A[0,:] = A[0,:]/np.sum(A[0,:])
        A[1,:] = A[1,:]/np.sum(A[1,:])
        # init B    
        B = histo_construct_B(n_bins, indic_S1, indic_S3, indic_ts)
    elif rand_init == 'yes':
        B = np.zeros((2, n_bins))
        B[0,:] = np.random.rand(n_bins)
        B[0,:] = B[0,:]/np.sum(B[0,:])
        B[1,:] = np.random.rand(n_bins)
        B[1,:] = B[1,:]/np.sum(B[1,:])
        A = np.zeros((2,2))
        A[0,:] = np.random.rand(2)
        A[0,:] = A[0,:]/np.sum(A[0,:])
        A[1,:] = np.random.rand(2)
        A[1,:] = A[1,:]/np.sum(A[1,:])
        pi = np.random.rand(2)
        pi = pi/np.sum(pi)
    # convert indic_ts to discretized observables
    id_observable = np.arange(0, n_bins, dtype = int)
    y = np.zeros(len(indic_ts), dtype = int)
    for i in range(len(indic_ts)):
        indic_diff_to_binCtr = np.abs(bin_centers - indic_ts[i])
        id_obs = id_observable[indic_diff_to_binCtr == np.min(indic_diff_to_binCtr)]
        y[i] = id_obs
    # full iters
    A_new, B_new, pi_new, model_scores = BW_HMM_log(y, A, B, pi, gamma_B = gamma_B, 
                                                    min_it=min_it, mutate=mutate)
    Gamma = comp_gamma(y, A_new, B_new, pi_new)
    # make sure row0 is state with lower mean
    B_inds = np.arange(0, n_bins)
    B_mean_row0 = np.nanmean(B_new[0,:]*B_inds)
    B_mean_row1 = np.nanmean(B_new[1,:]*B_inds)
    if B_mean_row0 > B_mean_row1:
        temp_Gamma_row = Gamma[0,:].copy()
        Gamma[0,:] = Gamma[1,:]
        Gamma[1,:] = temp_Gamma_row
        temp_B_row = B_new[0,:].copy()
        B_new[0,:] = B_new[1,:]
        B_new[1,:] = temp_B_row
        A_temp = A_new.copy()
        A_new[0,0] = A_temp[1,1]
        A_new[1,0] = A_temp[0,1]
        A_new[0,1] = A_temp[1,0]
        A_new[1,1] = A_temp[0,0]
    Gamma_S1 = np.sort(Gamma[0,:].copy())
    Gamma_5pct = Gamma_S1[np.int(len(Gamma_S1)*0.05)]
    Gamma_95pct = Gamma_S1[np.int(len(Gamma_S1)*0.95)]
    if (Gamma_5pct < 0.05) & (Gamma_95pct > 0.95):
        model_trained = 'yes'
    else:
        model_trained = 'no'
    info = {'t_segs': t_segs,
            'y': y,
            'Gamma': Gamma,
            'A': A_new,
            'B': B_new,
            'pi': pi_new,
            'model_scores': model_scores,
            'model_trained': model_trained,
            'bin_centers': bin_centers}
    return info

def BM_CHMM_day(dataname, col_name, indi_name, n_bins, gamma_B, min_it, mutate,
                cut_extreme_rate = 0.005):
    # ttf_B12, ttf_B23 = ttf_threshold, ttf_threshold
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    col = data[col_name]
    indic_TS = col[indi_name]
    indic_TS = indic_TS[0,:]
    t_segs = col['t_ind']
    indic_ts = indic_TS
    # convert indic_ts such that top/low extreme values are cut flat
    ts_nonan_sorted = np.sort(indic_ts[~np.isnan(indic_ts)])
    top_bound = ts_nonan_sorted[int(len(indic_ts)*(1-cut_extreme_rate))]
    lower_bound = ts_nonan_sorted[int(len(indic_ts)*cut_extreme_rate)]
    indic_ts[indic_ts>top_bound] = top_bound
    indic_ts[indic_ts<lower_bound] = lower_bound
    # random init HMM
    B = np.zeros((2, n_bins))
    B[0,:] = np.random.rand(n_bins)
    B[0,:] = B[0,:]/np.sum(B[0,:])
    B[1,:] = np.random.rand(n_bins)
    B[1,:] = B[1,:]/np.sum(B[1,:])
    A = np.zeros((2,2))
    A[0,:] = np.random.rand(2)
    A[0,:] = A[0,:]/np.sum(A[0,:])
    A[1,:] = np.random.rand(2)
    A[1,:] = A[1,:]/np.sum(A[1,:])
    pi = np.random.rand(2)
    pi = pi/np.sum(pi)
    # prep bins
    LB_domain = np.nanmin(indic_ts)
    UB_domain = np.nanmax(indic_ts)
    bin_splits = np.linspace(LB_domain, UB_domain, n_bins+1)
    bin_centers = (bin_splits[0:-1] + bin_splits[1:])/2
    # convert indic_ts to discretized observables
    id_observable = np.arange(0, n_bins, dtype = int)
    y = np.zeros(len(indic_ts), dtype = int)
    for i in range(len(indic_ts)):
        indic_diff_to_binCtr = np.abs(bin_centers - indic_ts[i])
        id_obs = id_observable[indic_diff_to_binCtr == np.min(indic_diff_to_binCtr)]
        y[i] = id_obs
    # full iters
    A_new, B_new, pi_new, model_scores = BW_HMM_log(y, A, B, pi, gamma_B = gamma_B, 
                                                    min_it=min_it, mutate=mutate)
    Gamma = comp_gamma(y, A_new, B_new, pi_new)
    # make sure row0 is state with lower mean
    B_inds = np.arange(0, n_bins)
    B_mean_row0 = np.nanmean(B_new[0,:]*B_inds)
    B_mean_row1 = np.nanmean(B_new[1,:]*B_inds)
    if B_mean_row0 > B_mean_row1:
        temp_Gamma_row = Gamma[0,:].copy()
        Gamma[0,:] = Gamma[1,:]
        Gamma[1,:] = temp_Gamma_row
        temp_B_row = B_new[0,:].copy()
        B_new[0,:] = B_new[1,:]
        B_new[1,:] = temp_B_row
        A_temp = A_new.copy()
        A_new[0,0] = A_temp[1,1]
        A_new[1,0] = A_temp[0,1]
        A_new[0,1] = A_temp[1,0]
        A_new[1,1] = A_temp[0,0]
    Gamma_S1 = np.sort(Gamma[0,:].copy())
    Gamma_5pct = Gamma_S1[np.int(len(Gamma_S1)*0.05)]
    Gamma_95pct = Gamma_S1[np.int(len(Gamma_S1)*0.95)]
    if (Gamma_5pct < 0.05) & (Gamma_95pct > 0.95):
        model_trained = 'yes'
    else:
        model_trained = 'no'
    info = {'t_segs': t_segs,
            'y': y,
            'Gamma': Gamma,
            'A': A_new,
            'B': B_new,
            'pi': pi_new,
            'model_scores': model_scores,
            'model_trained': model_trained,
            'bin_centers': bin_centers}
    return info

def sync_TsegsTeqs(t_segs, t_eqs, Tsegs_maxdiff = 1):
    '''
    t_segs can be the time stamp sequence of any indicator time series
    it is called t_segs because its first use is on the t_segs of BM_CHMMMI info, 
    for being the time stamps for Gamma and segindic_TSs
    t_eqs is the time stamp sequence of selected EQs
    this function does:
        1) remove t_segs lying out side t_eqs range
        2) remove t_eqs lying in-between abnormally long t_segs ticks
    function returns bool sequence instead of directly making the sync
    '''
    t_LB = t_eqs[0]-0.01
    t_UB = t_eqs[-1]+1
    Tsegs_keep = np.ones((len(t_segs),), dtype = bool)
    Tsegs_keep[t_segs < t_LB] = 0
    Tsegs_keep[t_segs > t_UB] = 0
    t_segs_new = t_segs[Tsegs_keep]
    if len(t_segs_new) == 0: # if no t_seg qualified due to very few EQs, deny this part
        Tsegs_keep = np.ones((len(t_segs),), dtype = bool)
        t_segs_new = t_segs[Tsegs_keep]
    Teqs_keep = np.ones((len(t_eqs),), dtype = bool)
    Teqs_keep[t_eqs < t_segs_new[0]] = 0
    Teqs_keep[t_eqs > t_segs_new[-1]] = 0
    '''
    corrected and simplified on 2020-7-7
    '''
#    i_seg = 0
#    for i in range(len(t_eqs)):
#        t_eq = t_eqs[i]
#        if t_eq < t_segs_new[i_seg]:
#            Teqs_keep[i] = 0
#            continue
#        while i_seg < len(t_segs_new)-1:
#            if t_segs_new[i_seg] < t_eq:
#                if t_segs_new[i_seg+1] > t_eq: # i.e. t_segs[i_seg+1] > t_eq > t_segs[i_seg]
#                    if t_segs_new[i_seg+1] - t_segs_new[i_seg] > Tsegs_maxdiff:
#                        Teqs_keep[i] = 0
#                    break
#            i_seg += 1
#    Teqs_keep[t_eqs > np.max(t_segs)] = 0
    return Tsegs_keep, Teqs_keep 

def compute_EQrates_old(state_threshold, t_eqs, info, return_all = 'no', sync_gamma = 'no'):
    '''
    state_threshold is the 'tolerance' of state classification, 0.5 being most loose
    assuming S1 occupies 'len_S1' duration, while 'eqs_S1' number of EQ happened, 
    then 'eq_rate_S1' is the EQ rate for S1; same for S2
    function returns the EQ rate vector for both S1 and S2, 
    can also optionally return all len_Sn and eqs_Sn
    '''
    Gamma = info['Gamma'].copy()
    t_segs = info['t_segs'].copy()
    Gamma[0,Gamma[0,:] < state_threshold] = 0
    Gamma[0,Gamma[0,:] > 1 - state_threshold] = 1
    Gamma[1,Gamma[1,:] < state_threshold] = 0
    Gamma[1,Gamma[1,:] > 1 - state_threshold] = 1
    len_S1 = np.sum(Gamma[0,:] == 1)
    len_S2 = np.sum(Gamma[1,:] == 1)
    if len(t_eqs) > 0:
        Tsegs_keep, Teqs_keep = sync_TsegsTeqs(t_segs, t_eqs) # added on 24 April
        if sync_gamma != 'no': # added on 30 April, to cope well in case t_eqs is selected few
            Gamma = Gamma[:,Tsegs_keep]
        t_eqs = t_eqs.copy()[Teqs_keep]
    else:
        if return_all == 'no':
            return np.nan
        else:
            return np.nan, 0, len_S1, 0, len_S2
    num_eqs = np.zeros((2,))
    if len(t_eqs) >= 1:
        id_eq = 0
        t_eq = t_eqs[id_eq]
        i = 1
        while (i < len(Gamma[0,:])) & (id_eq < len(t_eqs)-1):
            if (t_eq < t_segs[i]) & (t_eq >= t_segs[i-1]): # i.e. eq happened in between
                if Gamma[0,i-1] == 0:
                    num_eqs[0] += 1
                elif Gamma[0,i-1] == 1:
                    num_eqs[1] += 1
                id_eq += 1
                t_eq = t_eqs[id_eq]
            else:
                i += 1
    eqs_S1 = num_eqs[0]
    eqs_S2 = num_eqs[1]
    eq_rate_S1 = eqs_S1/len_S1
    eq_rate_S2 = eqs_S2/len_S2
    eq_rates = np.array([eq_rate_S1, eq_rate_S2])
    if return_all == 'no':
        return eq_rates
    else:
        return eq_rates, eqs_S1, len_S1, eqs_S2, len_S2

def compute_EQrates(state_threshold, t_eqs, info, return_all = 'no', sync_gamma = 'no',
                    min_EQ = 5, shift = 0):
    '''
    state_threshold is the 'tolerance' of state classification, 0.5 being most loose
    assuming S1 occupies 'len_S1' duration, while 'eqs_S1' number of EQ happened, 
    then 'eq_rate_S1' is the EQ rate for S1; same for S2
    function returns the EQ rate vector for both S1 and S2, 
    can also optionally return all len_Sn and eqs_Sn
    min_EQ is the minimum # of EQ to yield eq_rates
    '''
    Gamma = info['Gamma'].copy()
    t_segs = info['t_segs'].copy()
    Gamma[0,Gamma[0,:] < state_threshold] = 0
    Gamma[0,Gamma[0,:] > 1 - state_threshold] = 1
    Gamma[1,Gamma[1,:] < state_threshold] = 0
    Gamma[1,Gamma[1,:] > 1 - state_threshold] = 1
    len_S1 = np.sum(Gamma[0,:] == 1)
    len_S2 = np.sum(Gamma[1,:] == 1)
    if len(t_eqs) > 0:
        if sync_gamma != 'no': # added on 30 April, to cope well in case t_eqs is selected few
            Tsegs_keep, Teqs_keep = sync_TsegsTeqs(t_segs, t_eqs) # added on 24 April
            Gamma = Gamma[:,Tsegs_keep]
        t_eqs = t_eqs[t_eqs < t_segs[-1]] 
        t_eqs = t_eqs[t_eqs > t_segs[0]]
    else: 
        if return_all == 'no':
            return np.array([np.nan, np.nan])
        else:
            return np.array([np.nan, np.nan]), 0, len_S1, 0, len_S2
    num_eqs = np.zeros((2,))
    eqs_counted = len(t_eqs)
    if len(t_eqs) >= 1:
        t_eqs = t_eqs + shift
        for i in range(len(t_eqs)):
            t_eqi = t_eqs[i]
            ind = np.argmin(np.abs(t_segs-t_eqi))
            # -------- new!! --------
            t_diff = t_segs[ind] - t_eqi
            if t_diff > 1: # nearest ind is >1 day's far, this case doesn't count
                eqs_counted -= 1
            else:
            # -------- new!! --------
                if Gamma[0,ind] == 1:
                    num_eqs[0] += 1
                elif Gamma[1,ind] == 1:
                    num_eqs[1] += 1
    eqs_S1 = num_eqs[0]
    eqs_S2 = num_eqs[1]
    if eqs_counted > min_EQ:
        eq_rate_S1 = eqs_S1/len_S1
        eq_rate_S2 = eqs_S2/len_S2
        eq_rates = np.array([eq_rate_S1, eq_rate_S2])
    else:
        eq_rates = np.array([np.nan, np.nan])
    if return_all == 'no':
        return eq_rates
    else:
        return eq_rates, eqs_S1, len_S1, eqs_S2, len_S2

def plot_EQrates(info, state_threshold = 0.5, mags = [], num_div = 10,
                 rads = np.concatenate((np.arange(0., 2.61, 0.2), [3.0])),
                 numEQ_LB = 0, return_num = 0, plot = 'yes', shift = 0, 
                 sub_inds = [2,3,4], plot_prefix = ['', '', '']):
    '''
    funtion computes and plots (optional) the EQ_rates, eqs_S1, len_S2
    return_num is the number of values the funciton returns, in order of :
        EQratios, EQ_S1s, EQ_S2s, len_S1, len_S2
    plot: 'yes' for plotting EQratios, 'sub' for making 3 subplots of:
        EQratios, EQ_S1s, EQ_S2s
    sub_inds: the index for making subplots. subplot is defined by
    plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2]+0/1/2)
    '''
    dataname = info['dataname']
    if len(mags) == 0:
        eqks = mB.eqk()
        eqks.get_mag_divs(num_div)
        mags = eqks.mag_divs
    EQratios = np.zeros((len(mags)-1, len(rads)-1))
    EQ_S1s = np.zeros((len(mags)-1, len(rads)-1))
    EQ_S2s = np.zeros((len(mags)-1, len(rads)-1))
    for i_mag in range(len(mags)-1):
        for i_rad in range(len(rads)-1):
            eqks = mB.eqk()
            eqks.select_ULB(dataname, mag_UB = mags[i_mag+1], mag_LB = mags[i_mag], 
                            rad_UB = rads[i_rad+1], rad_LB = rads[i_rad])
            t_eqs = eqks.selected[:,0]
            if len(t_eqs) > numEQ_LB:
                eq_rates, eqs_S1, len_S1, eqs_S2, len_S2 = compute_EQrates(state_threshold, 
                                                           t_eqs, info, return_all = 'yes', 
                                                           shift = shift)
                EQratios[i_mag, i_rad] = eq_rates[0]/(eq_rates[1]+eq_rates[0]) # -changed!!-
                EQ_S1s[i_mag, i_rad] = eqs_S1
                EQ_S2s[i_mag, i_rad] = eqs_S2
            else:
                EQratios[i_mag, i_rad] = np.nan
                EQ_S1s[i_mag, i_rad] = np.nan
                EQ_S2s[i_mag, i_rad] = np.nan
    rads = np.round(rads*100)/100
    mags = np.round(mags*100)/100
    if plot != 'no':
        HS_mags, HS_rads = info['HS_mags'], info['HS_rads']
        eqks = mB.eqk()
        eqks.select_ULB(dataname, mag_UB = HS_mags[-1], mag_LB = HS_mags[0], 
                        rad_UB = HS_rads[-1], rad_LB = HS_rads[0], verbose = 'no')
        t_eqs = eqks.selected[:,0]
        eq_ratesG = compute_EQrates(state_threshold, t_eqs, info, shift = shift)
        EQratio_global = eq_ratesG[0]/(eq_ratesG[1]+eq_ratesG[0]) # ----changed!!----
        EQtext = 'global = %.4f at mag in [%s, %s], rad in [%s, %s])'%(EQratio_global,
                 HS_mags[0], HS_mags[1], HS_rads[0], HS_rads[1])
    if plot == 'yes':
        ax = sns.heatmap(EQratios, annot=True, cbar=False, 
                         mask=np.isnan(EQratios), center = 0.5, cmap="PiYG")
        ax.set_xticks(np.arange(0, len(rads),1))
        ax.set_xticklabels(rads)
        ax.set_yticks(np.arange(0, len(mags),1))
        ax.set_yticklabels(mags, rotation = 0)
        ax.set_title(plot_prefix[0]+' EQrate_S1 / EQrate_S1+S2, ' + EQtext)
        ax.set_xlabel('Radius Bounds')
        ax.set_ylabel('Magnitude Bounds')
    if plot == 'sub':
        # 1
        plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2])
        ax = sns.heatmap(EQ_S1s, annot=True, fmt='g', cbar=False)
        ax.set_xticks(np.arange(0, len(rads),1))
        ax.set_xticklabels(rads)
        ax.set_yticks(np.arange(0, len(mags),1))
        ax.set_yticklabels(mags, rotation = 0)
        ax.set_title(plot_prefix[0]+' EQ_S1s, with len_S1 = %s'%len_S1)
        ax.set_xlabel('Radius Bounds')
        ax.set_ylabel('Magnitude Bounds')
        # 2
        plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2]+1)
        ax = sns.heatmap(EQ_S2s, annot=True, fmt='g', cbar=False)
        ax.set_xticks(np.arange(0, len(rads),1))
        ax.set_xticklabels(rads)
        ax.set_yticks(np.arange(0, len(mags),1))
        ax.set_yticklabels(mags, rotation = 0)
        ax.set_title(plot_prefix[1]+' EQ_S2s, with len_S2 = %s'%len_S2)
        ax.set_xlabel('Radius Bounds')
        ax.set_ylabel('Magnitude Bounds')
        # 3
        plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2]+2)
        ax = sns.heatmap(EQratios, annot=True, cbar=False, 
                         mask=np.isnan(EQratios), center = 0.5, cmap="bwr")
        ax.set_xticks(np.arange(0, len(rads),1))
        ax.set_xticklabels(rads)
        ax.set_yticks(np.arange(0, len(mags),1))
        ax.set_yticklabels(mags, rotation = 0)
        ax.set_title(plot_prefix[2]+' EQrate_S1 / EQrate_S1+S2, ' + EQtext)
        ax.set_xlabel('Radius Bound')
        ax.set_ylabel('Magnitude Bound')
    if return_num == 0:
        return 
    elif return_num == 1:
        return EQratios
    elif return_num == 3:
        return EQratios, EQ_S1s, EQ_S2s
    else:
        return EQratios, EQ_S1s, EQ_S2s, len_S1, len_S2
    
# In[]
def best_n_heatmap(file_name, n_configs, nlabels_pick = 0, note = '', save = 'yes',
                   HS_rads = [], HS_mags = [], mags = [], shift = 0,
                   rads = np.concatenate((np.arange(0., 2.61, 0.2), [3.0]))):
    # file_name e.g. 'BM_CHMMMI_DABA_ti2_FCW500S2_CVSK200_CVSK_R100L90_0416G.pickle'
    with open(file_name, 'rb') as f:
        dic_config = pickle.load(f)
    best_n_configs = {}
    for config in dic_config:
        info_dic = dic_config[config]
        if nlabels_pick > 0:
            for info_name in info_dic:
                info = info_dic[info_name]
                break
            if info['n_labels'] != nlabels_pick:
                continue
        model_scores_all = list()
        for info_name in info_dic:
            model_scores_all.append(info_dic[info_name]['model_scores'][-1])
        sorted_scores = np.sort(np.array(model_scores_all))[::-1]
        score_threshold = sorted_scores[n_configs-1]
        for info_name in info_dic:
            if info_dic[info_name]['model_scores'][-1] >= score_threshold:
                best_n_configs[info_name] = info_dic[info_name]
    info_dic = best_n_configs
    for key in info_dic:
        info = info_dic[key]
        if len(HS_rads) == 0:
            HS_rads = info['HS_rads']
            HS_mags = info['HS_mags']
        else:
            info = standardize_HS(info, rads = HS_rads, mags = HS_mags, shift = shift)
        fig = plt.figure(figsize=(28,14))
        if np.isnan(info['model_scores'][-1]):
            end_score = 0
        else:
            end_score = np.int(info['model_scores'][-1])
        figure_name = file_name[0:-7] + note + '_' + key + '_score%s'%(end_score)
        plt.suptitle(figure_name)
        plt.subplot(2,3,1)
        plt.title('(a) Log model scores vs. iteration')
        plt.plot(info['model_scores'], 'o')
        plt.subplot(2,3,2)
        plt.title('(b) Indicator distribution (Emission) for two states')
        plt.plot(info['B'][0,:], label = 'S1')
        plt.plot(info['B'][1,:], label = 'S2')
        plt.legend()
        plt.xlabel('ID of B')
        plt.subplot(2,3,3)
        plt.plot(info['t_segs'], info['Gamma'][0,:])
        plt.title('(c) Posterior P(S1)')
        plt.xlabel("Time (4 years' long)")
        if ~np.isnan(info['Gamma'][0,0]):
            plot_prefix = ['(d)', '(e)', '(f)']
            plot_EQrates(info, return_num = 0, plot = 'sub', sub_inds = [2,3,4], 
                         plot_prefix = plot_prefix, mags = mags, rads = rads, shift = shift)
        else:
            print('Nan Gamma found, skipped heatmap')
        if save == 'yes':
            plt.savefig(figure_name +'.png', dpi = 500)
            plt.close(fig)
    return

# In[]
def sort_ccs(cluster_centers, labels, sort_standards, input_names):
    '''
    function re-organizes the results of kmeans clustering in BM_CHMMMI
    in order of ascending 'sort_standard' values along cluster labeling
    sort_standard can be: 'col1_Ks_MeanOfSegs'
    input_names is a list of [..., 'col1_Ks_MeanOfSegs', ...] as in BM_CHMMMI
    cluster_centers, labels = kmeans.cluster_centers_, kmeans.labels_
    '''
    for sort_standard in sort_standards:
        i_name = 0
        ind_to_sort = list()
        for input_name in input_names:
            if input_name == sort_standard:
                ind_to_sort.append(i_name)
                break
            else:
                i_name += 1
    seq_to_sort = np.zeros((cluster_centers.shape[0],))
    for i in range(len(ind_to_sort)):
        temp_vec = cluster_centers[:,ind_to_sort[i]].copy()
        temp_vec = temp_vec / np.nanstd(temp_vec)
        temp_vec = temp_vec - np.nanmean(temp_vec)
        seq_to_sort += temp_vec**2
    seq_to_sort = np.sqrt(seq_to_sort)
    a = np.argsort(seq_to_sort)
    sorting_scores = seq_to_sort[a]
    label_inds = np.arange(0, cluster_centers.shape[0])
    ccs_sorted = cluster_centers[a,:]
    label_function = label_inds[a]
    labels_new = labels.copy()
    for i in range(len(labels)):
        labels_new[i] = label_function[labels[i]]
    return ccs_sorted, labels_new, sorting_scores

# In[] 
#def standardize_HS(info, force = 'no', mode = 'heatmap', mags = np.array([2.4, 3.8]), 
#                   rads = np.array([0.2, 0.6])):
#    # for dictionary 'info' of format outputed by BM_CHMMMI or BM_CHMM with 2-by-2 A
#    switch = 'no'
#    if mode == 'Gamma':
#        # this mode makes sure S1 is more common than S2 in earlier times 
#        half_len = np.int(info['Gamma'].shape[1]/4)
#        if force == 'no':
#            if np.mean(info['Gamma'][0,0:half_len]) > 0.5:
#                switch = 'yes'
#        else:
#            switch = 'yes'
#    elif mode == 'heatmap': # make sure S1 has more local small EQs
#        EQratios = plot_EQrates(info, state_threshold = 0.5, plot = 'no', 
#                                return_num = 1, mags = mags, rads = rads)
#        if np.nanmean(EQratios) > 0.5:
#            switch = 'yes'
#    if switch == 'yes':
#        A_copy = info['A'].copy()
#        info['A'][0,0] = A_copy[1,1]
#        info['A'][1,0] = A_copy[0,1]
#        info['A'][0,1] = A_copy[1,0]
#        info['A'][1,1] = A_copy[0,0]
#        B_copy = info['B'].copy()
#        info['B'][0,:] = B_copy[1,:]
#        info['B'][1,:] = B_copy[0,:]
#        Gamma_copy = info['Gamma'].copy()
#        info['Gamma'][0,:] = Gamma_copy[1,:]
#        info['Gamma'][1,:] = Gamma_copy[0,:]
#        pi_copy = info['pi'].copy()
#        info['pi'][0] = pi_copy[1]
#        info['pi'][1] = pi_copy[0]
#    return info

def standardize_HS(info, force = 'no', mags = np.array([2.4, 3.8]), 
                   rads = np.array([0.2, 0.6]), shift = 0):
    # for dictionary 'info' of format outputed by BM_CHMMMI or BM_CHMM with 2-by-2 A
    switch = 'no'
    if force == 'yes':
        switch = 'yes'
    else:
        eqks = mB.eqk()
        t_eqs = eqks.mat[:,0]
        Tsegs_keep, Teqs_keep = sync_TsegsTeqs(info['t_segs'], t_eqs, Tsegs_maxdiff = 1)
        eqks.lld = eqks.lld[Teqs_keep, :]
        eqks.mat = eqks.mat[Teqs_keep, :]
        eqks.makeGM(div_y = 25)
        x_cuts = eqks.x_cuts
        y_cuts = eqks.y_cuts
        EQratios = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
        EQ_S1s = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
        EQ_S2s = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
        for ix in range(len(x_cuts)-1):
            for iy in range(len(y_cuts)-1):
                GMselected = eqks.GM_select(iy, ix)
                t_eqs = GMselected[:,0]
                if len(t_eqs) > 0:
                    eq_rates, eqs_S1, len_S1, eqs_S2, len_S2 = compute_EQrates(0.5, t_eqs, 
                                                               info, return_all = 'yes', 
                                                               shift = shift)
                    EQratios[iy, ix] = eq_rates[0]/(eq_rates[1]+eq_rates[0]) # -changed!!-
                    EQ_S1s[iy, ix] = eqs_S1
                    EQ_S2s[iy, ix] = eqs_S2
                else:
                    EQratios[iy, ix] = np.nan
                    EQ_S1s[iy, ix] = 0
                    EQ_S2s[iy, ix] = 0
        if (np.sum(EQratios>0.5)) > (np.sum(EQratios<0.5)):#np.nanmean(EQratios) > 0.5:
            switch = 'yes'
    if switch == 'yes':
        A_copy = info['A'].copy()
        info['A'][0,0] = A_copy[1,1]
        info['A'][1,0] = A_copy[0,1]
        info['A'][0,1] = A_copy[1,0]
        info['A'][1,1] = A_copy[0,0]
        B_copy = info['B'].copy()
        info['B'][0,:] = B_copy[1,:]
        info['B'][1,:] = B_copy[0,:]
        Gamma_copy = info['Gamma'].copy()
        info['Gamma'][0,:] = Gamma_copy[1,:]
        info['Gamma'][1,:] = Gamma_copy[0,:]
        pi_copy = info['pi'].copy()
        info['pi'][0] = pi_copy[1]
        info['pi'][1] = pi_copy[0]
    return info    
    
# In[] BM_CHMMMI and its systematic run
    
def BM_CHMMMI(dataname, col_names, indi_names, statistic_names, seglen, max_overlap, 
              min_it, n_labels, n_states, shift = 0):
    '''
    input format:
    dataname = 'DABA_full_ti2_fcln3_nonan_CVSK200_7to17'
    col_names = ['col1', 'col2']
    indi_names = ['Ss', 'Vs', 'Ks']
    statistic_names = ['mean', 'var']
    seglen: length of segments 
    '''
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    t_ind = data['col1']['t_ind']
    seg_pts = mB.get_seg_pts(0, len(t_ind)-1, seglen_max = seglen, seglen_min = seglen, 
                             max_overlap=max_overlap)
    seg_pts = np.int64(seg_pts)
    t_segs = t_ind[seg_pts[:,1]]
    input_names = list()
    segindic_TSs = list()
    for col_name in col_names:
        for indi_name in indi_names:
            col = data[col_name]
            indic_TS = col[indi_name]
            indic_TS = indic_TS[0,:]
            means_ts = np.zeros((len(seg_pts), ))
            vars_ts = np.zeros((len(seg_pts), ))
            for i in range(len(seg_pts)):
                means_ts[i] = np.nanmean(indic_TS[seg_pts[i,0]:seg_pts[i,1]])
                vars_ts[i] = np.nanvar(indic_TS[seg_pts[i,0]:seg_pts[i,1]])
            for s_name in statistic_names:
                if s_name == 'mean':
                    input_names.append(col_name + '_' + indi_name + '_MeanOfSegs')
                    segindic_TSs.append(means_ts)
                elif s_name == 'var':
                    input_names.append(col_name + '_' + indi_name + '_VarsOfSegs')
                    segindic_TSs.append(vars_ts)
    segindic_TSs = np.array(segindic_TSs)
    # standardizing segindic_TSs
    ys = segindic_TSs.copy()
    stds = np.zeros(ys.shape[0],)
    for i_row in range(ys.shape[0]):
        std = np.nanstd(ys[i_row, :])
        if std != 0:
            ys[i_row, :] = ys[i_row, :]/std
        stds[i_row] = std
    # kmeans labeling
    kmeans = KMeans(n_clusters = n_labels, init = 'k-means++')
    kmeans.fit(np.transpose(ys))
    # sort labels
    ccs = kmeans.cluster_centers_
    y_final = kmeans.labels_
    # randinit A, B, pi
    B = np.zeros((n_states, n_labels))
    for i in range(n_states):
        B[i,:] = np.random.rand(n_labels)
        B[i,:] = B[i,:]/np.sum(B[i,:])
    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        A[i,:] = np.random.rand(n_states)
        A[i,:] = A[i,:]/np.sum(A[i,:])
    pi = np.random.rand(n_states)
    pi = pi/np.sum(pi)
    # BW_HMM_log
    A_new, B_new, pi_new, model_scores = BW_HMM_log(y_final, A, B, pi, min_it=min_it)
    Gamma = comp_gamma(y_final, A_new, B_new, pi_new)
    # decide if model is successfully trained
    Gamma_S1 = np.sort(Gamma[0,:].copy())
    Gamma_5pct = Gamma_S1[np.int(len(Gamma_S1)*0.05)]
    Gamma_95pct = Gamma_S1[np.int(len(Gamma_S1)*0.95)]
    if (Gamma_5pct < 0.05) & (Gamma_95pct > 0.95):
        model_trained = 'yes'
        for i in range(n_states):
            if np.max(A_new[i,:]) < 0.9:
                model_trained = 'no'
    else:
        model_trained = 'no'
    info = {'t_segs': t_segs,
            'y': y_final,
            'ccs': ccs,
            'stds': stds,
            'Gamma': Gamma,
            'A': A_new,
            'B': B_new,
            'pi': pi_new,
            'model_scores': model_scores,
            'model_trained': model_trained,
            'segindic_TSs': segindic_TSs,
            'input_names': input_names,
            'seglen': seglen,
            'max_overlap': max_overlap,
            'min_it': min_it,
            'n_labels': n_labels,
            'n_states': n_states,
            'dataname': dataname}
    # standardize
    if model_trained == 'yes':
        HS_mags = np.array([0.0, 3.8])
        HS_rads = np.array([0.0, 2.0])
        info = standardize_HS(info, mags = HS_mags, rads = HS_rads, shift = shift)
        info['HS_mags'] = HS_mags
        info['HS_rads'] = HS_rads
    return info
#
def sys_BM_CHMMMI(indi_names, n_labelss, n_runs, save_name, 
                  dataname = 'DABA_full_ti2_fcln3_nonan_CVSK200_7to17',
                  statistic_names = ['mean', 'var'], col_names = ['col1', 'col2'],
                  seglen = 50, max_overlap = 2, min_it = 30, n_states = 2,
                  work_dir = [], maxtry_times = 2, allow_cv = 'no'):
    if len(work_dir) > 0:
        cd = os.getcwd()
        os.chdir(work_dir)
    max_try = np.int(n_runs*maxtry_times)
    dic_config = {}
    for j in range(len(n_labelss)):
        n_labels = n_labelss[j]
        n_passed = 0
        config = {}
        config_name = 'nlabels%s'%(n_labels)
        for r in range(max_try):
            print(n_passed)
            if n_passed == n_runs:
                break
            run_name = 'nlabels%s_try#%s'%(n_labels, r+1)
            info = BM_CHMMMI(dataname, col_names, indi_names, statistic_names, seglen, 
                              max_overlap, min_it, n_labels, n_states)
            if allow_cv == 'no':
                if max_try - r > n_runs - n_passed: # time left > No. needed converges
                    if info['model_trained'] == 'yes':
                        n_passed += 1
                        config[run_name] = info
                        print(run_name + ' converges.')
                    else:
                        print(run_name + ' not converges.')
                else:
                    n_passed += 1
                    config[run_name] = info
                    print(run_name + ' may not converged, but forced in.')
            else:
                n_passed += 1
                config[run_name] = info
                print(run_name + ' allowed.')
        dic_config[config_name] = config
    # save
    gen_paras = {'n_labelss': n_labelss}
    with open(save_name, 'wb') as f:
        pickle.dump(dic_config, f)
        pickle.dump(gen_paras, f)
    if len(work_dir) > 0:
        os.chdir(cd)
    return


# In[]
def BM_CHMMMI_day(dataname, col_names, indi_names, min_it, n_labels, 
                  n_states, shift = 0):
    '''
    modified from BM_CHMMMI
    '''
    with open('D_'+dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    t_ind = data['col1']['t_ind']
    t_segs = t_ind[1:]
    input_names = list()
    segindic_TSs = list()
    for col_name in col_names:
        for indi_name in indi_names:
            col = data[col_name]
            indic_TS = col[indi_name][0,:]
            segindic_TSs.append(indic_TS)
    segindic_TSs = np.array(segindic_TSs)
    # standardizing segindic_TSs
    ys = segindic_TSs.copy()
    stds = np.zeros(ys.shape[0],)
    for i_row in range(ys.shape[0]):
        std = np.nanstd(ys[i_row, :])
        if std != 0:
            ys[i_row, :] = ys[i_row, :]/std
        stds[i_row] = std
    # kmeans labeling
    kmeans = KMeans(n_clusters = n_labels, init = 'k-means++')
    kmeans.fit(np.transpose(ys))
    # sort labels
    ccs = kmeans.cluster_centers_
    y_final = kmeans.labels_
    # randinit A, B, pi
    B = np.zeros((n_states, n_labels))
    for i in range(n_states):
        B[i,:] = np.random.rand(n_labels)
        B[i,:] = B[i,:]/np.sum(B[i,:])
    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        A[i,:] = np.random.rand(n_states)
        A[i,:] = A[i,:]/np.sum(A[i,:])
    pi = np.random.rand(n_states)
    pi = pi/np.sum(pi)
    # BW_HMM_log
    A_new, B_new, pi_new, model_scores = BW_HMM_log(y_final, A, B, pi, min_it=min_it)
    Gamma = comp_gamma(y_final, A_new, B_new, pi_new)
    # decide if model is successfully trained
    Gamma_S1 = np.sort(Gamma[0,:].copy())
    Gamma_5pct = Gamma_S1[np.int(len(Gamma_S1)*0.05)]
    Gamma_95pct = Gamma_S1[np.int(len(Gamma_S1)*0.95)]
    if (Gamma_5pct < 0.05) & (Gamma_95pct > 0.95):
        model_trained = 'yes'
        for i in range(n_states):
            if np.max(A_new[i,:]) < 0.9:
                model_trained = 'no'
    else:
        model_trained = 'no'
    info = {'t_segs': t_segs,
            'y': y_final,
            'ccs': ccs,
            'stds': stds,
            'Gamma': Gamma,
            'A': A_new,
            'B': B_new,
            'pi': pi_new,
            'model_scores': model_scores,
            'model_trained': model_trained,
            'segindic_TSs': segindic_TSs,
            'input_names': input_names,
            'min_it': min_it,
            'n_labels': n_labels,
            'n_states': n_states,
            'dataname': dataname}
    # standardize
    if model_trained == 'yes':
        HS_mags = np.array([0.0, 3.8])
        HS_rads = np.array([0.0, 2.0])
        info = standardize_HS(info, mags = HS_mags, rads = HS_rads, shift = shift)
        info['HS_mags'] = HS_mags
        info['HS_rads'] = HS_rads
    return info


def sys_BM_CHMMMI_day(indi_names, n_labelss, n_runs, save_name, 
                      dataname = 'DABA_full_ti2_fcln3_nonan_CVSK200_7to17',
                      col_names = ['col1', 'col2'], min_it = 30, n_states = 2,
                      work_dir = [], maxtry_times = 5, allow_cv = 'no',
                      converged_only = 'no'):
    if len(work_dir) > 0:
        cd = os.getcwd()
        os.chdir(work_dir)
    max_try = np.int(n_runs*maxtry_times)
    dic_config = {}
    for j in range(len(n_labelss)):
        n_labels = n_labelss[j]
        n_passed = 0
        config = {}
        config_name = 'nlabels%s'%(n_labels)
        for r in range(max_try):
            print(n_passed)
            if n_passed == n_runs:
                break
            run_name = 'nlabels%s_try#%s'%(n_labels, r+1)
            info = BM_CHMMMI_day(dataname, col_names, indi_names, min_it, n_labels, 
                                 n_states)
            if allow_cv == 'no':
                if max_try - r > n_runs - n_passed: # time left > No. needed converges
                    if info['model_trained'] == 'yes':
                        n_passed += 1
                        config[run_name] = info
                        print(run_name + ' converges.')
                    else:
                        print(run_name + ' not converges.')
                else:
                    n_passed += 1
                    if converged_only == 'no':
                        config[run_name] = info
                        print(run_name + ' may not converged, but forced in.')
                    else:
                        print(run_name + ' not converges and discarded.')
            else:
                n_passed += 1
                config[run_name] = info
                print(run_name + ' allowed.')
        dic_config[config_name] = config
    # save
    gen_paras = {'n_labelss': n_labelss}
    with open(save_name, 'wb') as f:
        pickle.dump(dic_config, f)
        pickle.dump(gen_paras, f)
    if len(work_dir) > 0:
        os.chdir(cd)
    return

# In[]
def sys_BM_CHMMMI_day_0125(indi_names, n_labelss, n_runs, save_name, 
                           dataname = 'XXXX_...', col_names = ['col1', 'col2'], 
                           min_it = 30, n_states = 2, work_dir = []):
    '''
    new and simplified approach: initialize 100 times and later only choose the 
    model with the highest score
    '''
    if len(work_dir) > 0:
        cd = os.getcwd()
        os.chdir(work_dir)
    # max_try = np.int(n_runs*maxtry_times)
    dic_config = {}
    for j in range(len(n_labelss)):
        n_labels = n_labelss[j]
        config = {}
        config_name = 'nlabels%s'%(n_labels)
        for r in range(n_runs):
            run_name = 'nlabels%s_try#%s'%(n_labels, r+1)
            info = BM_CHMMMI_day(dataname, col_names, indi_names, min_it, n_labels, 
                                 n_states)
            config[run_name] = info
            if info['model_trained'] == 'yes':
                print(run_name + ' converges.')
            else:
                print(run_name + ' not converges.')
        dic_config[config_name] = config
    # save
    gen_paras = {'n_labelss': n_labelss}
    with open(save_name, 'wb') as f:
        pickle.dump(dic_config, f)
        pickle.dump(gen_paras, f)
    if len(work_dir) > 0:
        os.chdir(cd)
    return

# In[]
def cut_ends(seq, ratio_upper = 0.04, ratio_lower = 0.01):
    '''
    funciton cuts the upper most / lower most values of a seq, but disorderd
    '''
    seq_sorted = np.sort(seq.copy())
    cut_upper = np.int(ratio_upper * len(seq))
    cut_lower = np.int(ratio_lower * len(seq))
    seq_return = seq_sorted[cut_lower:len(seq) - cut_upper]
    return seq_return


# In[]
def score_dist_plot(file_name, note = '', save = 'yes'):
    # function simply plot the distogram of model scores
    # file_name e.g. 'BM_CHMMMI_DABA_ti2_FCW500S2_CVSK200_CVSK_R100L90_0416G.pickle'
    with open(file_name, 'rb') as f:
        dic_config = pickle.load(f)
    dic_scores = {}
    dic_names = {}
    for config in dic_config:
        scores = list()
        info_names = list()
        info_dic = dic_config[config]
        for info_name in info_dic:
            scores.append(info_dic[info_name]['model_scores'][-1])
            info_names.append(info_name)
        dic_scores[config] = scores
        dic_names[config] = info_names
    # In score dist plotting
    fig = plt.figure(figsize=(24,14))
    mean_scores = list()
    for list_name in dic_scores:
        current_scores = np.array(dic_scores[list_name])
        mean_score = np.int(np.mean(current_scores))
        sns.distplot(current_scores, bins = 12, label = list_name + '_mean=%s'%(mean_score))
        mean_scores.append(mean_score)
    global_mean = np.int(np.mean(mean_scores))
    plt.legend()
    plt.xlabel('Log Model Score')
    plt.ylabel('PDF')
    figure_name = file_name[0:-7]+note+'_scores_%sRunsEach_mean=%s'%(len(scores), 
                                        global_mean)
    plt.title(figure_name)
    if save == 'yes':
        plt.savefig(figure_name +'.png', dpi = 500)
        plt.close(fig)
    return

# In[]
def atlas_plot(file_name, state_threshold = 0.5, marker = 'default', shift = 0, 
               config_use = [1,0,0,0], color_fix = 'no', score_multi = 1):
    '''
    plot the model score vs first row of heatmap average
    file_name e.g. 'BM_CHMMMI_DABA_ti2_FCW500S2_CVSK200_CVSK_R100L90_0416G.pickle'
    config_use is for choosing which config(s) (i.e. n_labels) in dic_config to plot 
    '''
    with open(file_name, 'rb') as f:
        dic_config = pickle.load(f)
    dic_scores = {}
    dic_performances = {}
    dic_names = {}
    i_config = 0
    for config in dic_config:
        if config_use[i_config] == 0:
            i_config += 1
            continue
        i_config += 1
        scores = list()
        heatmap_performance = list()
        info_names = list()
        info_dic = dic_config[config]
        for info_name in info_dic:
            info = info_dic[info_name]
            eqks = mB.eqk()
            HS_mags, HS_rads = info['HS_mags'], info['HS_rads']
            eqks.select_ULB(info['dataname'], mag_UB = HS_mags[-1], mag_LB = HS_mags[0], 
                            rad_UB = HS_rads[-1], rad_LB = HS_rads[0], verbose = 'no')
            t_eqs = eqks.selected[:,0]
            eq_ratesG = compute_EQrates(state_threshold, t_eqs, info, shift = shift)
            EQratio_global = eq_ratesG[0]/eq_ratesG[1]
            if EQratio_global > 1:
                EQratio_global = np.nan
            scores.append(info['model_scores'][-1] * score_multi)
            info_names.append(info_name)
            heatmap_performance.append(EQratio_global)
        dic_performances[config] = heatmap_performance
        dic_scores[config] = scores
        dic_names[config] = info_names
    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
    i_color = 0
    for config in dic_scores:
        if marker == 'default':
            plt.scatter(dic_scores[config], dic_performances[config], 
                        label = file_name[7:-7] + '_' + config)
        else:
            if color_fix == 'yes':
                plt.scatter(dic_scores[config], dic_performances[config], marker = marker, 
                            label = file_name[7:-7] + '_' + config,
                            color = colors[i_color])
                i_color += 1
            else:
                plt.scatter(dic_scores[config], dic_performances[config], marker = marker, 
                            label = file_name[7:-7] + '_' + config)
    plt.legend()
    plt.xlabel('Log model score')
    plt.ylabel('EQratio at  mag in [%s, %s], rad in [%s, %s]'%(HS_mags[0], HS_mags[1],
               HS_rads[0], HS_rads[1]))
    return

# In[]
import scipy.cluster.hierarchy as sch
def plot_ccs_distance(info):
    ccs_sorted = info['ccs']
    n_labels = ccs_sorted.shape[0]
    ccs_distances = np.zeros((n_labels, n_labels))
    ccs_standardized = ccs_sorted.copy()
    for i in range(ccs_sorted.shape[1]):
        temp_vec = ccs_standardized[:,i]
        temp_vec = temp_vec / np.nanstd(temp_vec)
        temp_vec = temp_vec - np.nanmean(temp_vec)
        ccs_standardized[:,i] = temp_vec
    for i in range(n_labels):
        for j in range(n_labels):
            row_i = ccs_standardized[i,:]
            row_j = ccs_standardized[j,:]
            ccs_distances[i,j] = np.nansum((row_i - row_j)**2)
    #
    D = ccs_distances
    Y = sch.linkage(D, method='centroid')
    Z = sch.dendrogram(Y, orientation='right', no_plot=True)
    index = Z['leaves']
    info['ccs'] = info['ccs'][index,:]
    D = D[index,:]
    D = D[:,index]
    # added 2021-04-08
    if np.median(D[0:5, 0:5]) > np.median(D[-5:, -5:]):
        reverse_ind = np.arange(len(D), 1, -1)-1
        D = D[reverse_ind,:]
        D = D[:, reverse_ind]
    #
    sns.heatmap(D, cmap="YlGnBu")
    return

# In[] HSVA
def HSVA_voting(pS1s, spread = 0.1):
    n_runs, len_HS = pS1s.shape[0], pS1s.shape[1]
    pS1s_voted = np.zeros(len_HS,)
    pS1s_voted_UB = np.zeros(len_HS,)
    pS1s_voted_LB = np.zeros(len_HS,)
    for i in range(len_HS):
        pS1s_i = np.sort(pS1s[:,i])
        pS1s_voted[i] = pS1s_i[np.int(n_runs*0.5)]
        pS1s_voted_UB[i] = pS1s_i[np.int(n_runs*(0.5+spread))] 
        pS1s_voted_LB[i] = pS1s_i[np.int(n_runs*(0.5-spread))]
    return pS1s_voted, pS1s_voted_UB,  pS1s_voted_LB

def decide_label(test_indic, ccs):
    distances = np.zeros((ccs.shape[0],))
    for i in range(ccs.shape[0]):
        distances[i] = np.sum((ccs[i,:] - test_indic)**2)
    i_label = np.argmin(distances)
    return i_label

def HSCA(info_dic, info_max = 100, input_indic_TSs = []):
    '''
    hidden state constructing algorithm
    use the 'info_max' number of HMM-kmeans models stored in info_dic to recontruct 
    info_max number of HS TS, which can be processed with HSVA
    input_indic_TSs is the ~16-by-time segment indicator time series, in the same 
    indicator composition order as that in info_dic
    input info_dic can be either real info_dic or file_name-like:
    'CHMMMI_DABA_ti2_FCW500S2_CVSK200_cV_CVSK_R300L90_0428Ar.pickle'
    '''
    if type(info_dic) == str: 
        with open(info_dic, 'rb') as f:
            dic_config = pickle.load(f)
        for config in dic_config:
            info_dic = dic_config[config]
    i_info = 0
    if len(input_indic_TSs) == 0:
        input_new = 'no'
    else:
        input_new = 'yes'
    for info_name in info_dic:
        info = info_dic[info_name]
        if input_new != 'yes':
            input_indic_TSs = info['segindic_TSs']
        if i_info == 0:
            pS1s_input = np.zeros((len(info_dic), input_indic_TSs.shape[1]))
            ys_input = np.zeros((len(info_dic), input_indic_TSs.shape[1]), dtype = int)
        y_input = np.zeros((input_indic_TSs.shape[1]), dtype = int)
        for i in range(input_indic_TSs.shape[1]):
            input_indic = input_indic_TSs[:,i]
            stds =  info['stds']
            input_indic_stdd = input_indic/stds
            ccs = info['ccs']
            i_label = decide_label(input_indic_stdd, ccs)
            y_input[i] = i_label
        ys_input[i_info,:] = y_input
        Gamma = comp_gamma(y_input, info['A'], info['B'], np.array([0.5, 0.5]))
        pS1s_input[i_info,:] = Gamma[0,:]
        i_info += 1
        if i_info > info_max:
            break
    if info_max < len(info_dic):
        ys_input = ys_input[0:info_max, :]
        pS1s_input = pS1s_input[0:info_max, :]
    return pS1s_input

def HSCA_TrTe(TrData = 'CHMMMI_..._cV_10of10_TR_CVSK_R100L90_0704AH2S.pickle', 
              TeData = 'CHMMMI_..._cV_10of10_TE_CVSK_R1L90.pickle', 
              nlabels = 'nlabels90'):
    '''
    modified from HSCA, this function directly creates results using the trained model
    in TrData, and applying on segindic_TSs from TeData, and save prediction results
    '''
    # Tr
    with open(TrData, 'rb') as f:
        dic_config = pickle.load(f)
        gen_paras = pickle.load(f)
    for config in dic_config:
        info_dic = dic_config[config]
    # Te
    with open(TeData, 'rb') as f:
        data = pickle.load(f)
    for name in data[nlabels]:
        input_indic_TSs = data[nlabels][name]['segindic_TSs']
        input_t_segs = data[nlabels][name]['t_segs']
    # Te
    info_dic_Te = info_dic.copy()
    i_info = 0
    for info_name in info_dic:
        info = info_dic[info_name]
        if i_info == 0:
            pS1s_input = np.zeros((len(info_dic), input_indic_TSs.shape[1]))
            ys_input = np.zeros((len(info_dic), input_indic_TSs.shape[1]), dtype = int)
        y_input = np.zeros((input_indic_TSs.shape[1]), dtype = int)
        for i in range(input_indic_TSs.shape[1]):
            input_indic = input_indic_TSs[:,i]
            stds =  info['stds']
            input_indic_stdd = input_indic/stds
            ccs = info['ccs']
            i_label = decide_label(input_indic_stdd, ccs)
            y_input[i] = i_label
        ys_input[i_info,:] = y_input
        Gamma = comp_gamma(y_input, info['A'], info['B'], np.array([0.5, 0.5]))
        pS1s_input[i_info,:] = Gamma[0,:]
        info_dic_Te[info_name]['Gamma'] = Gamma
        info_dic_Te[info_name]['segindic_TSs'] = input_indic_TSs
        info_dic_Te[info_name]['t_segs'] = input_t_segs
        info_dic_Te[info_name]['y'] = y_input
        info_dic_Te[info_name]['pi'] = Gamma[:,0]
        i_info += 1
    with open(TrData[0:-7]+'_TEpred.pickle', 'wb') as f:
        pickle.dump(dic_config, f)
        pickle.dump(gen_paras, f)
    return

# In
def reHS(file_name, HS_mags = np.array([0, 3.8]), HS_rads = np.array([0.0, 2.0]), shift = 0):
    '''
    re-HS genereated CHMMMI data, 
    file_name e.g. 'CHMMMI_DABA_ti2_FCW500S2_CVSK200_cV_CVSK_R30L90_0427A.pickle'
    '''
    with open(file_name, 'rb') as f:
        dic_config = pickle.load(f)
        gen_paras = pickle.load(f)
    # initial HS: HS all with the first model, so they do not contradict each other much
    #
    for config in dic_config:
        info_dic = dic_config[config]
        for info_name in info_dic:
            info = info_dic[info_name]
            info = standardize_HS(info, mags = HS_mags, rads = HS_rads, shift = shift)
            info['HS_mags'] = HS_mags
            info['HS_rads'] = HS_rads
    with open(file_name[0:-7]+'NH.pickle', 'wb') as f:
        pickle.dump(dic_config, f)
        pickle.dump(gen_paras, f)
    return

def second_reHS(file_name, time_HS = 1, flip_all = 'no', shift = 0):
    '''
    re-HS genereated CHMMMI data, 
    file_name e.g. 'CHMMMI_DABA_ti2_FCW500S2_CVSK200_cV_CVSK_R30L90_0427A.pickle'
    '''
    with open(file_name, 'rb') as f:
        dic_config = pickle.load(f)
        gen_paras = pickle.load(f)
    for i in range(time_HS):
        if flip_all == 'no':
            info_v = HSVA_getinfo(dic_config)
            pS1s = info_v['pS1s']
            pS1s_voted, _, _ = HSVA_voting(pS1s)
        for config in dic_config:
            info_dic = dic_config[config]
            for info_name in info_dic:
                info = info_dic[info_name]
                if flip_all == 'no':
                    pS1 = info['Gamma'][0,:]
                    diff_seq = np.abs(pS1 - pS1s_voted)
                    if np.sum(diff_seq > 0.5)/len(diff_seq) > 0.5:
                        info = standardize_HS(info, force = 'yes', shift = shift)
                else:
                    info = standardize_HS(info, force = 'yes', shift = shift)
                info_dic[info_name] = info
            dic_config[config] = info_dic
    # final check
    if flip_all == 'yes':
        save_id = 'F'
    else:
        info_v = HSVA_getinfo(dic_config)
        eqks = mB.eqk()
        t_eqs = eqks.mat[:,0]
        Tsegs_keep, Teqs_keep = sync_TsegsTeqs(info_v['t_segs'], t_eqs, Tsegs_maxdiff = 1)
        eqks.lld = eqks.lld[Teqs_keep, :]
        eqks.mat = eqks.mat[Teqs_keep, :]
        eqks.makeGM(div_y = 30)
        x_cuts = eqks.x_cuts
        y_cuts = eqks.y_cuts
        EQratios = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
        EQ_S1s = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
        EQ_S2s = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
        for ix in range(len(x_cuts)-1):
            for iy in range(len(y_cuts)-1):
                GMselected = eqks.GM_select(iy, ix)
                t_eqs = GMselected[:,0]
                if len(t_eqs) > 0:
                    eq_rates, eqs_S1, len_S1, eqs_S2, len_S2 = compute_EQrates(0.45, t_eqs, 
                                                               info_v, return_all = 'yes', 
                                                               shift = shift)
                    EQratios[iy, ix] = eq_rates[0]/(eq_rates[1]+eq_rates[0]) # -changed!!-
                    EQ_S1s[iy, ix] = eqs_S1
                    EQ_S2s[iy, ix] = eqs_S2
                else:
                    EQratios[iy, ix] = np.nan
                    EQ_S1s[iy, ix] = 0
                    EQ_S2s[iy, ix] = 0
        if (np.sum(EQratios>0.5)) > (np.sum(EQratios<0.5)):
            print('Fliped all in final check: ' + file_name)
            for config in dic_config:
                info_dic = dic_config[config]
                for info_name in info_dic:
                    info = info_dic[info_name]
                    info = standardize_HS(info, force = 'yes', shift = shift)
                    info_dic[info_name] = info
                dic_config[config] = info_dic
        save_id = '%sS'%time_HS
    with open(file_name[0:-7]+save_id+'.pickle', 'wb') as f:
        pickle.dump(dic_config, f)
        pickle.dump(gen_paras, f)
    return

# In
def HSVA_getinfo(file_name, old_version = 'yes'):
    '''
    file_name like 'CHMMMI_DABA_ti2_FCW500S2_CVSK200_cV_CVSK_R300L90_0428Ar.pickle'
    file_name can also be dic_config dictionary format
    function creates pS1s for the info_dic, and proceed to HSVA
    returns a info_v that is ready to be put into bm.plot_EQrates
    '''
    if type(file_name) == str:
        with open(file_name, 'rb') as f:
            dic_config = pickle.load(f)
    else:
        dic_config = file_name # the alternative input format
        file_name = file_name.copy()
    for config in dic_config:
        info_dic = dic_config[config]
    i_info = 0
    for info_name in info_dic:
        info = info_dic[info_name]
        if i_info == 0:
            pS1s = np.zeros((len(info_dic), info['Gamma'].shape[1]))
            pS1s[i_info, :] = info['Gamma'][0,:]
        else:
            pS1s[i_info, :] = info['Gamma'][0,:]
        i_info += 1    
    pS1s_voted, _, _ = HSVA_voting(pS1s)
    info_v = {}
    pS2s_voted = 1 - pS1s_voted
    Gamma_v = np.zeros((2, len(pS1s_voted)))
    Gamma_v[0,:], Gamma_v[1,:] = pS1s_voted, pS2s_voted
    if type(file_name) != str:
        file_name = 'CHMMMI_'+info['dataname']+'_X'
    info_v = {'Gamma': Gamma_v,
              't_segs': info['t_segs'],
              'dataname': info['dataname'],
              'pS1s': pS1s,
              'pS1s_voted': pS1s_voted,
              'pS2s_voted': pS2s_voted,
              'file_name': file_name,
              'segindic_TSs': info['segindic_TSs'],
              'input_names': info['input_names']}
    if old_version == 'yes':
        info_v['HS_mags'] = info['HS_mags']
        info_v['HS_rads'] = info['HS_rads']
    return info_v

def sort_HSTSs(final_scores, P_S1s):
    # function written for HSVA_getinfo_0125(file_name, sort_HS = 'yes')
    final_scores[np.isnan(final_scores)] = np.nanmin(final_scores)-1 # deal with nan
    sort_order = np.argsort(final_scores)
    sort_order = sort_order[::-1]
    P_S1s_sorted = P_S1s[sort_order, :]
    pS1s_best = P_S1s_sorted[0,:]
    for i in range(len(sort_order)-1):
        diff_seq = np.abs(P_S1s_sorted[i+1,:] - pS1s_best)
        if np.nanmean(diff_seq) > 0.5:
            P_S1s_sorted[i+1,:] = 1 - P_S1s_sorted[i+1,:]
    return P_S1s_sorted

def HSVA_getinfo_0125(file_name, sort_HS = 'no', keep_models = 'no', max_models = 30):
    '''
    adepted from HSVA_getinfo, sort_HS = 'yes' to add sorted HS TSs, 
    keep_models = 'yes' to add the model parameters of the optimal model
    '''
    with open(file_name, 'rb') as f:
        dic_config = pickle.load(f)
    for info_name in dic_config:
        info_dic = dic_config[info_name]
        num_models = len(info_dic)
        model_scores = np.zeros((max_models, 31))
        ind_dic = 0 # this loop stores the models scores
        for model_name in info_dic:
            if ind_dic < max_models:
                model_score = info_dic[model_name]['model_scores']
                model_scores[ind_dic, :] = model_score
                ind_dic += 1
        gamma_len = np.shape(info_dic[model_name]['Gamma'])[1]
        P_S1s = np.zeros((num_models, gamma_len))
        ind_dic = 0 # this loop stores P_S1s while modify model socres in case of anomaly
        for model_name in info_dic:
            if ind_dic < max_models:
                P_S1s[ind_dic] = info_dic[model_name]['Gamma'][0,:]
                if np.sum(np.isnan(P_S1s[ind_dic])) > 0:
                    model_scores[ind_dic, :] = np.nanmin(model_scores[:,-1])-1
                elif (np.sum(P_S1s[ind_dic]) < 1)|(np.sum(P_S1s[ind_dic]) > gamma_len-1):
                    model_scores[ind_dic, :] = np.nanmin(model_scores[:,-1])-1
                ind_dic += 1
        ind_optimal = np.nanargmax(model_scores[:,-1]) # get the model with optimal ind
        if keep_models == 'yes':
            ind_dic = 0 # this loop stores A, B, pi if applicable
            for model_name in info_dic:
                if ind_dic == ind_optimal:
                    A_opti = info_dic[model_name]['A']
                    B_opti = info_dic[model_name]['B']
                    pi_opti = info_dic[model_name]['pi']
                    break
                else:
                    ind_dic += 1
        break # only intend one item in dic_config now
    pS1s_voted = P_S1s[ind_optimal,:]
    model_scores_opti = model_scores[ind_optimal,:]
    info_v = {}
    pS2s_voted = 1 - pS1s_voted
    Gamma_v = np.zeros((2, len(pS1s_voted)))
    Gamma_v[0,:], Gamma_v[1,:] = pS1s_voted, pS2s_voted
    info = info_dic[model_name]
    info_v = {'Gamma': Gamma_v,
              't_segs': info['t_segs'],
              'dataname': info['dataname'],
              'pS1s_voted': pS1s_voted,
              'pS2s_voted': pS2s_voted,
              'file_name': file_name,
              'model_scores_opti': model_scores_opti,
              'segindic_TSs': info['segindic_TSs'],
              'input_names': info['input_names']}
    if keep_models == 'yes':
        info_v['A'] = A_opti
        info_v['B'] = B_opti
        info_v['pi'] = pi_opti
    if sort_HS == 'yes':
        final_scores = model_scores[:,-1]
        P_S1s_sorted = sort_HSTSs(final_scores, P_S1s)
        info_v['pS1s'] = P_S1s_sorted
    return info_v

# In[] ensemble part
    
def split_len_of_X(len_of_X, seg_UB = 100):
    lenc = len_of_X.copy()
    while np.sum(np.array(lenc) > seg_UB) > 0:
        for i in range(len(lenc)):
            if lenc[i] > seg_UB:
                split_part1 = np.random.randint(lenc[i]-seg_UB/20)
                lenc.append(lenc[i]-split_part1)
                lenc[i] = split_part1
    return lenc

def create_HSensembles(info, num_ensemble = 20):
    Gamma = info['Gamma'].copy()
    Gamma[0,Gamma[0,:] < 0.5] = 0
    Gamma[0,Gamma[0,:] >= 0.5] = 1
    Gamma[1,Gamma[1,:] < 0.5] = 0
    Gamma[1,Gamma[1,:] >= 0.5] = 1
    P_S1 = Gamma[0,:]
    len_of_0 = []
    len_of_1 = []
    current_S = P_S1[0]
    current_len = 0 
    for i in range(len(P_S1)):
        if P_S1[i] != current_S:
            if current_S == 0:
                len_of_0.append(current_len)
            else:
                len_of_1.append(current_len)
            current_S = P_S1[i]
            current_len = 0
        current_len += 1
    if current_S == 0:
        len_of_0.append(current_len)
    else:
        len_of_1.append(current_len)
    len_of_0c = np.array(len_of_0.copy())
    len_of_1c = np.array(len_of_1.copy())
    len_of_0c = len_of_0c[len_of_0c > len(P_S1)/100]
    len_of_1c = len_of_1c[len_of_1c > len(P_S1)/100]
    # 
    seg_UB = np.mean(np.concatenate((len_of_0c, len_of_1c)))
    if seg_UB > len(P_S1)/6:
        seg_UB = len(P_S1)/6
    ensembles = np.zeros((num_ensemble+1, len(P_S1)))
    for i in range(num_ensemble):
        len0_i = split_len_of_X(len_of_0, seg_UB)
        len1_i = split_len_of_X(len_of_1, seg_UB)
        a = np.random.permutation(len(len1_i) + len(len0_i))
        seq = []
        for j in range(len(a)):
            if a[j] < len(len0_i):
                seq = seq + [0]*len0_i[a[j]]
            else:
                seq = seq + [1]*len1_i[a[j]-len(len0_i)]
        ensembles[i+1,:] = np.array(seq)
    ensembles[0,:] = P_S1
    return ensembles


def compute_EQGM_ensemble(info, state_threshold = 0.5, div_y = 30, div_x = 0, min_EQ = 5, 
                          mag_LB = 0, num_ensemble = 20, shift = 0):
    eqks = mB.eqk()
    if mag_LB > 0:
        M_eqs = eqks.mat[:,1].copy()
        eqks.lld = eqks.lld[M_eqs>mag_LB, :]
        eqks.mat = eqks.mat[M_eqs>mag_LB, :]
    t_eqs = eqks.mat[:,0]
    Tsegs_keep, Teqs_keep = sync_TsegsTeqs(info['t_segs'], t_eqs, Tsegs_maxdiff = 1)
    eqks.lld = eqks.lld[Teqs_keep, :]
    eqks.mat = eqks.mat[Teqs_keep, :]
    eqks.makeGM(div_y = div_y, div_x = div_x)
    x_cuts = eqks.x_cuts
    y_cuts = eqks.y_cuts
    ensembles = create_HSensembles(info, num_ensemble = num_ensemble)
    EQratios_en = np.zeros((num_ensemble, len(y_cuts)-1, len(x_cuts)-1))
    for i_en in range(num_ensemble):
        len_TS = len(info['t_segs'])
        Gamma = np.zeros((2, len_TS))
        Gamma[0,:] = ensembles[i_en, :]
        Gamma[1,:] = 1- ensembles[i_en, :]
        sub_info = {'Gamma': Gamma, 't_segs':info['t_segs']}
        EQratios = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
        EQratios[:] = np.nan
        for ix in range(len(x_cuts)-1):
            for iy in range(len(y_cuts)-1):
                GMselected = eqks.GM_select(iy, ix)
                t_eqs = GMselected[:,0]
                if len(t_eqs) > min_EQ:
                    eq_rates = compute_EQrates(state_threshold, t_eqs, sub_info, 
                                               return_all = 'no', min_EQ = min_EQ, shift = shift)
                    EQratios[iy, ix] = eq_rates[0]/(eq_rates[1]+eq_rates[0]) 
#        EQratios_en[i_en, :, :] = np.flipud(EQratios)
        EQratios_en[i_en, :, :] = EQratios
    return EQratios_en 

# In[] EQGM parts
def get_tick(tick_label, tick_labels):
    # function return tick given wanted tick_label, assuming ticks are 0, 1, 2, ...
    # tl = k*t + h, where h = tls[0]
    k = tick_labels[1] - tick_labels[0]
    return (tick_label - tick_labels[0])/k

def plot_EQGM(info, state_threshold = 0.5, div_y = 30, div_x = 0, plot = 'sub', 
              sub_inds = [2,3,4], plot_prefix = ['', '', ''], NS_keep_eq = '', 
              return_info = 'no', min_EQ = 5, mag_LB = 0, EQrate_mode = 'S1',
              num_ensemble = 200, fully_save = 'yes', shift = 0):
    # EQrate_mode can be 'S1' or 'abs': 'S1' for normal, 'abs' for measuring discrimination 
    # 'hypo' for simple binomial hypothesis, 'ensemble' for ensemble test
    eqks = mB.eqk()
    if len(NS_keep_eq) == 1:
        eqks.NS_divide(keep_eq = NS_keep_eq)
    div_x = div_y # new update on 20 Oct 2020
    eqks.makeGM(div_y = div_y, div_x = div_x) # shifted up on 20 Oct 2020
    if mag_LB > 0:
        M_eqs = eqks.mat[:,1].copy()
        eqks.lld = eqks.lld[M_eqs>mag_LB, :]
        eqks.mat = eqks.mat[M_eqs>mag_LB, :]
    # modified 07 Sep 2020
    t_eqs = eqks.mat[:,0].copy()
    eqks.lld = eqks.lld[t_eqs <= info['t_segs'][-1], :]
    eqks.mat = eqks.mat[t_eqs <= info['t_segs'][-1], :]
    t_eqs = eqks.mat[:,0].copy()
    eqks.lld = eqks.lld[t_eqs >= info['t_segs'][0], :]
    eqks.mat = eqks.mat[t_eqs >= info['t_segs'][0], :]
    x_cuts = eqks.x_cuts
    y_cuts = eqks.y_cuts
    EQratios = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
    EQ_S1s = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
    EQ_S2s = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
    if EQrate_mode == 'ensemble':
        EQratios_en = compute_EQGM_ensemble(info, div_y = div_y, num_ensemble = num_ensemble,
                                            min_EQ = min_EQ, mag_LB = mag_LB, shift = shift)
    for ix in range(len(x_cuts)-1):
        for iy in range(len(y_cuts)-1):
            GMselected = eqks.GM_select(iy, ix)
            t_eqs = GMselected[:,0]
            if len(t_eqs) > min_EQ:
                eq_rates, eqs_S1, len_S1, eqs_S2, len_S2 = compute_EQrates(state_threshold, 
                                                           t_eqs, info, return_all = 'yes',
                                                           min_EQ = min_EQ, shift = shift)
                if EQrate_mode == 'S1':
                    EQratios[iy, ix] = eq_rates[0]/(eq_rates[1]+eq_rates[0]) # -changed!!-
                if EQrate_mode == 'abs':
                    EQratios[iy, ix] = np.min(eq_rates)/np.max(eq_rates)
                if EQrate_mode == 'hypo':
                    EQratios[iy, ix] = cm.compute_P_binohypo(eqs_S1, len_S1, eqs_S2, len_S2)
                elif EQrate_mode == 'ensemble':
                    EQratio_xy = eq_rates[0]/(eq_rates[1]+eq_rates[0])
                    EQratio_en_xy = EQratios_en[:, iy, ix]
                    EQratios[iy, ix] = np.sum(EQratio_xy > EQratio_en_xy)/num_ensemble
                EQ_S1s[iy, ix] = eqs_S1
                EQ_S2s[iy, ix] = eqs_S2
            else:
                EQratios[iy, ix] = np.nan
                if len(t_eqs) > 0:
                    _, eqs_S1, len_S1, eqs_S2, len_S2 = compute_EQrates(state_threshold, 
                                                        t_eqs, info, return_all = 'yes',
                                                        min_EQ = min_EQ, shift = shift)
                    EQ_S1s[iy, ix] = eqs_S1
                    EQ_S2s[iy, ix] = eqs_S2
    x_cuts = np.round(x_cuts*100)/100
    y_cuts = np.round(y_cuts*100)/100
#    if plot != 'no':
    EQratios_plot = np.flipud(EQratios)
    EQS1_plot = np.flipud(EQ_S1s)
    EQS2_plot = np.flipud(EQ_S2s)
    EQS1_plot[EQS1_plot == 0] = np.nan
    EQS2_plot[EQS2_plot == 0] = np.nan
    xticks = np.arange(0, len(x_cuts),1)-0.5
    xtick_labels = x_cuts
    yticks = np.arange(0, len(y_cuts),1)-0.5
    ytick_labels = np.flipud(y_cuts)
    stn_name = info['dataname'][0:4]
    stn_coord = eqks.position[stn_name]
    fs = 7
    if plot == 'yes':
        ax = plt.subplot() 
        if (EQrate_mode == 'S1')|(EQrate_mode == 'hypo')|(EQrate_mode == 'ensemble'):
            plt.imshow(EQratios_plot, cmap="PiYG", vmin=0, vmax=1)
        elif EQrate_mode == 'abs':
            plt.imshow(EQratios_plot, cmap="Reds_r")
        for iy in range(len(y_cuts)-1):
            for ix in range(len(x_cuts)-1):
                if ~np.isnan(EQratios_plot[iy, ix]):
                    entry = EQratios_plot[iy, ix]*100
                    if (EQrate_mode == 'S1')|(EQrate_mode == 'hypo')|(EQrate_mode == 'ensemble'):
                        if (entry > 25) & (entry < 75):
                            c = 'k'
                        else:
                            c = 'w'
                    elif EQrate_mode == 'abs':
                        if entry > 50:
                            c = 'k'
                        else:
                            c = 'w'
                    ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                            fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 90)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0)
        ax.scatter(get_tick(stn_coord[1],xtick_labels)-0.5,get_tick(stn_coord[0],ytick_labels)-0.5, 
                   marker = '*', color='r', s = 240, label = stn_name)
        plt.legend()
        if EQrate_mode == 'S1': 
            ax.set_title('Heatmap (%) of R_f')
        if EQrate_mode == 'abs':
            ax.set_title('EQratios (%) heatmap of EQRate_whichever_lower/EQRate_whichever_higher')
        if EQrate_mode == 'hypo':
            ax.set_title('p(%) of rejecting hypothesis (binomial): EQRate_S1 < EQRate_S2')
        elif EQrate_mode == 'ensemble':
            title_txt = 'p(%) of rejecting hypothesis '
            title_txt += '(%s-ensemble): EQRate_S1 < EQRate_S2'%num_ensemble
            ax.set_title(title_txt)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    if plot == 'sub':
        # 1
        ax = plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2])
        plt.imshow(EQS1_plot)
        max_EQS1 = np.nanmax(EQS1_plot)
        for iy in range(len(y_cuts)-1):
            for ix in range(len(x_cuts)-1):
                if ~np.isnan(EQS1_plot[iy, ix]):
                    if EQS1_plot[iy, ix] > max_EQS1*0.6:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.0f'%(EQS1_plot[iy, ix]), ha="center", 
                            va="center", color=c, fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 90)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0)
        ax.scatter(get_tick(stn_coord[1],xtick_labels)-0.5,get_tick(stn_coord[0],ytick_labels)-0.5, 
                   marker = '*', color='r', s = 240, label = stn_name)
        plt.legend()
        ax.set_title(plot_prefix[0]+' $N_{1}$, with |$T_{1}$| = %s'%len_S1)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        # 2
        ax = plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2]+1)
        plt.imshow(EQS2_plot)
        max_EQS2 = np.nanmax(EQS2_plot)
        for iy in range(len(y_cuts)-1):
            for ix in range(len(x_cuts)-1):
                if ~np.isnan(EQS2_plot[iy, ix]):
                    if EQS2_plot[iy, ix] > max_EQS2*0.6:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.0f'%(EQS2_plot[iy, ix]), ha="center", 
                            va="center", color=c, fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 90)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0)
        ax.scatter(get_tick(stn_coord[1],xtick_labels)-0.5,get_tick(stn_coord[0],ytick_labels)-0.5, 
                   marker = '*', color='r', s = 240, label = stn_name)
        plt.legend()
        ax.set_title(plot_prefix[1]+' $N_{2}$, with |$T_{2}$| = %s'%len_S2)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        # 3
        ax = plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2]+2)
        if (EQrate_mode == 'S1')|(EQrate_mode == 'hypo')|(EQrate_mode == 'ensemble'):
            plt.imshow(EQratios_plot, cmap="PiYG", vmin=0, vmax=1)
        elif EQrate_mode == 'abs':
            plt.imshow(EQratios_plot, cmap="Reds_r")
        for iy in range(len(y_cuts)-1):
            for ix in range(len(x_cuts)-1):
                if ~np.isnan(EQratios_plot[iy, ix]):
                    entry = EQratios_plot[iy, ix]*100
                    if (EQrate_mode == 'S1')|(EQrate_mode == 'hypo')|(EQrate_mode == 'ensemble'):
                        if (entry > 25) & (entry < 75):
                            c = 'k'
                        else:
                            c = 'w'
                    elif EQrate_mode == 'abs':
                        if entry > 50:
                            c = 'k'
                        else:
                            c = 'w'
                    ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                            fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 90)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0)
        ax.scatter(get_tick(stn_coord[1],xtick_labels)-0.5,get_tick(stn_coord[0],ytick_labels)-0.5, 
                   marker = '*', color='r', s = 240, label = stn_name)
        plt.legend()
        if EQrate_mode == 'S1':
            ax.set_title(plot_prefix[2]+' Heatmap (%) of $R_{f}$')
        if EQrate_mode == 'abs':
            ax.set_title(plot_prefix[2]+' EQratios (%) heatmap of EQRate_whichever_lower/EQRate_whichever_higher')
        if EQrate_mode == 'hypo':
            ax.set_title(plot_prefix[2]+' p(%) of rejecting hypothesis (binomial): EQRate_S1 < EQRate_S2')
        elif EQrate_mode =='ensemble':
            title_txt = plot_prefix[2]+ ' p(%) of rejecting hypothesis '
            title_txt += '(%s-ensemble): EQRate_S1 < EQRate_S2'%num_ensemble
            ax.set_title(title_txt)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    if return_info == 'no':
        return
    else:
        plot_info = {'EQS1_plot': EQS1_plot,
                     'EQS2_plot': EQS2_plot,
                     'EQratios_plot': EQratios_plot,
                     'len_S1': len_S1,
                     'len_S2': len_S2,
                     'xticks': xticks,
                     'yticks': yticks,
                     'xtick_labels': xtick_labels,
                     'ytick_labels': ytick_labels,
                     'stn_name': stn_name,
                     'stn_coord': stn_coord,
                     'x_cuts': x_cuts,
                     'y_cuts': y_cuts
                     }
        if fully_save == 'yes':
            plot_info['Gamma'] = info['Gamma']
            plot_info['t_segs'] = info['t_segs']
        return plot_info

def get_EQEs(state_threshold, GMselected, info):
    '''
    state_threshold is the 'tolerance' of state classification, 0.5 being most loose
    assuming S1 occupies 'len_S1' duration, while 'eqs_S1' number of EQ happened, 
    then 'eq_rate_S1' is the EQ rate for S1; same for S2
    function returns the EQ rate vector for both S1 and S2, 
    can also optionally return all len_Sn and eqs_Sn
    min_EQ is the minimum # of EQ to yield eq_rates
    '''
    Gamma = info['Gamma'].copy()
    t_segs = info['t_segs'].copy()
    Gamma[0,Gamma[0,:] < state_threshold] = 0
    Gamma[0,Gamma[0,:] > 1 - state_threshold] = 1
    Gamma[1,Gamma[1,:] < state_threshold] = 0
    Gamma[1,Gamma[1,:] > 1 - state_threshold] = 1
    len_S1 = np.sum(Gamma[0,:] == 1)
    len_S2 = np.sum(Gamma[1,:] == 1)
    t_eqs = GMselected[:,0]
    t_segs = info['t_segs']
    M_eqs = GMselected[:,1]
    E_eqs = (10**(11.8+1.5*M_eqs))/(10**15)
    if len(t_eqs) > 0:
        E_eqs = E_eqs[t_eqs < t_segs[-1]]
        t_eqs = t_eqs[t_eqs < t_segs[-1]]
        E_eqs = E_eqs[t_eqs > t_segs[0]]
        t_eqs = t_eqs[t_eqs > t_segs[0]]
    else:
        return np.array([]), np.array([]), len_S1, len_S2
    E_eqs_S1 = []
    E_eqs_S2 = []
    if len(t_eqs) >= 1:
        for i in range(len(t_eqs)):
            t_eqi = t_eqs[i]
            ind = np.argmin(np.abs(t_segs-t_eqi))
            if Gamma[0,ind] == 1:
                E_eqs_S1.append(E_eqs[i])
            elif Gamma[1,ind] == 1:
                E_eqs_S2.append(E_eqs[i])
    E_eqs_S1 = np.array(E_eqs_S1)
    E_eqs_S2 = np.array(E_eqs_S2)
    return E_eqs_S1, E_eqs_S2, len_S1, len_S2

def plot_avEQEGM(info, state_threshold = 0.5, div_y = 30, div_x = 0, plot = 'yes', 
                 sub_inds = [2,3,4], plot_prefix = ['', '', ''], NS_keep_eq = '', 
                 return_info = 'no', avEQE_minsum = 5):
    eqks = mB.eqk()
    if len(NS_keep_eq) == 1:
        eqks.NS_divide(keep_eq = NS_keep_eq)
    t_eqs = eqks.mat[:,0]
    Tsegs_keep, Teqs_keep = sync_TsegsTeqs(info['t_segs'], t_eqs, Tsegs_maxdiff = 1)
    eqks.lld = eqks.lld[Teqs_keep, :]
    eqks.mat = eqks.mat[Teqs_keep, :]
    eqks.makeGM(div_y = div_y, div_x = div_x)
    x_cuts = eqks.x_cuts
    y_cuts = eqks.y_cuts
    avEQEratios = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
    avEQE_S1s = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
    avEQE_S2s = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
    for ix in range(len(x_cuts)-1):
        for iy in range(len(y_cuts)-1):
            GMselected = eqks.GM_select(iy, ix)
            t_eqs = GMselected[:,0]
            if len(t_eqs) > 0:
                E_eqs_S1, E_eqs_S2, len_S1, len_S2 = get_EQEs(state_threshold, GMselected, info)
                avEQE_S1s[iy, ix] = np.sum(E_eqs_S1)/len_S1
                avEQE_S2s[iy, ix] = np.sum(E_eqs_S2)/len_S2
                ratio = avEQE_S1s[iy, ix]/(avEQE_S1s[iy, ix]+avEQE_S2s[iy, ix])
                if avEQE_S1s[iy, ix] + avEQE_S2s[iy, ix] < avEQE_minsum:
                    avEQEratios[iy, ix] = np.nan
                else:
                    avEQEratios[iy, ix] = ratio
            else:
                avEQEratios[iy, ix] = np.nan
                avEQE_S1s[iy, ix] = 0
                avEQE_S2s[iy, ix] = 0
    x_cuts = np.round(x_cuts*100)/100
    y_cuts = np.round(y_cuts*100)/100
#    if plot != 'no':
    EQratios_plot = np.flipud(avEQEratios)
    EQS1_plot = np.flipud(avEQE_S1s)
    EQS2_plot = np.flipud(avEQE_S2s)
    EQS1_plot[EQS1_plot == 0] = np.nan
    EQS2_plot[EQS2_plot == 0] = np.nan
    xticks = np.arange(0, len(x_cuts),1)
    xtick_labels = x_cuts
    yticks = np.arange(0, len(y_cuts),1)
    ytick_labels = np.flipud(y_cuts)
    stn_name = info['dataname'][0:4]
    stn_coord = eqks.position[stn_name]
    fs = 7
    if plot == 'yes':
        ax = plt.subplot() 
#        m = Basemap(projection='lcc', llcrnrlon = eqks.x_min, llcrnrlat = eqks.y_min, 
#                    urcrnrlon = eqks.x_max, urcrnrlat = eqks.y_max, resolution='h',
#                    lat_0 = (eqks.y_max + eqks.y_min)/2, lon_0=(eqks.x_min+eqks.x_max)/2)
        plt.imshow(EQratios_plot, cmap="PiYG")
        for iy in range(len(y_cuts)-1):
            for ix in range(len(x_cuts)-1):
                if ~np.isnan(EQratios_plot[iy, ix]):
                    entry = EQratios_plot[iy, ix]*100
                    if (entry > 30) & (entry < 70):
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.0f'%(entry), ha="center", va="center", color=c, 
                            fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 90)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0)
        ax.scatter(get_tick(stn_coord[1],xtick_labels), get_tick(stn_coord[0],ytick_labels), 
                   marker = '*', color='r', s = 240, label = stn_name)
        plt.legend()
        ax.set_title('avEQEratios (%) heatmap ')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    if plot == 'sub':
        # 1
        ax = plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2])
#        m = Basemap(projection='lcc', llcrnrlon = eqks.x_min, llcrnrlat = eqks.y_min, 
#                    urcrnrlon = eqks.x_max, urcrnrlat = eqks.y_max, resolution='h',
#                    lat_0 = (eqks.y_max + eqks.y_min)/2, lon_0=(eqks.x_min+eqks.x_max)/2)
        plt.imshow(EQS1_plot)
        max_EQS1 = np.nanmax(EQS1_plot)
        for iy in range(len(y_cuts)-1):
            for ix in range(len(x_cuts)-1):
                if ~np.isnan(EQS1_plot[iy, ix]):
                    if EQS1_plot[iy, ix] > max_EQS1*0.6:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.0f'%(EQS1_plot[iy, ix]), ha="center", 
                            va="center", color=c, fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 90)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0)
        ax.scatter(get_tick(stn_coord[1],xtick_labels),get_tick(stn_coord[0],ytick_labels), 
                   marker = '*', color='r', s = 240, label = stn_name)
        plt.legend()
        ax.set_title(plot_prefix[0]+' avEQE_S1s, with len_S1 = %s'%len_S1)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        # 2
        ax = plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2]+1)
        plt.imshow(EQS2_plot)
        max_EQS2 = np.nanmax(EQS2_plot)
        for iy in range(len(y_cuts)-1):
            for ix in range(len(x_cuts)-1):
                if ~np.isnan(EQS2_plot[iy, ix]):
                    if EQS2_plot[iy, ix] > max_EQS2*0.6:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.0f'%(EQS2_plot[iy, ix]), ha="center", 
                            va="center", color=c, fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 90)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0)
        ax.scatter(get_tick(stn_coord[1],xtick_labels),get_tick(stn_coord[0],ytick_labels), 
                   marker = '*', color='r', s = 240, label = stn_name)
        plt.legend()
        ax.set_title(plot_prefix[1]+' avEQE_S2s, with len_S2 = %s'%len_S2)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        # 3
        ax = plt.subplot(sub_inds[0], sub_inds[1], sub_inds[2]+2)
        plt.imshow(EQratios_plot, cmap="PiYG")
        for iy in range(len(y_cuts)-1):
            for ix in range(len(x_cuts)-1):
                if ~np.isnan(EQratios_plot[iy, ix]):
                    entry = EQratios_plot[iy, ix]*100
                    if (entry > 30) & (entry < 70):
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.0f'%(entry), ha="center", va="center", color=c, 
                            fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 90),
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0)
        ax.scatter(get_tick(stn_coord[1],xtick_labels),get_tick(stn_coord[0],ytick_labels), 
                   marker = '*', color='r', s = 240, label = stn_name)
        plt.legend()
        ax.set_title(plot_prefix[2]+' avEQEratios (pct), avEQE_minsum = %s'%avEQE_minsum)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    if return_info == 'no':
        return
    else:
        plot_info = {'EQS1_plot': EQS1_plot,
                     'EQS2_plot': EQS2_plot,
                     'EQratios_plot': EQratios_plot,
                     'len_S1': len_S1,
                     'len_S2': len_S2,
                     'xticks': xticks,
                     'yticks': yticks,
                     'xtick_labels': xtick_labels,
                     'ytick_labels': ytick_labels,
                     'stn_name': stn_name,
                     'stn_coord': stn_coord}
        return plot_info

def best_n_heatmap_GM(file_name, n_configs, nlabels_pick = 0, note = '', save = 'yes',
                      div_y = 30, HS_rads = [], HS_mags = [], shift = 0):
    # file_name e.g. 'BM_CHMMMI_DABA_ti2_FCW500S2_CVSK200_CVSK_R100L90_0416G.pickle'
    with open(file_name, 'rb') as f:
        dic_config = pickle.load(f)
    best_n_configs = {}
    for config in dic_config:
        info_dic = dic_config[config]
        if nlabels_pick > 0:
            for info_name in info_dic:
                info = info_dic[info_name]
                break
            if info['n_labels'] != nlabels_pick:
                continue
        model_scores_all = list()
        for info_name in info_dic:
            model_scores_all.append(info_dic[info_name]['model_scores'][-1])
        sorted_scores = np.sort(np.array(model_scores_all))[::-1]
        score_threshold = sorted_scores[n_configs-1]
        for info_name in info_dic:
            if info_dic[info_name]['model_scores'][-1] >= score_threshold:
                best_n_configs[info_name] = info_dic[info_name]
    info_dic = best_n_configs
    for key in info_dic:
        info = info_dic[key]
        if len(HS_rads) == 0:
            HS_rads = info['HS_rads']
            HS_mags = info['HS_mags']
        else:
            info = standardize_HS(info, rads = HS_rads, mags = HS_mags, shift = shift)
        fig = plt.figure(figsize=(28,14))
        if np.isnan(info['model_scores'][-1]):
            end_score = 0
        else:
            end_score = np.int(info['model_scores'][-1])
        figure_name = 'EQGM_'+file_name[0:-7] + note + '_' + key + '_score%s'%(end_score)
        plt.suptitle(figure_name)
        plt.subplot(2,3,1)
        plt.title('(a) Log model scores vs. iteration')
        plt.plot(info['model_scores'], 'o')
        plt.subplot(2,3,2)
        plt.title('(b) Indicator distribution (Emission) for two states')
        plt.plot(info['B'][0,:], label = 'S1')
        plt.plot(info['B'][1,:], label = 'S2')
        plt.legend()
        plt.xlabel('ID of B')
        plt.subplot(2,3,3)
        plt.plot(info['t_segs'], info['Gamma'][0,:])
        plt.title('(c) Posterior P(S1)')
        plt.xlabel("Time")
        if ~np.isnan(info['Gamma'][0,0]):
            plot_prefix = ['(d)', '(e)', '(f)']
            plot_EQGM(info, state_threshold = 0.5, div_y = div_y, plot = 'sub', 
                      sub_inds = [2,3,4], plot_prefix = plot_prefix, shift = shift)
        else:
            print('Nan Gamma found, skipped heatmap')
        if save == 'yes':
            plt.savefig(figure_name +'.png', dpi = 500)
            plt.close(fig)
    return

def HSVA_getinfo_optiR(file_name, sum_info):
    # file_name = 'CHMMMI_ENAN_ti2_FCW500S2_BW0813_RWday0.5_05JAN_trV_CVSK_R30L80_0201NA.pickle'
    stn_name = file_name[7:11]
    stn_info = sum_info[stn_name]
    LocRads = np.arange(0.8, 3.0, 0.1)
    LocRads = np.round(LocRads*10)/10
    ind_rad = np.argmax(stn_info['rate_mean_prod_tops_stn'])
    opti_rad = LocRads[ind_rad]
    info = HSVA_getinfo_0125(file_name, sort_HS = 'yes')
    eqks = mB.eqk()
    t_eqs = eqks.mat[:,0].copy()
    eqks.lld = eqks.lld[t_eqs <= info['t_segs'][-1], :]
    eqks.mat = eqks.mat[t_eqs <= info['t_segs'][-1], :]
    t_eqs = eqks.mat[:,0].copy()
    eqks.lld = eqks.lld[t_eqs >= info['t_segs'][0], :]
    eqks.mat = eqks.mat[t_eqs >= info['t_segs'][0], :]
    mag = 3
    eqks.select(stn_name, mag, radius = opti_rad, verbose = 'no')
    # lld = eqks.selected_lld
    # plt.scatter(lld[:,0], lld[:,1])
    t_eqs = eqks.selected[:,0]
    eq_rates, eqs_S1, len_S1, eqs_S2, len_S2 = compute_EQrates(0.5, t_eqs, info, 
                                                               return_all = 'yes', 
                                                               min_EQ = 1, shift = 0)
    # define S1 as the active state
    info_new = info.copy()
    info_new['HS_mags'] = mag
    info_new['HS_rads'] = opti_rad
    new_file_name = file_name[:-7] + '_optiR%s.pickle'%opti_rad
    info_new['file_name'] = new_file_name
    if eq_rates[0] < eq_rates[1]:
        info_new['Gamma'] = 1 - info['Gamma']
        info_new['pS1s'] = 1 - info['pS1s']
        info_new['pS1s_voted'] = 1 - info['pS1s_voted']
        info_new['pS2s_voted'] = 1 - info['pS2s_voted']
    return info_new

def EQGM_5plot(file_name, div_y = 30, save = 'yes', NS_keep_eq = '', mode = 'EQGM',
               avEQE_minsum = 5, min_EQ = 5, mag_LB = 0, EQrate_mode = 'S1', shift = 0, 
               num_ensemble = 200, save_info = 'no', save_dir = '', plot = 'sub',
               version = 'old', sum_info = '(from sum_name, if use version: optiR)',
               max_models = 30):
    # file_name like 'CHMMMI_SIHU_ti2_FCW500S2_CVSK200_cV_4YRs_CVSK_R100L90_0430Ar.pickle'
    # mode can be 'EQGM' or 'avEQEGM'
    # sum_name = 'OptimalLocRadsDicts_div16_CVSK_UFD_trV_minEQ1_mag3_RWday0.5_05JAN_trV_CVSK_
    # R30L80_0201NA.pickle'
    if version == '0125':
        info = HSVA_getinfo_0125(file_name, sort_HS = 'yes', max_models = max_models)
    elif version == 'optiR':
        info = HSVA_getinfo_optiR(file_name, sum_info)
    else:
        info = HSVA_getinfo(file_name)
    figure_name = mode+'_'+EQrate_mode+'_'+info['file_name'][0:-7]
    if shift != 0:
        figure_name += '_shift%s'%shift
    if plot == 'sub':
        plt.figure(figsize=(22,14))
        plt.suptitle(figure_name)
        grid = plt.GridSpec(2, 22)
        plt.subplot(grid[0, 0:12])
        time_axis = pdate.num2date(info['t_segs'])
        time_axis_short = time_axis.copy()
        for i in range(len(time_axis)):
            time_axis_short[i] = time_axis_short[i].strftime("%b %Y")
        # df = pd.DataFrame(info['pS1s'], columns=time_axis_short)
        sns.heatmap(info['pS1s'])
        xtick_points = np.int64(np.linspace(0,len(time_axis), 10))
        xtick_points = xtick_points[1:-1]
        xtick_pt_labels = []
        for i in range(len(xtick_points)):
            xtick_pt_labels.append(time_axis_short[xtick_points[i]])
        plt.xticks(xtick_points, xtick_pt_labels, rotation='horizontal')
        # sns.heatmap(info['pS1s'], xticklabels=pdate.num2date(info['t_segs']))
        plt.xlabel('Time')
        plt.ylabel('ID of HMM models')
        plt.title('(a) All Posterior P(S1) TSs') 
        plt.subplot(grid[0, 12:])
        # plt.plot(time_axis, info['Gamma'][0,:])
        plt.plot(info['Gamma'][0,:], linewidth=0.3)
        plt.xlim([0, len(info['Gamma'][0,:])])
        # sns.heatmap(info['pS1s'][0:1,:])
        plt.xticks(xtick_points, xtick_pt_labels, rotation='horizontal')
        plt.title('(b) Best Model Posterior P(S1)')
        plt.xlabel("Time")
    if ~np.isnan(info['Gamma'][0,0]):
        # plot_prefix = ['(c)', '(d)', '(e)']
        plot_prefix = ['(a)', '(b)', '(c)']
        if mode == 'EQGM':
            if save_info == 'no':
                plot_EQGM(info, div_y = div_y, plot = plot, sub_inds = [2,3,4], 
                          plot_prefix = plot_prefix, NS_keep_eq = NS_keep_eq, min_EQ = min_EQ, 
                          mag_LB = mag_LB, EQrate_mode = EQrate_mode, shift = shift, 
                          num_ensemble = num_ensemble, return_info = 'no')
            else:
                plot_info = plot_EQGM(info, div_y = div_y, plot = plot, sub_inds = [2,3,4], 
                                      plot_prefix = plot_prefix, NS_keep_eq = NS_keep_eq, 
                                      min_EQ = min_EQ, mag_LB = mag_LB, EQrate_mode = EQrate_mode, 
                                      num_ensemble = num_ensemble, return_info = 'yes', 
                                      shift = shift)
        elif mode == 'avEQEGM':
            plot_avEQEGM(info, state_threshold = 0.5, div_y = div_y, plot = plot, 
                         sub_inds = [2,3,4], plot_prefix = plot_prefix, 
                         NS_keep_eq = NS_keep_eq, avEQE_minsum = avEQE_minsum)
    else:
        print('Nan Gamma found, skipped heatmap')
    if len(NS_keep_eq) == 1:
            NS_keep_eq = '_' + NS_keep_eq
    if mode == 'EQGM':
        txt = '_minEQ%s'%min_EQ
    elif mode == 'avEQEGM':
        txt = '_avEQE_minsum%s'%avEQE_minsum
    if mag_LB > 0:
        txt += '_magLB%s'%mag_LB
    txt += '_div%s'%div_y
    if save == 'yes':
        plt.savefig(save_dir + figure_name + NS_keep_eq + txt + '.png', dpi = 500)
        plt.close('all')
    if save_info == 'yes':
        with open(save_dir + figure_name + NS_keep_eq + txt +'_info.pickle', 'wb') as f:
            pickle.dump(plot_info, f)
    return

def EQGM_5plot_pubF4(file_name, div_y = 30, save = 'yes', NS_keep_eq = '', mode = 'EQGM',
                     avEQE_minsum = 5, min_EQ = 5, mag_LB = 0, EQrate_mode = 'S1', shift = 0, 
                     num_ensemble = 200, save_info = 'no', save_dir = '', plot = 'sub',
                     version = 'old', sum_info = '(from sum_name, if use version: optiR)',
                     max_models = 30):
    # file_name like 'CHMMMI_SIHU_ti2_FCW500S2_CVSK200_cV_4YRs_CVSK_R100L90_0430Ar.pickle'
    # mode can be 'EQGM' or 'avEQEGM'
    # sum_name = 'OptimalLocRadsDicts_div16_CVSK_UFD_trV_minEQ1_mag3_RWday0.5_05JAN_trV_CVSK_
    # R30L80_0201NA.pickle'
    if version == '0125':
        info = HSVA_getinfo_0125(file_name, sort_HS = 'yes', max_models = max_models)
    elif version == 'optiR':
        info = HSVA_getinfo_optiR(file_name, sum_info)
    else:
        info = HSVA_getinfo(file_name)
    figure_name = mode+'_'+EQrate_mode+'_'+info['file_name'][0:-7]
    if shift != 0:
        figure_name += '_shift%s'%shift
    if plot == 'sub':
        plt.figure(figsize=(20,12))
        # plt.suptitle(figure_name)
        grid = plt.GridSpec(2, 22)
        plt.subplot(grid[0, 0:12])
        time_axis = pdate.num2date(info['t_segs'])
        time_axis_short = time_axis.copy()
        for i in range(len(time_axis)):
            time_axis_short[i] = time_axis_short[i].strftime("%b %Y")
        # df = pd.DataFrame(info['pS1s'], columns=time_axis_short)
        sns.heatmap(info['pS1s'])
        xtick_points = np.int64(np.linspace(0,len(time_axis), 10))
        xtick_points = xtick_points[1:-1]
        xtick_pt_labels = []
        for i in range(len(xtick_points)):
            xtick_pt_labels.append(time_axis_short[xtick_points[i]])
        plt.xticks(xtick_points, xtick_pt_labels, rotation='horizontal')
        # sns.heatmap(info['pS1s'], xticklabels=pdate.num2date(info['t_segs']))
        plt.xlabel('Time')
        plt.ylabel('ID of HMM models')
        plt.title('(a) All Posterior P(S1) TSs') 
        plt.subplot(grid[0, 12:])
        # plt.plot(time_axis, info['Gamma'][0,:])
        plt.plot(info['Gamma'][0,:], linewidth=0.3)
        plt.xlim([0, len(info['Gamma'][0,:])])
        # sns.heatmap(info['pS1s'][0:1,:])
        plt.xticks(xtick_points, xtick_pt_labels, rotation='horizontal')
        plt.title('(b) Best Model Posterior P(S1)')
        plt.xlabel("Time")
    
    if len(NS_keep_eq) == 1:
            NS_keep_eq = '_' + NS_keep_eq
    if mode == 'EQGM':
        txt = '_minEQ%s'%min_EQ
    elif mode == 'avEQEGM':
        txt = '_avEQE_minsum%s'%avEQE_minsum
    if mag_LB > 0:
        txt += '_magLB%s'%mag_LB
    txt += '_div%s'%div_y
    if save == 'yes':
        plt.savefig(save_dir + figure_name + NS_keep_eq + txt + '.tiff', dpi = 300, 
                    bbox_inches='tight')
        plt.close('all')
    return

def EQGM_5plot_pubF7(file_name, div_y = 30, save = 'yes', NS_keep_eq = '', mode = 'EQGM',
                     avEQE_minsum = 5, min_EQ = 5, mag_LB = 0, EQrate_mode = 'S1', shift = 0, 
                     num_ensemble = 200, save_info = 'no', save_dir = '', plot = 'sub',
                     version = 'old', sum_info = '(from sum_name, if use version: optiR)',
                     max_models = 30):
    # file_name like 'CHMMMI_SIHU_ti2_FCW500S2_CVSK200_cV_4YRs_CVSK_R100L90_0430Ar.pickle'
    # mode can be 'EQGM' or 'avEQEGM'
    # sum_name = 'OptimalLocRadsDicts_div16_CVSK_UFD_trV_minEQ1_mag3_RWday0.5_05JAN_trV_CVSK_
    # R30L80_0201NA.pickle'
    if version == '0125':
        info = HSVA_getinfo_0125(file_name, sort_HS = 'yes', max_models = max_models)
    elif version == 'optiR':
        info = HSVA_getinfo_optiR(file_name, sum_info)
    else:
        info = HSVA_getinfo(file_name)
    figure_name = mode+'_'+EQrate_mode+'_'+info['file_name'][0:-7]
    if shift != 0:
        figure_name += '_shift%s'%shift
    if plot == 'sub':
        plt.figure(figsize=(20,12))
        # plt.suptitle(figure_name)
        # grid = plt.GridSpec(2, 22)
        # plt.subplot(grid[0, 0:12])
        # time_axis = pdate.num2date(info['t_segs'])
        # time_axis_short = time_axis.copy()
        # for i in range(len(time_axis)):
        #     time_axis_short[i] = time_axis_short[i].strftime("%b %Y")
        # # df = pd.DataFrame(info['pS1s'], columns=time_axis_short)
        # sns.heatmap(info['pS1s'])
        # xtick_points = np.int64(np.linspace(0,len(time_axis), 10))
        # xtick_points = xtick_points[1:-1]
        # xtick_pt_labels = []
        # for i in range(len(xtick_points)):
        #     xtick_pt_labels.append(time_axis_short[xtick_points[i]])
        # plt.xticks(xtick_points, xtick_pt_labels, rotation='horizontal')
        # # sns.heatmap(info['pS1s'], xticklabels=pdate.num2date(info['t_segs']))
        # plt.xlabel('Time')
        # plt.ylabel('ID of HMM models')
        # plt.title('(a) All Posterior P(S1) TSs') 
        # plt.subplot(grid[0, 12:])
        # # plt.plot(time_axis, info['Gamma'][0,:])
        # plt.plot(info['Gamma'][0,:], linewidth=0.3)
        # plt.xlim([0, len(info['Gamma'][0,:])])
        # # sns.heatmap(info['pS1s'][0:1,:])
        # plt.xticks(xtick_points, xtick_pt_labels, rotation='horizontal')
        # plt.title('(b) Best Model Posterior P(S1)')
        # plt.xlabel("Time")
    if ~np.isnan(info['Gamma'][0,0]):
        # plot_prefix = ['(c)', '(d)', '(e)']
        plot_prefix = ['(a)', '(b)', '(c)']
        if mode == 'EQGM':
            if save_info == 'no':
                plot_EQGM(info, div_y = div_y, plot = plot, sub_inds = [2,3,4], 
                          plot_prefix = plot_prefix, NS_keep_eq = NS_keep_eq, min_EQ = min_EQ, 
                          mag_LB = mag_LB, EQrate_mode = EQrate_mode, shift = shift, 
                          num_ensemble = num_ensemble, return_info = 'no')
    else:
        print('Nan Gamma found, skipped heatmap')
    if len(NS_keep_eq) == 1:
            NS_keep_eq = '_' + NS_keep_eq
    if mode == 'EQGM':
        txt = '_minEQ%s'%min_EQ
    elif mode == 'avEQEGM':
        txt = '_avEQE_minsum%s'%avEQE_minsum
    if mag_LB > 0:
        txt += '_magLB%s'%mag_LB
    txt += '_div%s'%div_y
    if save == 'yes':
        plt.savefig(save_dir + figure_name + NS_keep_eq + txt + '_F7.tiff', dpi = 300, 
                    bbox_inches='tight')
        plt.close('all')
    return

def voted_distplot(file_name, save = 'yes'):
    fig = plt.figure(figsize=(22,14))
    info = HSVA_getinfo(file_name)
    max_plot = 16
    fig_name = 'Voted_Indicator_Distplot_'+file_name[0:-7]
    plt.suptitle(fig_name, fontsize = 16)
    for i_input in range(max_plot): 
        plt.subplot(4,4,i_input+1)
        indi_S1 = info['segindic_TSs'][i_input, info['Gamma'][0,:] > 0.6]
        indi_S2 = info['segindic_TSs'][i_input, info['Gamma'][1,:] > 0.6]
        sns.distplot(cut_ends(indi_S1), label = 'S1')
        sns.distplot(cut_ends(indi_S2), label = 'S2')
        plt.title(info['input_names'][i_input])
        plt.legend()
    if save == 'yes':
        plt.savefig(fig_name +'.png', dpi = 500)
        plt.close(fig)
    return

reHS_rads = {'SIHU': np.array([0, 2]), 
             'PULI': np.array([0, 0.7]), 
             'DABA': np.array([0, 2]), 
             'CHCH': np.array([0, 2]), 
             'LIOQ': np.array([0, 2]), 
             'YULI': np.array([0, 2]), 
             'HERM': np.array([0, 2]), 
             'FENL': np.array([0, 1.6])}