#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:15:49 2019

module primarily for EP_pred_models.py

@author: haoyuwen
"""

import numpy as np
import matplotlib.pyplot as plt
import EP_modB_new as mB
import seaborn as sns

# In[]
def EWI_score(x, para_k, b):
    '''
    para_k shoud be in [1, 100]
    b should be positive, and is the symmetric point of the curve
    x is 1-d array, k, b are scalars
    score = a/(1+exp^k(x-b)) + c for x < 2b, shifted sigmoid score function 
    s.t. score(0) = 0, score(b) = 1/2, score(2b) = 1
    score(x > 2b) = 1
    '''
    k = -1/para_k
    kb = k*b
    a = (np.exp(kb) + np.exp(-kb) + 2)/(np.exp(-kb) - np.exp(kb))
    c = 0.5 - a/2
    score = a/(1+np.exp(k*(x-b))) + c
    score[x>2*b] = 1
    return score

def EWI_weight(x, para_k, b):
    '''
    para_k shoud be in [1, 100]
    x is 1-d array, k, b are scalars
    score = a/(1+exp^k(x-b)) + c for x < 2b, shifted sigmoid score function 
    s.t. score(0) = 1, score(b) = 1/2, score(2b) = 0
    score(x > 2b) = 0
    '''
    k = -1/para_k
    kb = k*b
    a = -(np.exp(kb) + np.exp(-kb) + 2)/(np.exp(-kb) - np.exp(kb))
    c = 0.5 - a/2
    score = a/(1+np.exp(k*(x-b))) + c
    score[x>2*b] = 0
    return score

# In[]
    
def compute_EWIs(indiTS, t_ind, EWI_WL, EWI_Rstep, EWI_paras):
    '''
    EWI_paras = [score_para_k, score_b, weight_para_k]
    score_para_k should be > 0, otherwise, will not apply score function
    '''
    inds = np.array(range(0,len(indiTS)))
    inds = inds[EWI_WL:len(inds):EWI_Rstep]
    EWIs = np.zeros((len(inds), ))
    if EWI_paras[0] == 0:
        scores = indiTS
    else:
        scores = EWI_score(indiTS, EWI_paras[0], EWI_paras[1])
    weight_inds = np.arange(EWI_WL)
    for i in range(len(inds)):
        ind = inds[i]
        RW = scores[ind-EWI_WL:ind]
        EWIs[i] = np.sum(RW * EWI_weight(weight_inds, EWI_paras[2], EWI_WL/2))
    EWIs = EWIs/EWI_WL
    t_EWIs = t_ind[inds]
    return EWIs, t_EWIs

def decide_eq_in_tEWI(t_EWIs, t_eqs):
    eq_in_tEWI = np.zeros((len(t_EWIs), ), dtype = bool)
    t_eqs_pointer = 0
    t_eqs_examine = t_eqs[t_eqs > t_EWIs[0]].copy()
    t_EWIs_pointer = 1
    while t_eqs_pointer < len(t_eqs_examine):
        t_eq = t_eqs_examine[t_eqs_pointer]
        while t_EWIs[t_EWIs_pointer] < t_eq and t_EWIs_pointer < len(t_EWIs)-1:
            t_EWIs_pointer += 1
        eq_in_tEWI[t_EWIs_pointer-1] = 1
        t_eqs_pointer += 1
    return eq_in_tEWI

def compute_disc_ratio(EWIs, eq_in_tEWI, verbose = 'no'):
    '''
    above 0.5: good
    '''
    EWI_mid = np.nanmedian(EWIs)
    EWIs_above_mid_eq = len(EWIs[np.logical_and(EWIs>EWI_mid,eq_in_tEWI)])
    EWIs_below_mid_eq = len(EWIs[np.logical_and(EWIs<EWI_mid,eq_in_tEWI)])
    EWIs_above_mid_noeq = len(EWIs[np.logical_and(EWIs>EWI_mid,~eq_in_tEWI)])
    EWIs_below_mid_noeq = len(EWIs[np.logical_and(EWIs<EWI_mid,~eq_in_tEWI)])
    if verbose == 'yes':
        print('     | eq | noeq')
        print('>mid | %s | %s'%(EWIs_above_mid_eq, EWIs_above_mid_noeq))
        print('<mid | %s | %s'%(EWIs_below_mid_eq, EWIs_below_mid_noeq))
    if (EWIs_below_mid_eq + EWIs_above_mid_eq) == 0:
        ratio = np.nan
    else:
        ratio = EWIs_above_mid_eq/(EWIs_below_mid_eq + EWIs_above_mid_eq)
    return ratio

def plot_EWI_heatmap(data, col, indiname, ind_WL, EWI_WL, EWI_Rstep, 
                     EWI_paras, save = 'no'):
    RW_len = '%s'%data[col]['paras']['RW_lens'][ind_WL]
    dataname = data[col]['dataname']
    dataname = dataname[2:len(dataname)]
    data_fullname = dataname+'_'+indiname+'_RWlen'+RW_len
    indi_TS = data[col][indiname][ind_WL, :]
    t_ind = data[col]['t_ind']
    t_ind = t_ind[~np.isnan(indi_TS)]
    indi_TS = indi_TS[~np.isnan(indi_TS)]
    EWIs, t_EWIs = compute_EWIs(indi_TS, t_ind, EWI_WL, EWI_Rstep, EWI_paras)
    eqks = mB.eqk()
    mags = np.arange(4., 6.01, 0.1)
    radiuses = np.arange(0.4, 3.6, 0.1)
    mags = np.round(mags*10)/10
    radiuses = np.round(radiuses*10)/10
    ratios = np.zeros((len(mags), len(radiuses)))
    for i_mag in range(len(mags)):
        for i_radius in range(len(radiuses)):
            eqks.select(dataname, mag=mags[i_mag], radius = radiuses[i_radius])
            t_eqs = eqks.selected[:,0]
            eq_in_tEWI = decide_eq_in_tEWI(t_EWIs, t_eqs)
            ratios[i_mag, i_radius] = compute_disc_ratio(EWIs, eq_in_tEWI, 
                  verbose = 'yes')
    fig = plt.figure(figsize=(16,9))
    sns.heatmap(ratios, yticklabels = mags, xticklabels = radiuses, annot=True, 
                cmap="YlGnBu", cbar=False)
    plt.xlabel('Radius (degrees)')
    plt.ylabel('Magnitude Lower Bound')
    plt.title(data_fullname+'EWI WL=%s, Rstep=%s, paras=%s, %s, %s'%(EWI_WL,
                    EWI_Rstep,EWI_paras[0],EWI_paras[1],EWI_paras[2]))
    if save == 'yes':
        fig_name = data_fullname+'_WL%s_Rstep%s_paras%s_%s_%s.png'%(EWI_WL,
                    EWI_Rstep,EWI_paras[0],EWI_paras[1],EWI_paras[2])
        plt.savefig(fig_name, dpi = 300)
        plt.close(fig)

def plot_eqnum_map(dataname, save = 'no'):
    mags = np.arange(4., 6.01, 0.1)
    radiuses = np.arange(0.4, 3.6, 0.1)
    mags = np.round(mags*10)/10
    radiuses = np.round(radiuses*10)/10
    eqnums = np.zeros((len(mags), len(radiuses)))
    eqks = mB.eqk()
    for i_mag in range(len(mags)):
        for i_radius in range(len(radiuses)):
            eqks.select(dataname, mag=mags[i_mag], radius = radiuses[i_radius])
            t_eqs = eqks.selected[:,0]
            eqnums[i_mag, i_radius] = len(t_eqs)
    fig = plt.figure(figsize=(16,9))
    sns.heatmap(eqnums, yticklabels = mags, xticklabels = radiuses, cmap="YlGnBu", 
                annot=True, cbar=False)
    plt.xlabel('Radius (degrees)')
    plt.ylabel('Magnitude Lower Bound')
    plt.title('EQ numbers of '+dataname)
    if save == 'yes':
        fig_name = dataname+'_EQ_nums.png'
        plt.savefig(fig_name, dpi = 300)
        plt.close(fig)
        
# In[]
def gen_para(para_len, lb, ub, mean, std):
    '''
    function generate random gaussian variable of size (para_len, )
    lb, ub are hard cut-off bounds
    '''
    gen_len = para_len*5
    para = np.random.normal(mean, std, gen_len)
    para = para[para > lb]
    para = para[para < ub]
    if len(para) > para_len:
        para = para[0:para_len]
    else:
        print('Oops!')
        gen_len = para_len*20
        para = np.random.normal(mean, std, gen_len)
        para = para[para > lb]
        para = para[para < ub]
        para = para[0:para_len]
    return para