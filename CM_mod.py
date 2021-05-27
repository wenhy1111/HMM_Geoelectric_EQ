#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:55:37 2020

mod include functions mainly used in EP_animate_dev3.py and EP_NS_Scramble.py
"""

import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import animation
import EP_modB_new as mB
import BM_mod as bm
import matplotlib.dates as pdate
from sklearn.tree import DecisionTreeRegressor
import graphviz 
from sklearn import tree
import pickle
import seaborn as sns
from scipy import stats
import os
import datetime
# In[]
# datanames = ['D_FENG_ti2_FCW500S2',
#              'D_KUOL_ti2_FCW500S2',
#              'D_TOCH_ti2_FCW500S2',
#              'D_LIOQ_ti2_FCW500S2',
#              'D_YULI_ti2_FCW500S2',
#              'D_SIHU_ti2_FCW500S2',
#              'D_PULI_ti2_FCW500S2',
#              'D_HERM_ti2_FCW500S2',
#              'D_FENL_ti2_FCW500S2',
#              'D_CHCH_ti2_FCW500S2',
#              'D_WANL_ti2_FCW500S2',
#              'D_SHCH_ti2_FCW500S2',
#              'D_LISH_ti2_FCW500S2',
#              'D_KAOH_ti2_FCW500S2',
#              'D_HUAL_ti2_FCW500S2',
#              'D_ENAN_ti2_FCW500S2',
#              'D_DAHU_ti2_FCW500S2',
#              'D_DABAXti2_FCW500S2',
#              'D_RUEY_ti2_FCW500S2',
#              'D_SHRL_ti2_FCW500S2']
# In[]
def make_pS1mat(file_names, day_divider = 2, stn_NS = '', eq_NS = ''):
    t_segss = list()
    pS1s_voteds = list()
    stations = list()
    locs = list()
    eqks = mB.eqk() #???
    if len(eq_NS) == 1:
        eqks.NS_divide(eq_NS)
    if len(stn_NS) == 1:
        new_file_names = []
        for file_name in file_names:
            stn = file_name[7:11]
            if stn in eqks.north_stns.keys():
                if stn_NS == 'N':
                    new_file_names.append(file_name)
            if stn in eqks.south_stns.keys():
                if stn_NS == 'S':
                    new_file_names.append(file_name)
        file_names = new_file_names
    for file_name in file_names:
        info = bm.HSVA_getinfo(file_name)
        t_segss.append(info['t_segs'])
        pS1s_voteds.append(info['pS1s_voted'])
        stations.append(file_name[7:11])
        locs.append(eqks.position[file_name[7:11]])
    # graph visualization before animation
    t_segss_day = list()
    pS1s_voteds_day = list()
    i_stn = 0
    t_seg_min = 1000000
    t_seg_max = 0
    for i_stn in range(len(t_segss)):
        t_segs = t_segss[i_stn]
        t_segs = t_segs*day_divider
        pS1s_voted = pS1s_voteds[i_stn]
        t_segs_day = np.zeros((len(t_segs), ))
        pS1s_voted_day = np.zeros((len(t_segs), ))
        current_day = np.floor(t_segs[0])
        current_day_recorded = 0
        for i in range(len(t_segs)):
            if np.floor(t_segs[i]) == current_day:
                if current_day_recorded == 0:
                    t_segs_day[i] = current_day
                    pS1s_voted_day[i] = pS1s_voted[i]
                    current_day_recorded = 1
            else:
                current_day = np.floor(t_segs[i])
                t_segs_day[i] = current_day
                pS1s_voted_day[i] = pS1s_voted[i]
        pS1s_voted_day = pS1s_voted_day[t_segs_day != 0]
        t_segs_day = t_segs_day[t_segs_day != 0]
        t_segs_day = t_segs_day/day_divider
        t_segss_day.append(t_segs_day)
        pS1s_voteds_day.append(pS1s_voted_day)
        t_seg_min = np.min([np.min(t_segs_day), t_seg_min])
        t_seg_max = np.max([np.max(t_segs_day), t_seg_max])
    #
    duration = np.int(t_seg_max - t_seg_min + 1)*day_divider
    indss_day = list()
    for i_stn in range(len(t_segss)):
        indss_day.append(np.int64((t_segss_day[i_stn] - t_seg_min)*day_divider))
    # 
    pS1s_mat = np.ones((len(t_segss), duration))*(np.nan)
    for i_stn in range(len(t_segss)):
        pS1s_mat[i_stn, indss_day[i_stn]] = pS1s_voteds_day[i_stn]
    #
    t_mat = np.arange(t_seg_min, t_seg_max+1/day_divider, 1/day_divider)
    if len(t_mat) < pS1s_mat.shape[1]:
        t_mat = np.arange(t_seg_min, t_seg_max+2/day_divider, 1/day_divider)
#    t_dates = pdate.num2date(t_segs_day)
    return pS1s_mat, t_mat, locs, stations


# In
def get_local_eq(t_mat, div_y = 20, iy = 12, ix = 11, plot_directory = 'no', 
                 plot_local = 'no', keep_eq_NS = ''):
    eqks = mB.eqk()
    if len(keep_eq_NS) == 1:
        eqks.NS_divide(keep_eq_NS)
    t_eqs = eqks.mat[:,0]
    Tsegs_keep, Teqs_keep = bm.sync_TsegsTeqs(t_mat, t_eqs, Tsegs_maxdiff = 1)
    eqks.lld = eqks.lld[Teqs_keep, :]
    eqks.mat = eqks.mat[Teqs_keep, :]
    eqks.makeGM(div_y = div_y, plot = plot_directory)
    GMselected = eqks.GM_select(iy, ix, plot = plot_local)
    #
    have_eq = np.zeros((len(t_mat)))
    for i in range(GMselected.shape[0]):
        t_eqi = GMselected[i,0]
        ind = np.argmin(np.abs(t_mat-t_eqi))
        have_eq[ind] += 1
    return have_eq


def get_CLEE(pS1s_mat, t_mat, div_y = 20, iy = 12, ix = 11, plot_directory = 'no', 
             plot_local = 'no'):
    # make HS vector codebook
    pS1s_codenums = np.ones((pS1s_mat.shape[1]))*(np.nan)
    pS1s_codes = list()
    for j in range(pS1s_mat.shape[1]):
        HS_vec = pS1s_mat[:,j]
        if np.sum(np.isnan(HS_vec)) > 1: # if only 1 missing, treat it as 0 instead of nan
            pS1s_codes.append('')
            continue
        else:
            code = ''
            for i in range(len(HS_vec)):
                if HS_vec[i] > 0.5:
                    code = code + '1'
                else:
                    if i > 0:
                        code = code + '0'
            pS1s_codenums[j] = int(code, 2)
            pS1s_codes.append(code)
    #
    eqks = mB.eqk()
    t_eqs = eqks.mat[:,0]
    Tsegs_keep, Teqs_keep = bm.sync_TsegsTeqs(t_mat, t_eqs, Tsegs_maxdiff = 1)
    eqks.lld = eqks.lld[Teqs_keep, :]
    eqks.mat = eqks.mat[Teqs_keep, :]
    eqks.makeGM(div_y = div_y, plot = plot_directory)
    GMselected = eqks.GM_select(iy, ix, plot = plot_local)
    #
    codenums_eqs = np.zeros((GMselected.shape[0], ))
    pS1s_eqs = np.zeros((pS1s_mat.shape[0], GMselected.shape[0]))
    have_eq = np.zeros((pS1s_mat.shape[1]))
    for i in range(GMselected.shape[0]):
        t_eqi = GMselected[i,0]
        ind = np.argmin(np.abs(t_mat-t_eqi))
        have_eq[ind] += 1
        pS1s_eqs[:,i] = pS1s_mat[:, ind]
        codenums_eqs[i] = pS1s_codenums[ind]
    CLEE_info = np.zeros((2**pS1s_mat.shape[0], 4)) #codenum_len_eq_eqrate
    CLEE_info[:,0] = np.arange(2**pS1s_mat.shape[0])
    for i in range(len(pS1s_codenums)):
        if ~np.isnan(pS1s_codenums[i]):
            CLEE_info[int(pS1s_codenums[i]), 1] += 1
    for i in range(len(codenums_eqs)):
        if ~np.isnan(codenums_eqs[i]):
            CLEE_info[int(codenums_eqs[i]), 2] += 1
    CLEE_info = CLEE_info[CLEE_info[:,1] != 0, :]
    for i in range(CLEE_info.shape[0]):
        CLEE_info[i,3] = CLEE_info[i,2]/CLEE_info[i,1]
    CLEE_code = np.zeros((CLEE_info.shape[0], pS1s_mat.shape[0]))
    for i in range(CLEE_info.shape[0]):
        code = format(int(CLEE_info[i,0]), 'b')
        full_code = code
        while len(full_code) < pS1s_mat.shape[0]:
            full_code = '0'+full_code
        for j in range(pS1s_mat.shape[0]):
            if full_code[j] == '1':
                CLEE_code[i,j] = 1
    CLEE_full = np.concatenate((CLEE_code, CLEE_info[:,1:4]), axis = 1)
    return CLEE_full, have_eq

def plot_decision_tree(pS1s_mat, have_eq, div_y, iy, ix, stations, min_sample = 300):
    regressor = DecisionTreeRegressor(random_state=0)
    X = np.transpose(pS1s_mat.copy())
    not_nan = np.ones((len(have_eq),), dtype = bool)
    for i in range(len(have_eq)):
        if np.sum(np.isnan(X[i,:])) >= 3: # at least 3 stns not nan
            not_nan[i] = 0
    X = X[not_nan,:]
    X = np.round(X)
    y = have_eq.copy()[not_nan]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i,j]):
                X[i,j] = 0.5
    regressor = DecisionTreeRegressor(min_samples_leaf = min_sample)
    regressor.fit(X,y)
    
    dot_data = tree.export_graphviz(regressor, out_file=None, filled=True, rounded=True, 
                                    feature_names=stations, special_characters=True)  
    graph = graphviz.Source(dot_data) 
    graph.render('DT_MinSample%s_ix%s_iy%s_divy%s'%(min_sample, ix, iy, div_y))
    return

def pred_decision_tree(pS1s_mat, t_mat, have_eq, div_y, iy, ix, min_sample = 300,
                       pS1s_mat_pred = [], have_eq_pred = [], plot = 'no'):
    regressor = DecisionTreeRegressor(random_state=0)
    X = np.transpose(pS1s_mat.copy())
    not_nan = np.ones((len(have_eq),), dtype = bool)
    for i in range(len(have_eq)):
        if np.sum(np.isnan(X[i,:])) >= 3: # at least 3 stns not nan
            not_nan[i] = 0
    X = X[not_nan,:]
    X = np.round(X)
    y = have_eq.copy()[not_nan]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i,j]):
                X[i,j] = 0.5
    regressor = DecisionTreeRegressor(min_samples_leaf = min_sample)
    regressor.fit(X,y)
    # pred part
    if len(pS1s_mat_pred) == 0:
        pS1s_mat_pred = pS1s_mat
        have_eq_pred = have_eq
    X = np.transpose(pS1s_mat_pred.copy())
    X = np.round(X)
    y = have_eq_pred.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i,j]):
                X[i,j] = 0.5
    y_pred = regressor.predict(X)
    if plot != 'no':
        x_axis = pdate.num2date(t_mat)
        plt.plot(x_axis, y_pred)
        y_axis = (y>0)*y_pred
        y_axis[y_axis==0] = np.nan
        plt.plot(x_axis, y_axis, 'ro')
    return y_pred

def plot_OOS_EQrate(Tr_names, Te_names, ix = 0, iy = 0, div_y = 10, min_sample = 300,
                    min_TrEQ = 100, plot = 'yes', save = 'yes'):
    pS1s_mat, t_mat, locs, stations = make_pS1mat(Tr_names)
    have_eq = get_local_eq(t_mat, div_y = div_y, iy = iy, ix = ix, plot_directory = 'no', 
                           plot_local = 'no')
    pS1s_mat_pred, t_mat, locs, stations = make_pS1mat(Te_names)
    have_eq_pred = get_local_eq(t_mat, div_y = div_y, iy = iy, ix = ix, 
                                plot_directory = 'no', plot_local = 'no')
    if np.sum(have_eq) < min_TrEQ: 
        print('not enough EQs')
        return
    # plotting (a)
    fig = plt.figure(figsize=(24,14))
    sup_title = 'Out-of-sample test of decision trees (DT) for area (%s, %s), '%(ix, iy)
    sup_title += "with DT's min_sample = %s, min_TrEQ = %s"%(min_sample, min_TrEQ)
    plt.suptitle(sup_title)
    plt.subplot(2,1,1)
    y_pred = pred_decision_tree(pS1s_mat, t_mat, have_eq, div_y, iy, ix, 
                                pS1s_mat_pred = pS1s_mat_pred, min_sample = min_sample, 
                                have_eq_pred = have_eq_pred, plot = 'yes')
    plt.title("(a) Testing data's EQrates obtained with DT trained from training data")
    plt.xlabel('Time')
    plt.ylabel('DT-labeled EQ rate')
    # In dev check pred consistency 
    y_pred_list = []
    for i in range(len(y_pred)):
        if y_pred[i] not in y_pred_list:
            y_pred_list.append(y_pred[i])
    y_pred_list = np.array(y_pred_list)
    y_pred_info = np.zeros((len(y_pred_list), 4)) # cols: y_pred, len, numEQ, EQrate
    y_pred_info[:,0] = y_pred_list
    inds_pred = np.arange(len(y_pred_list))
    for i in range(len(y_pred)):
        i_pred = inds_pred[y_pred_list == y_pred[i]]
        y_pred_info[i_pred,1] += 1
        y_pred_info[i_pred,2] += have_eq_pred[i]
    for i in range(len(y_pred_info)):
        y_pred_info[i,3] = y_pred_info[i,2]/y_pred_info[i,1]
    # plotting (b)
    plt.subplot(2,1,2)
    label_len = y_pred_info[:,1]
    y_pred_info_major = y_pred_info[label_len>20,:]
    for i in range(y_pred_info_major.shape[0]):
        plt.scatter(y_pred_info_major[i,0], y_pred_info_major[i,3], 
                    label = 'Len = %.0f, EQ count = %.0f'%(y_pred_info_major[i,1], 
                                                           y_pred_info_major[i,2]))
    plt.legend()
    plt.xlabel('DT-labeled EQ rate')
    plt.ylabel('out-of-sample-tested EQ rate')
    plt.title('(b) DT-labeled EQ rates vs. corresponding out-of-sample-tested EQ rates')
    fig_name = 'OutOfSample_EQrate_MS%s_MTrEQ%s_ix%siy%s.png'%(min_sample, min_TrEQ, ix, iy)
    if save == 'yes':
        plt.savefig(fig_name, dpi = 500)
        plt.close(fig)
    return

def get_OOS_EQrate(Tr_names, Te_names, ix = 0, iy = 0, div_y = 10, min_sample = 300,
                   min_TrEQ = 100):
    pS1s_mat, t_mat, locs, stations = make_pS1mat(Tr_names)
    have_eq = get_local_eq(t_mat, div_y = div_y, iy = iy, ix = ix, plot_directory = 'no', 
                           plot_local = 'no')
    pS1s_mat_pred, t_mat, locs, stations = make_pS1mat(Te_names)
    have_eq_pred = get_local_eq(t_mat, div_y = div_y, iy = iy, ix = ix, 
                                plot_directory = 'no', plot_local = 'no')
    if np.sum(have_eq) < min_TrEQ: 
        print('not enough EQs')
        return np.array([]) 
    y_pred = pred_decision_tree(pS1s_mat, t_mat, have_eq, div_y, iy, ix, 
                                pS1s_mat_pred = pS1s_mat_pred, min_sample = min_sample, 
                                have_eq_pred = have_eq_pred, plot = 'no')
    y_pred_list = []
    for i in range(len(y_pred)):
        if y_pred[i] not in y_pred_list:
            y_pred_list.append(y_pred[i])
    y_pred_list = np.array(y_pred_list)
    y_pred_info = np.zeros((len(y_pred_list), 4)) # cols: y_pred, len, numEQ, EQrate
    y_pred_info[:,0] = y_pred_list
    inds_pred = np.arange(len(y_pred_list))
    for i in range(len(y_pred)):
        i_pred = inds_pred[y_pred_list == y_pred[i]]
        y_pred_info[i_pred,1] += 1
        y_pred_info[i_pred,2] += have_eq_pred[i]
    for i in range(len(y_pred_info)):
        y_pred_info[i,3] = y_pred_info[i,2]/y_pred_info[i,1]
#    EQrate_TR = np.sum(have_eq)/len(have_eq)
#    EQrate_TE = np.sum(have_eq_pred)/len(have_eq_pred)
    EQrate_TE_multiplier = 1#EQrate_TR/EQrate_TE
    y_pred_info_multi = np.zeros((y_pred_info.shape[0], 5))
    y_pred_info_multi[:, 0:4] = y_pred_info
    y_pred_info_multi[:,4] = y_pred_info[:,3]*EQrate_TE_multiplier
    label_len = y_pred_info[:,1]
    y_pred_info_major_multi = y_pred_info_multi[label_len > 20,:]
    return y_pred_info_major_multi


# ------ below: from EP_NS_Scramble ------
    
def make_nan_zero(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i,j]):
                mat[i,j] = 0
    return mat

def accumulated_map(file_names, GM_dim = 30, NS_keep_eq = 'S', NS_stn = 'N', plot = 'yes',
                    save = 'yes', return_info = 'no'):
    # inits
    eqks = mB.eqk()
    position = eqks.position
    south_stns = dict()
    north_stns = dict()
    latitudes = []
    for stn in position:
        latitudes.append(position[stn][0])
    latitudes = np.array(latitudes) 
    latitudes = np.sort(latitudes)
    mid_latitude = (latitudes[9] + latitudes[10])/2
    eq_lats = eqks.lld[:,0]
    div_y_N = np.int((np.max(eq_lats)-mid_latitude)/(np.max(eq_lats)-np.min(eq_lats))*GM_dim)
    div_y_S = GM_dim - div_y_N
    for stn in position:
        if position[stn][0] > mid_latitude:
            north_stns[stn] = position[stn]
        else:
            south_stns[stn] = position[stn]
    # starting
    if NS_stn == 'N':
        NS_stns = north_stns
    elif NS_stn == 'S':
        NS_stns = south_stns
    if NS_keep_eq == 'S':
        div_y = div_y_S
    elif NS_keep_eq == 'N':
        div_y = div_y_N
    plot_infos = []
    accu_i = 0
    for file_name in file_names:
        stn = file_name[7:11]
        if stn in NS_stns.keys():
            info = bm.HSVA_getinfo(file_name)
            plot_info = bm.plot_EQGM(info, state_threshold = 0.5, div_y = div_y, 
                                     div_x = GM_dim, plot = 'no', 
                                     NS_keep_eq = NS_keep_eq, return_info = 'yes')
            plot_infos.append(plot_info)
            if accu_i == 0:
                accu_EQS1 = make_nan_zero(plot_info['EQS1_plot'])
                accu_EQS2 = make_nan_zero(plot_info['EQS2_plot'])
                accu_lenS1 = plot_info['len_S1']
                accu_lenS2 = plot_info['len_S2']
            else:
                accu_EQS1 += make_nan_zero(plot_info['EQS1_plot'])
                accu_EQS2 += make_nan_zero(plot_info['EQS2_plot'])
                accu_lenS1 += plot_info['len_S1']
                accu_lenS2 += plot_info['len_S2']
            accu_i += 1
    accu_EQrateS1 = accu_EQS1/accu_lenS1
    accu_EQrateS2 = accu_EQS2/accu_lenS2
    accu_EQratios = accu_EQrateS1/(accu_EQrateS1+accu_EQrateS2)
    # plotting
    xticks = plot_info['xticks']
    yticks = plot_info['yticks']
    xtick_labels = plot_info['xtick_labels']
    ytick_labels = plot_info['ytick_labels']
    yticks = plot_info['yticks']
    if plot == 'yes':
        fig = plt.figure(figsize=(14,14))
        fs = 7
        ax = plt.subplot()
        plt.imshow(accu_EQratios, cmap="PiYG")
        for iy in range(div_y):
            for ix in range(GM_dim):
                if ~np.isnan(accu_EQratios[iy, ix]):
                    entry = accu_EQratios[iy, ix]*100
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
        for plot_info in plot_infos:
            stn_coord = plot_info['stn_coord']
            stn_name = plot_info['stn_name']
            ax.scatter(bm.get_tick(stn_coord[1],xtick_labels), 
                       bm.get_tick(stn_coord[0],ytick_labels), 
                       marker = '*', s = 240, label = stn_name)
        plt.legend()
        NS_txt = 'stations in '+NS_stn+' and EQs in '+NS_keep_eq
        ax.set_title('Accumulated EQratios (%) heatmap of EQRate_S1/EQRate_S(1+2) with '+NS_txt)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        if save == 'yes':
            fig_name = 'EQratios_stns='+NS_stn+'_EQs='+NS_keep_eq
            plt.savefig(fig_name + '.png', dpi = 500)
            plt.close(fig)
        return
    if return_info == 'yes':
        return accu_EQratios, plot_infos
    
def get_EQratios_array(plot_infos, minEQ = 20):
    # plot_infos is the output of function accumulated_map
    # function returns the array of EQratios gathering at all locations with >minEQ EQs
    EQratios_list = []
    for i_info in range(10):
        plot_info = plot_infos[i_info]
        EQratios_plot = plot_info['EQratios_plot']
        EQ_counts = plot_info['EQS1_plot'] + plot_info['EQS2_plot']
        for ix in range(EQ_counts.shape[0]):
            for iy in range(EQ_counts.shape[1]):
                if EQ_counts[ix, iy] > minEQ:
                    if ~np.isnan(EQratios_plot[ix, iy]):
                        EQratios_list.append(EQratios_plot[ix, iy])
        EQratios_array = np.array(EQratios_list)
    return EQratios_array

def NS_standardize(file_name, GM_dim = 30): 
#    for each H2S file, we need to decide if we need to 'flip' the entire H2S file, 
#    by checking the voted HS TS's EQ_ratio for local (N or S) EQs only.
    eqks = mB.eqk()
    position = eqks.position
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
    stn = file_name[7:11]
    if stn in north_stns.keys():
        NS_stn = 'N'
    elif stn in south_stns.keys():
        NS_stn = 'S'
    file_names = [file_name]
    accu_EQratios_NS, _ = accumulated_map(file_names, GM_dim = GM_dim, NS_keep_eq = NS_stn, 
                                          NS_stn = NS_stn, plot = 'no', return_info = 'yes')
    if np.nanmean(accu_EQratios_NS) > 0.5:
        print(stn + ' force fliped.')
        with open(file_name, 'rb') as f:
            dic_config = pickle.load(f)
            gen_paras = pickle.load(f)
        for config in dic_config:
            info_dic = dic_config[config]
            for info_name in info_dic:
                info = info_dic[info_name]
                info = bm.standardize_HS(info, force = 'yes')
                info_dic[info_name] = info
            dic_config[config] = info_dic
        with open(file_name[0:-7]+'NS.pickle', 'wb') as f:
            pickle.dump(dic_config, f)
            pickle.dump(gen_paras, f)
    else:
        print(stn + ' not force fliped.')
    return


def Rad_standardize(file_name, mag_LB = 0, rad_UB = 0.8, minEQ_rad = 0):
    # for dictionary 'info' of format outputed by BM_CHMMMI or BM_CHMM with 2-by-2 A
    info = bm.HSVA_getinfo(file_name)
    eqks = mB.eqk()
    eqks.select(info['dataname'], mag_LB, rad_UB)
    t_eqs = eqks.selected[:,0]
    if minEQ_rad > 0:
        new_rad_UB = rad_UB
        while len(t_eqs) < minEQ_rad:
            new_rad_UB += 0.1
            print(info['dataname'][0:4] + ' rad_UB expands to %s.'%new_rad_UB)
            eqks = mB.eqk()
            eqks.select(info['dataname'], mag_LB, new_rad_UB)
            t_eqs = eqks.selected[:,0]
    eq_rates, eqs_S1, len_S1, eqs_S2, len_S2 = bm.compute_EQrates(0.45, t_eqs, info, 
                                                                  return_all = 'yes')
    EQratio = eq_rates[0]/(eq_rates[1]+eq_rates[0])
    with open(file_name, 'rb') as f:
            dic_config = pickle.load(f)
            gen_paras = pickle.load(f)
    if EQratio > 0.5:#np.nanmean(EQratios) > 0.5:
        print(info['dataname'][0:4] + ' force fliped.')
        for config in dic_config:
            info_dic = dic_config[config]
            for info_name in info_dic:
                info = info_dic[info_name]
                info = bm.standardize_HS(info, force = 'yes')
                info_dic[info_name] = info
            dic_config[config] = info_dic
    else:
        print(info['dataname'][0:4] + ' not force fliped.')
        gen_paras['new_rad_UB'] = new_rad_UB
        gen_paras['minEQ_rad'] = minEQ_rad
    with open(file_name[0:-7]+'_rad%s_min%s.pickle'%(rad_UB, minEQ_rad), 'wb') as f:
        pickle.dump(dic_config, f)
        pickle.dump(gen_paras, f)
    return 


def indi_standardize(file_name, target = 'Vs_MeanOfSegs', compare = 'S1<S2', save = 'yes'):
    info = bm.HSVA_getinfo(file_name)
    max_plot = 16
    for i_input in range(max_plot):
        indi_name = info['input_names'][i_input]
        if indi_name.endswith('col1_'+target):
            indi_c1_S1 = info['segindic_TSs'][i_input, info['Gamma'][0,:] > 0.6]
            indi_c1_S2 = info['segindic_TSs'][i_input, info['Gamma'][1,:] > 0.6]
            indi_c1_S1 = bm.cut_ends(indi_c1_S1)
            indi_c1_S2 = bm.cut_ends(indi_c1_S2)
            indi_c1 = np.concatenate((indi_c1_S1, indi_c1_S2))
            indi_c1_S1 = (indi_c1_S1-np.mean(indi_c1))/np.std(indi_c1)
            indi_c1_S2 = (indi_c1_S2-np.mean(indi_c1))/np.std(indi_c1)
        if indi_name.endswith('col2_'+target):
            indi_c2_S1 = info['segindic_TSs'][i_input, info['Gamma'][0,:] > 0.6]
            indi_c2_S2 = info['segindic_TSs'][i_input, info['Gamma'][1,:] > 0.6]
            indi_c2_S1 = bm.cut_ends(indi_c2_S1)
            indi_c2_S2 = bm.cut_ends(indi_c2_S2)
            indi_c2 = np.concatenate((indi_c2_S1, indi_c2_S2)) 
            indi_c2_S1 = (indi_c2_S1-np.mean(indi_c2))/np.std(indi_c2)
            indi_c2_S2 = (indi_c2_S2-np.mean(indi_c2))/np.std(indi_c2)
    indi_S1 = np.concatenate((indi_c1_S1, indi_c2_S1))
    indi_S2 = np.concatenate((indi_c1_S2, indi_c2_S2))
    flip = 'no'
    if compare == 'S1>S2':
        if np.mean(indi_S1) < np.mean(indi_S2): 
            flip = 'yes'
    elif compare == 'S1<S2':
        if np.mean(indi_S1) > np.mean(indi_S2): 
            flip = 'yes'
    if save == 'yes':
        with open(file_name, 'rb') as f:
            dic_config = pickle.load(f)
            gen_paras = pickle.load(f)
        if flip == 'yes': 
            print(info['dataname'][0:4] + ' force fliped.')
            for config in dic_config:
                info_dic = dic_config[config]
                for info_name in info_dic:
                    info = info_dic[info_name]
                    info = bm.standardize_HS(info, force = 'yes')
                    info_dic[info_name] = info
                dic_config[config] = info_dic
        elif flip == 'no':
            print(info['dataname'][0:4] + ' not force fliped.')
        with open(file_name[0:-7]+'_'+target+'_'+compare+'.pickle', 'wb') as f:
            pickle.dump(dic_config, f)
            pickle.dump(gen_paras, f)
    elif save == 'no':
        if flip == 'yes': 
            print(info['dataname'][0:4] + ' to be force fliped.')
        elif flip == 'no':
            print(info['dataname'][0:4] + ' not to be force fliped.')
    return

from scipy.special import comb
from decimal import Decimal

def compute_P_binohypo(eqs_S1, len_S1, eqs_S2, len_S2, range_1 = 'yes'):
    # function returns the p value of the hypothesis that EQrate_S1 < EQrate_S2
    # if p < 0.05, then it is true. (assumnig binomial distribution)
    # by computing probability of SL having eqs_SL or less EQs than SM
    if eqs_S1/len_S1 >= eqs_S2/len_S2: # SL: state of less EQrate, SM: state of more EQrate
        eqs_SL, eqs_SM = eqs_S2, eqs_S1
        len_SL, len_SM = len_S2, len_S1
        SL = 'S2'
    else:
        eqs_SL, eqs_SM = eqs_S1, eqs_S2
        len_SL, len_SM = len_S1, len_S2
        SL = 'S1'
    EQ_total = eqs_SL + eqs_SM
    P_aEQatSL = len_SL/(len_SL+len_SM) # the assumptions of the binomial dist
    P_aEQatSM = len_SM/(len_SL+len_SM)
    p_hypo = 0
    num_EQatSL = eqs_SL
    while num_EQatSL >= 0: # P(eqs_SL at SL) + P(eqs_SL-1 at SL) + ... + P(0 at SL)
        num_EQatSM = EQ_total - num_EQatSL
        if num_EQatSL > 220:
            p = Decimal(comb(EQ_total, num_EQatSL, exact=True))
            p = p*Decimal((P_aEQatSL**num_EQatSL)*(P_aEQatSM**num_EQatSM))
            p_hypo += float(p)
        else:
            p = comb(EQ_total, num_EQatSL, exact=True)
            p = p*(P_aEQatSL**num_EQatSL)*(P_aEQatSM**num_EQatSM)
            p_hypo += p
        num_EQatSL -= 1
    if range_1 == 'yes':
        if SL == 'S2':
            p_hypo = 1 - p_hypo
    return p_hypo


def LFS(file_batch, div_y = 20, div_x = 0, state_threshold = 0.5, EQrate_mode = 'S1',
        min_EQ = 20, mag_LB = 0):
    file_names = ['D_FENG_ti2_FCW500S2',
                  'D_KUOL_ti2_FCW500S2',
#                  'D_TOCH_ti2_FCW500S2',
                  'D_LIOQ_ti2_FCW500S2',
                  'D_YULI_ti2_FCW500S2',
                  'D_SIHU_ti2_FCW500S2',
                  'D_PULI_ti2_FCW500S2',
                  'D_HERM_ti2_FCW500S2',
#                  'D_FENL_ti2_FCW500S2',
                  'D_CHCH_ti2_FCW500S2',
                  'D_WANL_ti2_FCW500S2',
                  'D_SHCH_ti2_FCW500S2',
#                  'D_LISH_ti2_FCW500S2',
                  'D_KAOH_ti2_FCW500S2',
                  'D_HUAL_ti2_FCW500S2',
                  'D_ENAN_ti2_FCW500S2',
                  'D_DAHU_ti2_FCW500S2',
                  'D_DABAXti2_FCW500S2',
#                  'D_RUEY_ti2_FCW500S2',
                  'D_SHRL_ti2_FCW500S2']
#    file_batch = 'BW0813_CVSK200_cV_BW0813_VSK_R100L60_0828ANH2S_rad0.6_min500'
    for i_file in range(len(file_names)):
        file_name = 'CHMMMI_' + file_names[i_file][2:] + '_' + file_batch + '.pickle'
        file_names[i_file] = file_name
    eqks = mB.eqk()
    if mag_LB > 0:
        M_eqs = eqks.mat[:,1].copy()
        eqks.lld = eqks.lld[M_eqs>mag_LB, :]
        eqks.mat = eqks.mat[M_eqs>mag_LB, :]
    eqks.makeGM(div_y = div_y, div_x = div_x)
    x_cuts = eqks.x_cuts
    y_cuts = eqks.y_cuts
    numEQs = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
    numEQs_got = 0
    EQratio_dic = {}
    # 
    for file_name in file_names:
        stn_name = file_name[7:11]
        info = bm.HSVA_getinfo(file_name)
        EQratios = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
        if numEQs_got == 0:
            for ix in range(len(x_cuts)-1):
                for iy in range(len(y_cuts)-1):
                    GMselected = eqks.GM_select(iy, ix)
                    t_eqs = GMselected[:,0]
                    numEQs[iy, ix] = len(t_eqs)
            numEQs_got = 1
        for ix in range(len(x_cuts)-1):
            for iy in range(len(y_cuts)-1):
                GMselected = eqks.GM_select(iy, ix)
                t_eqs = GMselected[:,0]
                if len(t_eqs) > min_EQ:
                    eq_rates, _, _, _, _ = bm.compute_EQrates(state_threshold, t_eqs, info, 
                                                              return_all = 'yes', min_EQ = min_EQ)
                    if EQrate_mode == 'S1':
                        EQratios[iy, ix] = eq_rates[0]/(eq_rates[1]+eq_rates[0]) # -changed!!-
                    elif EQrate_mode == 'abs':
                        EQratios[iy, ix] = np.min(eq_rates)/np.max(eq_rates)
                else:
                    EQratios[iy, ix] = np.nan
        EQratio_dic[stn_name] = EQratios
    # flipping
    for file_name in file_names:
        stn_name = file_name[7:11]
        EQratio_dic[stn_name] = np.flipud(EQratio_dic[stn_name])
    numEQs = np.flipud(numEQs)
    y_cuts = np.flip(y_cuts)
    # making slopes
    slopes = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
    p_values = np.zeros((len(y_cuts)-1, len(x_cuts)-1))
    slopes[:] = np.nan
    p_values[:] = np.nan
    # dev
#    ix, iy = 7, 14
    # dev
    for ix in range(len(x_cuts)-1):
        for iy in range(len(y_cuts)-1):
            if numEQs[iy, ix] > min_EQ:
                ix_ctr = np.mean([x_cuts[ix], x_cuts[ix+1]])
                iy_ctr = np.mean([y_cuts[iy], y_cuts[iy+1]])
                #print(numEQs[iy, ix])
                distances = []
                y_metrics = []
                for file_name in file_names:
                    stn_name = file_name[7:11]
                    position = eqks.position[stn_name]
                    distance = np.sqrt((ix_ctr-position[1])**2 + (iy_ctr-position[0])**2)
                    y_metric = np.abs(EQratio_dic[stn_name][iy, ix]-0.5)
                    # dev
#                    y_metric = EQratio_dic[stn_name][iy, ix]
#                    plt.scatter(distance, y_metric, label = stn_name)
#                    plt.legend()
                    # dev
                    distances.append(distance)
                    y_metrics.append(y_metric)
#                slope, _, _, p_value, _ = stats.linregress(distances, y_metrics)
                tau, p_value = stats.kendalltau(distances, y_metrics)
                slopes[iy, ix] = tau
                p_values[iy, ix] = p_value
    # plotting
    slopes_cp = slopes.copy()
    slopes_cp[numEQs < min_EQ] = np.nan
    p_values_cp = p_values.copy()
    p_values_cp[numEQs < min_EQ] = np.nan
    numEQs_cp = numEQs.copy()
    numEQs_cp[numEQs < min_EQ] = np.nan
    fig = plt.figure(figsize=(22,14))
    # original
    plt.subplot(3,3,1)
    sns.heatmap(slopes_cp*100, center = 0.0, annot=True, cmap="PiYG", cbar=False)
    plt.ylabel('numEQs > %s'%min_EQ)
    plt.title('Tau * 100')
    plt.subplot(3,3,2)
    plt.title('Fit p_values * 100')
    sns.heatmap(p_values_cp*100, annot=True, cbar=False)
    plt.subplot(3,3,3)
    plt.title('numEQs * 10')
    sns.heatmap(numEQs_cp/10, annot=True, cbar=False, vmin = 0)
    # p value restrict
    slopes_cp[p_values_cp > 0.5] = np.nan
    numEQs_cp[p_values_cp > 0.5] = np.nan
    p_values_cp[p_values_cp > 0.5] = np.nan
    plt.subplot(3,3,4)
    sns.heatmap(slopes_cp*100, center = 0.0, annot=True, cmap="PiYG", cbar=False)
    plt.ylabel('numEQs > 20, p_values < 0.5')
    plt.subplot(3,3,5)
    sns.heatmap(p_values_cp*100, annot=True, cbar=False)
    plt.subplot(3,3,6)
    sns.heatmap(numEQs_cp/10, annot=True, cbar=False, vmin = 0)
    # numEQs restrict
    slopes_cp = slopes.copy()
    slopes_cp[numEQs < 80] = np.nan
    p_values_cp = p_values.copy()
    p_values_cp[numEQs < 80] = np.nan
    numEQs_cp = numEQs.copy()
    numEQs_cp[numEQs < 80] = np.nan
    plt.subplot(3,3,7)
    sns.heatmap(slopes_cp*100, center = 0.0, annot=True, cmap="PiYG", cbar=False)
    plt.ylabel('numEQs > 80')
    plt.subplot(3,3,8)
    sns.heatmap(p_values_cp*100, annot=True, cbar=False)
    plt.subplot(3,3,9)
    sns.heatmap(numEQs_cp/10, annot=True, cbar=False, vmin = 0)
    plt.suptitle('Tau of abs(EQratio-0.5) vs distance for '+ file_batch)
    plt.savefig('Tau_div%s'%div_y +'_'+file_batch + 'skip4.png', dpi = 500)
    plt.close(fig)
    return


def compute_extreme_rates(datanames, after_fix, min_EQ, mag_LB, div_y, p = 0.05,
                          info_dir = 'EQGM ensemble plot infos/'):
    EQratios_plots = np.zeros((len(datanames), div_y, div_y))
    EQratios_plots[:] = np.nan
    for i_file_name in range(len(datanames)):
        file_name = datanames[i_file_name]
        file_name = 'CHMMMI_' + file_name[2:] + after_fix
        pltinfo_name = 'EQGM_ensemble_' + file_name[0:-7]
        pltinfo_name += '_minEQ%s_magLB%s_div%s_info.pickle'%(min_EQ, mag_LB, div_y)
        with open(info_dir + pltinfo_name, 'rb') as f:
            plot_info = pickle.load(f)
        if plot_info['EQratios_plot'].shape == (div_y, div_y): 
            # a make shift way, to improve: standardize GM cuts across stations
            EQratios_plots[i_file_name, :, :] = plot_info['EQratios_plot']
    EQRplot_xtrem = np.zeros((div_y, div_y))
    EQRplot_xtrem[:] = np.nan
    for i in range(div_y):
        for j in range(div_y):
            EQratios_plot_ij = EQratios_plots[:, i, j] 
            num_participating = len(datanames) - np.sum(np.isnan(EQratios_plot_ij))
            if num_participating > 0:
                EQRplot_xtrem[i,j] = np.sum(EQratios_plot_ij < p)
                EQRplot_xtrem[i,j] += np.sum(EQratios_plot_ij > 1-p)
                EQRplot_xtrem[i,j] /= num_participating
    return EQRplot_xtrem, pltinfo_name

def get_EQR_values(datanames, after_fix, min_EQ, mag_LB, div_y, 
                   info_dir = 'EQGM ensemble plot infos/'):
    EQratios_plots = np.zeros((len(datanames), div_y, div_y))
    EQratios_plots[:] = np.nan
    EQR_values = np.array([])
    for i_file_name in range(len(datanames)):
        file_name = datanames[i_file_name]
        file_name = 'CHMMMI_' + file_name[2:] + after_fix
        pltinfo_name = 'EQGM_ensemble_' + file_name[0:-7]
        pltinfo_name += '_minEQ%s_magLB%s_div%s_info.pickle'%(min_EQ, mag_LB, div_y)
        with open(info_dir + pltinfo_name, 'rb') as f:
            plot_info = pickle.load(f)
        EQratios_plot = plot_info['EQratios_plot']
        EQR_values = np.concatenate((EQR_values, EQratios_plot[~np.isnan(EQratios_plot)]))
    return EQR_values, pltinfo_name

# below: tool for create_bootstrap_plot_info
def next_state(weights):
    choice = np.random.random() * sum(weights)
    for i, w in enumerate(weights):
        choice -= w
        if choice < 0:
            return i

def simulate_hmm(transmat, emimat, pi, t_len = 100, states_only = 'no'):
    init_state = next_state(pi)
    observes = np.zeros((t_len,), dtype = int)
    states = np.zeros((t_len,), dtype = int)
    states[0] = init_state
    observes[0] = next_state(emimat[init_state,:])
    if states_only == 'no':
        for i in range(t_len-1):
            states[i+1] = next_state(transmat[states[i],:])
            observes[i+1] = next_state(emimat[states[i+1],:])
    else:
        for i in range(t_len-1):
            states[i+1] = next_state(transmat[states[i],:])
        observes = np.nan
    return states, observes

def create_HSensembles_wModelPara(info, num_ensemble = 20, flip = 'no'):
    t_len = len(info['pS1s_voted'])
    # pS1s_voted, A= info['pS1s_voted'], info['A']
    # flip = 'no'
    # if np.sum(pS1s_voted)/len(pS1s_voted) > 0.5:
    #     if A[0,0] < A[1,1]:
    #         flip = 'yes'
    # else:
    #     if A[0,0] > A[1,1]:
    #         flip = 'yes'
    if flip == 'yes':
        A_copy = info['A'].copy()
        info['A'][0,0] = A_copy[1,1]
        info['A'][1,0] = A_copy[0,1]
        info['A'][0,1] = A_copy[1,0]
        info['A'][1,1] = A_copy[0,0]
        B_copy = info['B'].copy()
        info['B'][0,:] = B_copy[1,:]
        info['B'][1,:] = B_copy[0,:]
        # Gamma_copy = info['Gamma'].copy()
        # info['Gamma'][0,:] = Gamma_copy[1,:]
        # info['Gamma'][1,:] = Gamma_copy[0,:]
        pi_copy = info['pi'].copy()
        info['pi'][0] = pi_copy[1]
        info['pi'][1] = pi_copy[0]
    ensembles = np.zeros((num_ensemble+1, t_len))
    for i in range(num_ensemble):
        states, _ = simulate_hmm(info['A'], info['B'], info['pi'], t_len, states_only = 'yes')
        ensembles[i+1,:] = states
    ensembles[0,:] = info['pS1s_voted']
    return ensembles
# above: tool for create_bootstrap_plot_info

def create_bootstrap_plot_info(datanames, after_fix, min_EQ, mag_LB, div_y, shuffle_perstn = 10, 
                               num_ensemble = 100, ind_start = 0, save_dir = '', 
                               fully_save = 'no', version = 'old', ReSimu = 'no'):
    # assign version as '0125' for 0125 version
    # ReSimu = 'yes' for re-simulating from optimal models parameter to create ensemble instead 
    # of reshuffling
    for i_file_name in range(len(datanames)):
        file_name = datanames[i_file_name]
        file_name = 'CHMMMI_' + file_name[2:] + after_fix
        if version == 'old':
            info = bm.HSVA_getinfo(file_name)
        elif version == '0125':
            if ReSimu == 'no':
                info = bm.HSVA_getinfo_0125(file_name)
            else:
                info = bm.HSVA_getinfo_0125(file_name, keep_models = 'yes')
        if ReSimu == 'no':
            ensembles = bm.create_HSensembles(info, num_ensemble = shuffle_perstn)
        else:
            ensembles = create_HSensembles_wModelPara(info, num_ensemble = shuffle_perstn)
        ensembles = ensembles[1:, :]
        for i_en in range(shuffle_perstn):
            len_TS = len(info['t_segs'])
            Gamma = np.zeros((2, len_TS))
            Gamma[0,:] = ensembles[i_en, :]
            Gamma[1,:] = 1 - ensembles[i_en, :]
            ensemb_info = {'Gamma': Gamma, 't_segs':info['t_segs'], 'dataname':info['dataname']}
            plot_info = bm.plot_EQGM(ensemb_info, state_threshold = 0.5, div_y = div_y, 
                                     plot = 'no', return_info = 'yes', min_EQ = min_EQ, 
                                     mag_LB = mag_LB, EQrate_mode = 'S1', 
                                     num_ensemble = num_ensemble, fully_save = fully_save)
            txt = '_minEQ%s'%min_EQ
            if mag_LB > 0:
                txt += '_magLB%s'%mag_LB
            txt += '_div%s'%div_y
            save_txt = save_dir+file_name[0:-7]+txt+'_info_S1ensemble%s.pickle'%(i_en+ind_start)
            with open(save_txt, 'wb') as f:
                pickle.dump(plot_info, f)
    return

def plot_from_info(pltinfo_name, info_dir = '', EQrate_mode = 'ensemble'):
#    sub_inds = [2,2,1]
#    plot_prefix = ['(a)', '(b)', '(c)', '(d)']
    with open(info_dir + pltinfo_name, 'rb') as f:
        plot_info = pickle.load(f)
    EQratios_plot = plot_info['EQratios_plot']
    EQS1_plot, EQS2_plot = plot_info['EQS1_plot'], plot_info['EQS2_plot']
    xticks, xtick_labels= plot_info['xticks'], plot_info['xtick_labels']
    yticks, ytick_labels = plot_info['yticks'], plot_info['ytick_labels']
    stn_name, stn_coord = plot_info['stn_name'], plot_info['stn_coord']
    x_cuts, y_cuts = plot_info['x_cuts'], plot_info['y_cuts']
    len_S1, len_S2 = plot_info['len_S1'], plot_info['len_S2']
    fs = 7
    fig = plt.figure(figsize=(22,9))
    figure_name = pltinfo_name[0:-7]
    plt.suptitle(figure_name)
    # 1
    ax = plt.subplot(2, 1, 1)
    plt.plot(plot_info['Gamma'][0,:])
    plt.title('(a) Voted Posterior P(S1)')
    plt.xlabel("Time")
    # 2
    ax = plt.subplot(2,3,4)
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
    ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
               marker = '*', color='r', s = 240, label = stn_name)
    plt.legend()
    ax.set_title('(b) EQ_S1s, with len_S1 = %s'%len_S1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # 2
    ax = plt.subplot(2,3,5)
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
    ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
               marker = '*', color='r', s = 240, label = stn_name)
    plt.legend()
    ax.set_title('(c) EQ_S2s, with len_S2 = %s'%len_S2)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # 4
    ax = plt.subplot(2,3,6)
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
    ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
               marker = '*', color='r', s = 240, label = stn_name)
    plt.legend()
    if EQrate_mode == 'S1':
        ax.set_title('(d) EQratios (%) heatmap of EQRate_S1/EQRate_S(1+2)')
    if EQrate_mode == 'abs':
        ax.set_title('(d) EQratios (%) heatmap of EQRate_whichever_lower/EQRate_whichever_higher')
    if EQrate_mode == 'hypo':
        ax.set_title('(d) p(%) of rejecting hypothesis (binomial): EQRate_S1 < EQRate_S2')
    elif EQrate_mode =='ensemble':
        ax.set_title('(d) p(%) of rejecting hypothesis (ensemble): EQRate_S1 < EQRate_S2')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(info_dir + pltinfo_name[0:-7] + '_replot.png', dpi = 500)
    plt.close(fig)
    return

def all_stns_plot(min_EQ = 20, mag_LB = 3, div_y = 12, main_folder = 'testF_main', 
                  ensemble_folder = 'testF', indi_txt = '_cV_BW0813_CSK_', plot = 'yes', 
                  indi_txt_key = 'CSK', save_extratxt = '', shift = 0, plot_Disc = 'yes', 
                  return_dic = 'no', datanames = '', num_EQ_plot = 1,
                  paratxt = '_CVSKXXXLXX'):
    div_txt = '_div%s'%div_y
    EQ_txt = '_minEQPlot%s_mag%s'%(num_EQ_plot, mag_LB)
    if shift != 0:
        EQ_txt += '_shift%s'%shift
    EQ_txt += save_extratxt
    ratio_higher_dic = {}
    EQRs_highdisc_dic = {}
    EQratios_dic = {}
    EQRs_en_disc_dic = {}
    for dataname in datanames:
        stn_txt = dataname[2:6]
        name_list = os.listdir(main_folder)
        for name in name_list:
            if '.pickle' in name:
                if indi_txt in name:
                    if div_txt in name:
                        if stn_txt in name:
                            if shift != 0:
                                if '_shift%s'%shift in name:
                                    target_file = name
                            else:
                                target_file = name
        #
        name_list = os.listdir(ensemble_folder)  
        ensemble_names = []
        for name in name_list:
            if '.pickle' in name:
                if indi_txt in name:
                    if div_txt in name:
                        if stn_txt in name:
                            ensemble_names.append(name)
        # 
        EQratios_en = np.zeros((len(ensemble_names), div_y, div_y))
        EQratios_en[:] = np.nan
        EQRs_en_disc = np.zeros((len(ensemble_names), div_y, div_y))
        EQRs_en_disc[:] = np.nan # the EQratio's distance to midpoint of 0.5, discrimination power
        for i_en in range(len(ensemble_names)):
            ensemble_name = ensemble_names[i_en]
            with open(ensemble_folder + '/' + ensemble_name, 'rb') as f:
                plot_info = pickle.load(f)
            EQratios_plot = plot_info['EQratios_plot']
            EQratios_en[i_en, :, :] = EQratios_plot
            EQRs_en_disc[i_en, :, :] = np.abs(EQratios_plot - 0.5)
        EQRs_en_disc_dic[stn_txt] = EQRs_en_disc
        # 
        with open(main_folder + '/' + target_file, 'rb') as f:
            plot_info = pickle.load(f)
        EQratios = plot_info['EQratios_plot']
        EQratios_disc = np.abs(EQratios - 0.5)
        # 
        EQRs_highdisc = np.zeros((div_y, div_y))
        EQRs_highdisc[:] = np.nan
        for i in range(div_y):
            for j in range(div_y):
                EQratio_disc = EQratios_disc[i,j]
                if ~np.isnan(EQratio_disc):
                    EQR_en_disc = EQRs_en_disc[:, i, j]
                    EQRs_highdisc[i,j] = np.sum(EQratio_disc > EQR_en_disc)
        EQRs_highdisc = EQRs_highdisc/EQR_en_disc.shape[0]
        # 
        mat = EQRs_highdisc
        if len(mat[~np.isnan(mat)]) > 0:
            ratio_higher = len(mat[mat>0.5])/len(mat[~np.isnan(mat)])
        else:
            ratio_higher = np.nan
        ratio_higher_dic[stn_txt] = ratio_higher
        EQRs_highdisc_dic[stn_txt] = EQRs_highdisc
        EQratios_dic[stn_txt] = EQratios
    # In
    xticks, yticks = plot_info['xticks'][0:div_y-2], plot_info['yticks'][0:div_y-2]
    x_cuts, y_cuts = plot_info['x_cuts'][1:div_y-1], plot_info['y_cuts'][1:div_y-1]
    xtick_labels = plot_info['xtick_labels'][1:div_y-1]
    ytick_labels = plot_info['ytick_labels'][1:div_y-1]
    # ------ new: 20201103 ------
    EQS1 = plot_info['EQS1_plot']
    EQS1[np.isnan(EQS1)] = 0
    EQS2 = plot_info['EQS2_plot']
    EQS2[np.isnan(EQS2)] = 0
    nEQs = EQS1 + EQS2
    EQ_pass = np.zeros((div_y, div_y), dtype = bool)
    EQ_pass[nEQs>num_EQ_plot] = True
    EQ_pass = EQ_pass[1:div_y-1, 1:div_y-1]
    # ------ new: 20201103 ------
    eqks = mB.eqk()
    fs = np.int(7*10/(div_y-2))
    if plot == 'yes':
        i_plot = 1
        fig = plt.figure(figsize=(20,14))
        for stn_txt in EQRs_highdisc_dic:
            EQRs_highdisc = EQRs_highdisc_dic[stn_txt]
            EQRs_highdisc = EQRs_highdisc[1:div_y-1, 1:div_y-1]
            stn_coord = eqks.position[stn_txt]
            ax = plt.subplot(4,5, i_plot)
            EQRs_highdisc_plot = np.copy(EQRs_highdisc)
            EQRs_highdisc_plot[~EQ_pass] = np.nan
            plt.imshow(EQRs_highdisc_plot, cmap="Reds", vmin=0, vmax=1)
            for iy in range(len(y_cuts)):
                for ix in range(len(x_cuts)):
                    if ~np.isnan(EQRs_highdisc_plot[iy, ix]):
                        entry = EQRs_highdisc_plot[iy, ix]*100
                        if entry < 50:
                            c = 'k'
                        else:
                            c = 'w'
                        ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                                fontsize=fs)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
            ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
                       marker = '*', color='b', s = 160, label = stn_txt)
            plt.legend(loc = 'upper left')
            i_plot += 1
        fig_name = 'Probability (%) of EQ ratio having higher discrimination power than its'
        fig_name += ' random shuffles for ' + indi_txt_key + div_txt
        plt.suptitle(fig_name)
        plt.savefig('Discrimination plot 20 stns, for '+indi_txt_key+div_txt+EQ_txt+'.png', 
                    dpi = 500)
        plt.close(fig)
        # In
        i_plot = 1
        fig = plt.figure(figsize=(20,14))
        for stn_txt in EQRs_highdisc_dic:
            EQratios = EQratios_dic[stn_txt][1:div_y-1, 1:div_y-1]
            stn_coord = eqks.position[stn_txt]
            ax = plt.subplot(4,5, i_plot)
            EQratios_plot = np.copy(EQratios)
            EQratios_plot[~EQ_pass] = np.nan
            plt.imshow(EQratios_plot, cmap="PiYG", vmin=0, vmax=1)
            for iy in range(len(y_cuts)):
                for ix in range(len(x_cuts)):
                    if ~np.isnan(EQratios_plot[iy, ix]):
                        entry = EQratios_plot[iy, ix]*100
                        if (entry > 25) & (entry < 75):
                            c = 'k'
                        else:
                            c = 'w'
                        ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                                fontsize=fs)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
            ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
                       marker = '*', color='b', s = 160, label = stn_txt)
            plt.legend(loc = 'upper left')
            i_plot += 1
        fig_name = 'EQratios (%) heatmap of EQRate_S1/EQRate_S(1+2)'
        fig_name += ' for ' + indi_txt_key + div_txt 
        plt.suptitle(fig_name)
        plt.savefig('EQratios plot 20 stns, for '+indi_txt_key+div_txt+EQ_txt+'.png', dpi = 500)
        plt.close(fig) 
    if plot_Disc == 'yes':
        i_plot = 1
        fig = plt.figure(figsize=(20,14))
        for stn_txt in EQRs_highdisc_dic:
            EQratios = EQratios_dic[stn_txt][1:div_y-1, 1:div_y-1]
            Discs = np.abs(EQratios-0.5)
            stn_coord = eqks.position[stn_txt]
            ax = plt.subplot(4,5, i_plot)
            Discs_plot = np.copy(Discs)
            Discs_plot[~EQ_pass] = np.nan
            plt.imshow(Discs_plot, cmap="Reds", vmin=0, vmax=0.5)
            for iy in range(len(y_cuts)):
                for ix in range(len(x_cuts)):
                    if ~np.isnan(Discs_plot[iy, ix]):
                        entry = Discs_plot[iy, ix]*100
                        if entry < 25:
                            c = 'k'
                        else:
                            c = 'w'
                        ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                                fontsize=fs)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
            ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
                       marker = '*', color='b', s = 160, label = stn_txt)
            plt.legend(loc = 'upper left')
            i_plot += 1
        fig_name = 'Discrimination power, or abs(EQratio-0.5) (%) heatmap of'
        fig_name += ' for ' + indi_txt_key + div_txt 
        plt.suptitle(fig_name)
        plt.savefig('Discs plot 20 stns, for '+indi_txt_key+div_txt+EQ_txt+'.png', dpi = 500)
        plt.close(fig) 
    if return_dic == 'yes':
        EQR_info = {'EQRs_highdisc_dic': EQRs_highdisc_dic,
                    'EQratios_dic': EQratios_dic,
                    'EQRs_en_disc_dic': EQRs_en_disc_dic,
                    'xticks': xticks,
                    'yticks': yticks,
                    'x_cuts': x_cuts,
                    'y_cuts': y_cuts,
                    'xtick_labels': xtick_labels,
                    'ytick_labels': ytick_labels,
                    'position': eqks.position,
                    'mag_LB': mag_LB,
                    'nEQs': nEQs,
                    'num_EQ_plot': num_EQ_plot}
        return EQR_info 
    else:
        return

def stn_shuffled_plot(stn_txt, EQR_info, min_EQ, mag_LB, div_y, indi_txt, 
                      indi_txt_key = ''):
    if len(indi_txt_key) == 0:
        indi_txt_key = indi_txt[11:-1]
    div_txt = '_minEQ%s_magLB%s_div%s'%(min_EQ, mag_LB, div_y)
    EQRs_highdisc = EQR_info['EQRs_highdisc_dic'][stn_txt]
    EQRs_en_disc =  EQR_info['EQRs_en_disc_dic'][stn_txt]
    xtick_labels = EQR_info['xtick_labels']
    ytick_labels = EQR_info['ytick_labels']
    xticks = EQR_info['xticks']
    yticks = EQR_info['yticks']
    stn_coord = EQR_info['position'][stn_txt]
    fs = np.int(6*14/(div_y-2))
    fig = plt.figure(figsize=(26,14))
    ax = plt.subplot(3,4,1)
    EQRs_highdisc = EQRs_highdisc[1:div_y-1, 1:div_y-1]
    plt.imshow(EQRs_highdisc, cmap="Reds", vmin=0, vmax=1)
    for iy in range(div_y-2):
        for ix in range(div_y-2):
            if ~np.isnan(EQRs_highdisc[iy, ix]):
                entry = EQRs_highdisc[iy, ix]*100
                if entry < 50:
                    c = 'k'
                else:
                    c = 'w'
                ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                        fontsize=fs)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
    ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
                           marker = '*', color='b', s = 160, label = stn_txt)
    plt.legend(loc = 'upper left')
    plt.ylabel('The original HS TS')
    for ien in range(11):
        EQRs_ien_disc = EQRs_en_disc[ien, :, :]
        EQRs_ien_highdisc = np.zeros((div_y, div_y))
        EQRs_ien_highdisc[:] = np.nan
        for i in range(div_y):
            for j in range(div_y):
                EQR_ien_disc = EQRs_ien_disc[i,j]
                if ~np.isnan(EQR_ien_disc):
                    EQR_en_disc = EQRs_en_disc[:, i, j]
                    EQRs_ien_highdisc[i,j] = np.sum(EQR_ien_disc > EQR_en_disc)
        EQRs_ien_highdisc = EQRs_ien_highdisc/(len(EQR_en_disc)-1)
        # plot
        ax = plt.subplot(3,4,ien+2)
        EQRs_ien_highdisc = EQRs_ien_highdisc[1:div_y-1, 1:div_y-1]
        plt.imshow(EQRs_ien_highdisc, cmap="Reds", vmin=0, vmax=1)
        for iy in range(div_y-2):
            for ix in range(div_y-2):
                if ~np.isnan(EQRs_ien_highdisc[iy, ix]):
                    entry = EQRs_ien_highdisc[iy, ix]*100
                    if entry < 50:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                            fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
        ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
                               marker = '*', color='b', s = 160, label = stn_txt)
        plt.legend(loc = 'upper left')
        plt.ylabel('Shuffled HS TS No.%s'%(ien+1))
    fig_name = 'Probability (%) of EQ ratio having higher discrimination power than its'
    fig_name += ' random shuffles for ' + indi_txt_key + div_txt + ' and its 11 shuffles'
    plt.suptitle(fig_name)
    EQ_txt = '_minEQ%s_mag%s'%(min_EQ, mag_LB)
    plt.savefig('Shuffled EQR_disc plots for '+stn_txt+'_'+indi_txt_key+div_txt+EQ_txt+'.png',
                dpi = 500)
    plt.close(fig) 
    return

def stn_shuffled_plot_Disc(stn_txt, EQR_info, min_EQ, mag_LB, div_y, indi_txt, indi_txt_key = ''):
    if len(indi_txt_key) == 0:
        indi_txt_key = indi_txt[11:-1]
    div_txt = '_minEQ%s_magLB%s_div%s'%(min_EQ, mag_LB, div_y)
    EQratios = EQR_info['EQratios_dic'][stn_txt]
    Discs = np.abs(EQratios-0.5)
    EQRs_en_disc =  EQR_info['EQRs_en_disc_dic'][stn_txt]
    xtick_labels = EQR_info['xtick_labels']
    ytick_labels = EQR_info['ytick_labels']
    xticks = EQR_info['xticks']
    yticks = EQR_info['yticks']
    stn_coord = EQR_info['position'][stn_txt]
    fs = np.int(6*14/(div_y-2))
    fig = plt.figure(figsize=(26,14))
    ax = plt.subplot(3,4,1)
    Discs = Discs[1:div_y-1, 1:div_y-1]
    plt.imshow(Discs, cmap="Reds", vmin=0, vmax=0.5)
    for iy in range(div_y-2):
        for ix in range(div_y-2):
            if ~np.isnan(Discs[iy, ix]):
                entry = Discs[iy, ix]*100
                if entry < 25:
                    c = 'k'
                else:
                    c = 'w'
                ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                        fontsize=fs)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
    ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
                           marker = '*', color='b', s = 160, label = stn_txt)
    plt.legend(loc = 'upper left')
    plt.ylabel('The original HS TS')
    for ien in range(11):
        EQRs_ien_disc = EQRs_en_disc[ien, :, :]
        # plot
        ax = plt.subplot(3,4,ien+2)
        EQRs_ien_disc = EQRs_ien_disc[1:div_y-1, 1:div_y-1]
        plt.imshow(EQRs_ien_disc, cmap="Reds", vmin=0, vmax=0.5)
        for iy in range(div_y-2):
            for ix in range(div_y-2):
                if ~np.isnan(EQRs_ien_disc[iy, ix]):
                    entry = EQRs_ien_disc[iy, ix]*100
                    if entry < 25:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                            fontsize=fs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
        ax.scatter(bm.get_tick(stn_coord[1],xtick_labels),bm.get_tick(stn_coord[0],ytick_labels), 
                               marker = '*', color='b', s = 160, label = stn_txt)
        plt.legend(loc = 'upper left')
        plt.ylabel('Shuffled HS TS No.%s'%(ien+1))
    fig_name = 'Discrimination power, or abs(EQratio-0.5) (%) heatmap of '
    fig_name += indi_txt_key + div_txt + ' and its 11 shuffles'
    plt.suptitle(fig_name)
    EQ_txt = '_minEQ%s_mag%s'%(min_EQ, mag_LB)
    plt.savefig('Shuffled EQR_disc (Disc) plots for '+stn_txt+'_'+indi_txt_key+div_txt+EQ_txt+'.png',
                dpi = 500)
    plt.close(fig) 
    return


def get_LocEQs(stn_txt='DAHU', LocRad=1.5, div_y=12, min_EQ=20):
    eqks = mB.eqk()
    eqks.makeGM(div_y = div_y, div_x = 0)
    x_cuts = eqks.x_cuts
    y_cuts = eqks.y_cuts
    stn_coord = eqks.position[stn_txt]
    x_ctrs, y_ctrs = np.zeros((div_y, )), np.zeros((div_y, ))
    for i in range(div_y):
        x_ctrs[i] = np.mean([x_cuts[i], x_cuts[i+1]])
        y_ctrs[i] = np.mean([y_cuts[i], y_cuts[i+1]])
    is_Loc = np.zeros((div_y, div_y), dtype = bool) 
    for ix in range(div_y):
        for iy in range(div_y):
            distance = np.sqrt((x_ctrs[ix] - stn_coord[1])**2 + (y_ctrs[iy] - stn_coord[0])**2)
            if distance < LocRad:
                is_Loc[iy, ix] = 1
    is_Loc = np.flipud(is_Loc)
    numEQs = np.zeros((div_y, div_y))
    for ix in range(div_y):
        for iy in range(div_y):
            GMselected = eqks.GM_select(iy, ix)
            numEQs[iy,ix] = GMselected.shape[0]
    numEQs = np.flipud(numEQs)
    LocEQs = numEQs.copy()
    for iy in range(div_y):
        for ix in range(div_y):
            if ~is_Loc[iy, ix]:
                LocEQs[iy, ix] = 0
    return LocEQs

# In
def plot_LocDiscScore_hist(EQR_info, LocRad=1.5, div_y=16, min_EQ = 20, 
                           indi_txt = '_cV_BW0813_CVSK_', indi_txt_key = ''):
    if 'mag_LB' in EQR_info:
        mag_LB = EQR_info['mag_LB']
    else:
        mag_LB = 3
    if len(indi_txt_key) == 0:
        indi_txt_key = indi_txt[11:-1]
    i_plot = 1
    fig = plt.figure(figsize=(26,14))
    num_below20pct = 0
    num_below50pct = 0
    rate_mean_prod_tops = []
    for stn_txt in EQR_info['EQratios_dic']:
        LocEQs = get_LocEQs(stn_txt=stn_txt, LocRad=LocRad, div_y=div_y, min_EQ=min_EQ)
        EQRs_en_disc_dic = EQR_info['EQRs_en_disc_dic']
        EQRs_highdisc_dic = EQR_info['EQRs_highdisc_dic']
        EQRs_highdisc = EQRs_highdisc_dic[stn_txt]
        EQRs_en_disc = EQRs_en_disc_dic[stn_txt]
        #
        EQRs_en_highdisc = np.zeros((EQRs_en_disc.shape[0], div_y, div_y))
        for ien in range(EQRs_en_disc.shape[0]):
            EQRs_ien_disc = EQRs_en_disc[ien, :, :]
            EQRs_ien_highdisc = np.zeros((div_y, div_y))
            EQRs_ien_highdisc[:] = np.nan
            for i in range(div_y):
                for j in range(div_y):
                    EQR_ien_disc = EQRs_ien_disc[i,j]
                    if ~np.isnan(EQR_ien_disc):
                        EQR_en_disc = EQRs_en_disc[:, i, j]
                        EQRs_ien_highdisc[i,j] = np.sum(EQR_ien_disc > EQR_en_disc)
            EQRs_ien_highdisc = EQRs_ien_highdisc/(len(EQR_en_disc)-1)
            EQRs_en_highdisc[ien, :, :] = EQRs_ien_highdisc
        #
        products = EQRs_highdisc*LocEQs
        mean_prods = np.mean(products[products>0])
        mean_prods_en = np.zeros((EQRs_en_disc.shape[0], ))
        for i in range(EQRs_en_highdisc.shape[0]):
            products = EQRs_en_highdisc[i,:,:]*LocEQs
            mean_prods_en[i] = np.mean(products[products>0])
        # ploting
        plt.subplot(4,5,i_plot)
        sns.distplot(mean_prods_en)
        rate_mean_prod_top = np.sum(mean_prods_en > mean_prods)/EQRs_en_disc.shape[0]
        rate_mean_prod_tops.append(rate_mean_prod_top)
        if rate_mean_prod_top < 0.2:
            num_below20pct += 1
        if rate_mean_prod_top < 0.5:
            num_below50pct += 1
        txt = 'Station: %.1f (top %.1f'%(mean_prods, rate_mean_prod_top*100)
        txt += '%)'
        plt.axvline(x=mean_prods, color = 'r', label = txt)
        plt.legend(loc = 'upper left')
        plt.title('Station ' + stn_txt)
        if i_plot > 15:
            plt.xlabel('LocDiscScore')
        if i_plot in [1, 6, 11, 16]:
            plt.ylabel('Frequency')
        i_plot += 1
    title_txt = 'LocDiscScore of 20 stations (red vertical line) and their %s '%EQRs_en_highdisc.shape[0]
    title_txt += 'random-shuffled HS TSs (blue histogram), with LocRad=%s, div_y=%s'%(LocRad, div_y)
    title_txt += ', indicators=' + indi_txt_key
    title_txt += '\n(%s stns within top 20 pct, '%num_below20pct
    title_txt += '%s stns within top 50 pct, '%num_below50pct
    title_txt += 'mean=%.4f)'%np.mean(rate_mean_prod_tops)
    plt.suptitle(title_txt)
    EQ_txt = '_minEQ%s_mag%s'%(min_EQ, mag_LB)
    plt.savefig('LocDiscScore_hist_LocRad%s_div%s'%(LocRad, div_y) +'_'+indi_txt_key+EQ_txt+'.png',
                dpi = 500)
    plt.close(fig)
    return


def plot_LocDiscScore_hist_new(EQR_info, LocRad=1.5, div_y=16, min_EQ = 20, 
                               indi_txt = '_cV_BW0813_CVSK_', indi_txt_key = ''):
    if 'mag_LB' in EQR_info:
        mag_LB = EQR_info['mag_LB']
    else:
        mag_LB = 3
    if len(indi_txt_key) == 0:
        indi_txt_key = indi_txt[11:-1]
    i_plot = 1
    fig = plt.figure(figsize=(26,14))
    num_below20pct = 0
    num_below50pct = 0
    rate_mean_prod_tops = []
    for stn_txt in EQR_info['EQratios_dic']:
        LocEQs = get_LocEQs(stn_txt=stn_txt, LocRad=LocRad, div_y=div_y, min_EQ=min_EQ)
        EQRs_en_disc_dic = EQR_info['EQRs_en_disc_dic']
        EQRs_highdisc_dic = EQR_info['EQRs_highdisc_dic']
        EQRs_highdisc = EQRs_highdisc_dic[stn_txt]
        EQRs_en_disc = EQRs_en_disc_dic[stn_txt]
        #
        EQRs_en_highdisc = np.zeros((EQRs_en_disc.shape[0], div_y, div_y))
        for ien in range(EQRs_en_disc.shape[0]):
            EQRs_ien_disc = EQRs_en_disc[ien, :, :]
            EQRs_ien_highdisc = np.zeros((div_y, div_y))
            EQRs_ien_highdisc[:] = np.nan
            for i in range(div_y):
                for j in range(div_y):
                    EQR_ien_disc = EQRs_ien_disc[i,j]
                    if ~np.isnan(EQR_ien_disc):
                        EQR_en_disc = EQRs_en_disc[:, i, j]
                        EQRs_ien_highdisc[i,j] = np.sum(EQR_ien_disc > EQR_en_disc)
            EQRs_ien_highdisc = EQRs_ien_highdisc/(len(EQR_en_disc)-1)
            EQRs_en_highdisc[ien, :, :] = EQRs_ien_highdisc
        #
        WLDRScore = np.nansum(EQRs_highdisc*LocEQs)/np.sum(LocEQs)
        mean_prods = WLDRScore
        mean_prods_en = np.zeros((EQRs_en_disc.shape[0], ))
        for i in range(EQRs_en_highdisc.shape[0]):
            WLDRScore = np.nansum(EQRs_en_highdisc[i,:,:]*LocEQs)/np.sum(LocEQs)
            mean_prods_en[i] = WLDRScore
        # ploting
        plt.subplot(4,5,i_plot)
        sns.distplot(mean_prods_en)
        rate_mean_prod_top = np.sum(mean_prods_en > mean_prods)/EQRs_en_disc.shape[0]
        rate_mean_prod_tops.append(rate_mean_prod_top)
        if rate_mean_prod_top < 0.2:
            num_below20pct += 1
        if rate_mean_prod_top < 0.5:
            num_below50pct += 1
        txt = 'Station: %.1f (top %.1f'%(mean_prods, rate_mean_prod_top*100)
        txt += '%)'
        plt.axvline(x=mean_prods, color = 'r', label = txt)
        plt.legend(loc = 'upper left')
        plt.title('Station ' + stn_txt)
        if i_plot > 15:
            plt.xlabel('LocDiscScore')
        if i_plot in [1, 6, 11, 16]:
            plt.ylabel('Frequency')
        i_plot += 1
    title_txt = 'LocDiscScore of 20 stations (red vertical line) and their %s '%EQRs_en_highdisc.shape[0]
    title_txt += 'random-shuffled HS TSs (blue histogram), with LocRad=%s, div_y=%s'%(LocRad, div_y)
    title_txt += ', indicators=' + indi_txt_key
    title_txt += '\n(%s stns within top 20 pct, '%num_below20pct
    title_txt += '%s stns within top 50 pct, '%num_below50pct
    title_txt += 'mean=%.4f)'%np.mean(rate_mean_prod_tops)
    plt.suptitle(title_txt)
    EQ_txt = '_minEQ%s_mag%s'%(min_EQ, mag_LB)
    plt.savefig('NewLocDiscScore_hist_LocRad%s_div%s'%(LocRad, div_y) +'_'+indi_txt_key+EQ_txt+'.png',
                dpi = 500)
    plt.close(fig)
    return

def plot_OptiLocDiscScore_hist(EQR_info, LocRads=np.arange(0.8, 3.0, 0.1), div_y=16, 
                               min_EQ = 20, indi_txt = '_cV_BW0813_CVSK_', indi_txt_key = '',
                               save_endtxt = '', save_extratxt = '', plot = 'yes', 
                               return_dic = 'no', shift = 0, save_dic = 'no',
                               paratxt = '_CVSKXXXLXX'):
    LocRads = np.round(LocRads*10)/10
    if 'BW' in indi_txt:
        if 'NcV' in indi_txt:
            if len(indi_txt_key) == 0:
                indi_txt_key = indi_txt[12:-1]
        else:
            if len(indi_txt_key) == 0:
                indi_txt_key = indi_txt[11:-1] # for BW0813 data
    else:
        if len(save_endtxt) == 0: # if no specifically input
            save_endtxt = '_UFD' # for un-filtered data
    if 'NcV' in indi_txt:
        save_endtxt += '_NcV'
    if 'trV' in indi_txt:
        save_endtxt += '_trV'
    save_endtxt += save_extratxt
    if plot == 'yes':
        fig = plt.figure(figsize=(26,14))
    stns_dic = dict()
    i_plot = 1
    stn_txts = []
    if 'mag_LB' in EQR_info:
        mag_LB = EQR_info['mag_LB']
    else:
        mag_LB = 3
    for stn_txt in EQR_info['EQratios_dic']:
        stn_txts.append(stn_txt)
        rate_mean_prod_tops_stn = np.zeros((len(LocRads), ))
        mean_prods_ens_stn = []
        mean_prods_stn = []
        for i_LR in range(len(LocRads)):
            LocRad = LocRads[i_LR]
            LocEQs = get_LocEQs(stn_txt=stn_txt, LocRad=LocRad, div_y=div_y, min_EQ=min_EQ)
            EQRs_en_disc_dic = EQR_info['EQRs_en_disc_dic']
            EQRs_highdisc_dic = EQR_info['EQRs_highdisc_dic']
            EQRs_highdisc = EQRs_highdisc_dic[stn_txt]
            EQRs_en_disc = EQRs_en_disc_dic[stn_txt]
            #
            EQRs_en_highdisc = np.zeros((EQRs_en_disc.shape[0], div_y, div_y))
            for ien in range(EQRs_en_disc.shape[0]):
                EQRs_ien_disc = EQRs_en_disc[ien, :, :]
                EQRs_ien_highdisc = np.zeros((div_y, div_y))
                EQRs_ien_highdisc[:] = np.nan
                for i in range(div_y):
                    for j in range(div_y):
                        EQR_ien_disc = EQRs_ien_disc[i,j]
                        if ~np.isnan(EQR_ien_disc):
                            EQR_en_disc = EQRs_en_disc[:, i, j]
                            EQRs_ien_highdisc[i,j] = np.sum(EQR_ien_disc > EQR_en_disc)
                EQRs_ien_highdisc = EQRs_ien_highdisc/(len(EQR_en_disc)-1)
                EQRs_en_highdisc[ien, :, :] = EQRs_ien_highdisc
            #
            products = EQRs_highdisc*LocEQs
            mean_prods = np.mean(products[products>0])
            mean_prods_stn.append(mean_prods)
            mean_prods_en = np.zeros((EQRs_en_disc.shape[0], ))
            for i in range(EQRs_en_highdisc.shape[0]):
                products = EQRs_en_highdisc[i,:,:]*LocEQs
                mean_prods_en[i] = np.mean(products[products>0])
            mean_prods_ens_stn.append(mean_prods_en)
            rate_mean_prod_top = np.sum(mean_prods_en > mean_prods)/EQRs_en_disc.shape[0]
            rate_mean_prod_tops_stn[i_LR] = rate_mean_prod_top
        if plot == 'yes':
            plt.subplot(4,5,i_plot)
            plt.plot(LocRads, rate_mean_prod_tops_stn)
            plt.title('Station ' + stn_txt)
            if i_plot > 15:
                plt.xlabel('LocRad')
            if i_plot in [1, 6, 11, 16]:
                plt.ylabel('Confidence Level')
            i_plot += 1
        ind_min = np.argmin(rate_mean_prod_tops_stn)
        stns_dic[stn_txt] = dict()
        stns_dic[stn_txt]['rate_mean_prod_tops_stn'] = rate_mean_prod_tops_stn
        stns_dic[stn_txt]['rate_mean_prod_top'] = rate_mean_prod_tops_stn[ind_min]
        stns_dic[stn_txt]['mean_prods_en'] = mean_prods_ens_stn[ind_min]
        stns_dic[stn_txt]['mean_prods'] = mean_prods_stn[ind_min]
        stns_dic[stn_txt]['LocRad'] = LocRads[ind_min]
        stns_dic['_LocRads'] = LocRads
        stns_dic['_indi_txt'] = indi_txt
        stns_dic['_div_y'] = div_y
        stns_dic['_min_EQ'] = min_EQ
        stns_dic['_indi_txt_key'] = indi_txt_key
        stns_dic['_save_endtxt'] = save_endtxt
        stns_dic['_stn_txts'] = stn_txts
        stns_dic['_mag_LB'] = mag_LB
    EQ_txt = '_minEQ%s_mag%s'%(min_EQ, mag_LB)
    if shift != 0:
        EQ_txt += '_shift%s'%shift
    EQ_txt += paratxt
    if plot == 'yes':
        title_txt = "Stations's Confidence Level (compared with their their "
        title_txt += '%s random-shuffled HS TSs) at different LocRads '%EQRs_en_disc.shape[0]
        title_txt += "with div_y=%s, indicators="%div_y+ indi_txt_key
        plt.suptitle(title_txt)
        plt.savefig('ConfidenceVsLocRad_div%s'%(div_y) +'_'+indi_txt_key+save_endtxt+EQ_txt+'.png', 
                    dpi = 500)
        plt.close(fig)
        # next plot
        fig = plt.figure(figsize=(26,14))
        rate_mean_prod_tops = []
        i_plot = 1
        num_below20pct = 0
        num_below50pct = 0
        for stn_txt in EQR_info['EQratios_dic']:
            rate_mean_prod_top = stns_dic[stn_txt]['rate_mean_prod_top']
            mean_prods_en = stns_dic[stn_txt]['mean_prods_en']
            mean_prods = stns_dic[stn_txt]['mean_prods']
            plt.subplot(4,5,i_plot)
            sns.distplot(mean_prods_en)
            rate_mean_prod_top = np.sum(mean_prods_en > mean_prods)/EQRs_en_disc.shape[0]
            rate_mean_prod_tops.append(rate_mean_prod_top)
            if rate_mean_prod_top < 0.2:
                num_below20pct += 1
            if rate_mean_prod_top < 0.5:
                num_below50pct += 1
            txt = 'Station: %.1f (top %.1f'%(mean_prods, rate_mean_prod_top*100)
            txt += '%)'
            plt.axvline(x=mean_prods, color = 'r', label = txt)
            plt.legend(loc = 'upper left')
            plt.title('Station ' + stn_txt + ' LocRad=%s'%stns_dic[stn_txt]['LocRad'])
            if i_plot > 15:
                plt.xlabel('LocDiscScore')
            if i_plot in [1, 6, 11, 16]:
                plt.ylabel('Frequency')
            i_plot += 1
        title_txt = 'LocDiscScore of 20 stations (red vertical line) and their' 
        title_txt += '%s random-shuffled HS TSs (blue histogram) with'%EQRs_en_disc.shape[0]
        title_txt += ' optimal LocRad, div_y=%s, indicators='%div_y 
        title_txt += indi_txt_key + '\n(%s stns within top 20 pct, '%num_below20pct
        title_txt += '%s stns within top 50 pct, '%num_below50pct
        title_txt += 'mean=%.4f)'%np.mean(rate_mean_prod_tops)
        plt.suptitle(title_txt)
        save_txt = 'OptimalLocDiscScore_hist_div%s'%(div_y) +'_'+indi_txt_key+save_endtxt+EQ_txt
        plt.savefig(save_txt+'.png',dpi = 500)
        plt.close(fig)
    if save_dic == 'yes':
        with open('OptimalLocRadsDicts_div%s'%(div_y)+'_'+indi_txt_key+save_endtxt+EQ_txt+'.pickle', 
                  'wb') as f:
            pickle.dump(stns_dic, f)
    if return_dic == 'yes':
        return stns_dic
    else:
        return


def plot_OptiLocDiscScore_hist_new(EQR_info, LocRads=np.arange(0.8, 3.0, 0.05), div_y=16, 
                                   min_EQ = 20, indi_txt = '_cV_BW0813_CVSK_', indi_txt_key = '',
                                   save_endtxt = '', save_extratxt = '', plot = 'yes', 
                                   return_dic = 'no', shift = 0, save_dic = 'no', 
                                   paratxt = '_CVSKXXXLXX'):
    LocRads = np.round(LocRads*10)/10
    if 'BW' in indi_txt:
        if 'NcV' in indi_txt:
            if len(indi_txt_key) == 0:
                indi_txt_key = indi_txt[12:-1]
        else:
            if len(indi_txt_key) == 0:
                indi_txt_key = indi_txt[11:-1] # for BW0813 data
    else:
        if len(save_endtxt) == 0: # if no specifically input
            save_endtxt = '_UFD' # for un-filtered data
    if 'NcV' in indi_txt:
        save_endtxt += '_NcV'
    if 'trV' in indi_txt:
        save_endtxt += '_trV'
    save_endtxt += save_extratxt
    if plot == 'yes':
        fig = plt.figure(figsize=(26,14))
    stns_dic = dict()
    i_plot = 1
    stn_txts = []
    if 'mag_LB' in EQR_info:
        mag_LB = EQR_info['mag_LB']
    else:
        mag_LB = 3
    for stn_txt in EQR_info['EQratios_dic']:
        stn_txts.append(stn_txt)
        rate_mean_prod_tops_stn = np.zeros((len(LocRads), ))
        mean_prods_ens_stn = []
        mean_prods_stn = []
        for i_LR in range(len(LocRads)):
            LocRad = LocRads[i_LR]
            LocEQs = get_LocEQs(stn_txt=stn_txt, LocRad=LocRad, div_y=div_y, min_EQ=min_EQ)
            EQRs_en_disc_dic = EQR_info['EQRs_en_disc_dic']
            EQRs_highdisc_dic = EQR_info['EQRs_highdisc_dic']
            EQRs_highdisc = EQRs_highdisc_dic[stn_txt]
            EQRs_en_disc = EQRs_en_disc_dic[stn_txt]
            #
            EQRs_en_highdisc = np.zeros((EQRs_en_disc.shape[0], div_y, div_y))
            for ien in range(EQRs_en_disc.shape[0]):
                EQRs_ien_disc = EQRs_en_disc[ien, :, :]
                EQRs_ien_highdisc = np.zeros((div_y, div_y))
                EQRs_ien_highdisc[:] = np.nan
                for i in range(div_y):
                    for j in range(div_y):
                        EQR_ien_disc = EQRs_ien_disc[i,j]
                        if ~np.isnan(EQR_ien_disc):
                            EQR_en_disc = EQRs_en_disc[:, i, j]
                            EQRs_ien_highdisc[i,j] = np.sum(EQR_ien_disc > EQR_en_disc)
                EQRs_ien_highdisc = EQRs_ien_highdisc/(len(EQR_en_disc)-1)
                EQRs_en_highdisc[ien, :, :] = EQRs_ien_highdisc
            #
            WLDRScore = np.nansum(EQRs_highdisc*LocEQs)/np.sum(LocEQs)
            mean_prods = WLDRScore
            mean_prods_stn.append(mean_prods)
            mean_prods_en = np.zeros((EQRs_en_disc.shape[0], ))
            for i in range(EQRs_en_highdisc.shape[0]):
                WLDRScore = np.nansum(EQRs_en_highdisc[i,:,:]*LocEQs)/np.sum(LocEQs)
                mean_prods_en[i] = WLDRScore
            mean_prods_ens_stn.append(mean_prods_en)
            rate_mean_prod_top = np.sum(mean_prods_en > mean_prods)/EQRs_en_disc.shape[0]
            rate_mean_prod_tops_stn[i_LR] = rate_mean_prod_top
        if plot == 'yes':
            plt.subplot(4,5,i_plot)
            plt.plot(LocRads, rate_mean_prod_tops_stn)
            plt.title('Station ' + stn_txt)
            if i_plot > 15:
                plt.xlabel('R')
            if i_plot in [1, 6, 11, 16]:
                plt.ylabel('p-value')
            i_plot += 1
        ind_min = np.argmin(rate_mean_prod_tops_stn)
        stns_dic[stn_txt] = dict()
        stns_dic[stn_txt]['rate_mean_prod_tops_stn'] = rate_mean_prod_tops_stn
        stns_dic[stn_txt]['rate_mean_prod_top'] = rate_mean_prod_tops_stn[ind_min]
        stns_dic[stn_txt]['mean_prods_en'] = mean_prods_ens_stn[ind_min]
        stns_dic[stn_txt]['mean_prods'] = mean_prods_stn[ind_min]
        stns_dic[stn_txt]['LocRad'] = LocRads[ind_min]
        stns_dic['_LocRads'] = LocRads
        stns_dic['_indi_txt'] = indi_txt
        stns_dic['_div_y'] = div_y
        stns_dic['_min_EQ'] = min_EQ
        stns_dic['_indi_txt_key'] = indi_txt_key
        stns_dic['_save_endtxt'] = save_endtxt
        stns_dic['_stn_txts'] = stn_txts
        stns_dic['_mag_LB'] = mag_LB
    EQ_txt = '_minEQ%s_mag%s'%(min_EQ, mag_LB)
    if shift != 0:
        EQ_txt += '_shift%s'%shift
    EQ_txt += paratxt
    if plot == 'yes':
        title_txt = "Stations's Confidence Level (compared with their their "
        title_txt += '%s random-shuffled HS TSs) at different LocRads '%EQRs_en_disc.shape[0]
        title_txt += "with div_y=%s, indicators="%div_y+ indi_txt_key
        plt.suptitle(title_txt)
        plt.savefig('NewConfidenceVsLocRad_div%s'%(div_y) +'_'+indi_txt_key+save_endtxt+EQ_txt+'.png', 
                    dpi = 500)
        plt.close(fig)
        # next plot
        fig = plt.figure(figsize=(26,14))
        rate_mean_prod_tops = []
        i_plot = 1
        num_below20pct = 0
        num_below50pct = 0
        for stn_txt in EQR_info['EQratios_dic']:
            rate_mean_prod_top = stns_dic[stn_txt]['rate_mean_prod_top']
            mean_prods_en = stns_dic[stn_txt]['mean_prods_en']
            mean_prods = stns_dic[stn_txt]['mean_prods']
            plt.subplot(4,5,i_plot)
            sns.distplot(mean_prods_en)
            rate_mean_prod_top = np.sum(mean_prods_en > mean_prods)/EQRs_en_disc.shape[0]
            rate_mean_prod_tops.append(rate_mean_prod_top)
            if rate_mean_prod_top < 0.2:
                num_below20pct += 1
            if rate_mean_prod_top < 0.5:
                num_below50pct += 1
            txt = 'Original TS: %.1f (p=%.1f'%(mean_prods, rate_mean_prod_top*100)
            txt += '%)'
            plt.axvline(x=mean_prods, color = 'r', label = txt)
            plt.legend(loc = 'upper left')
            plt.title('Station ' + stn_txt + ', at optimal R=%s'%stns_dic[stn_txt]['LocRad'])
            if i_plot > 15:
                plt.xlabel('WLDR Score')
            if i_plot in [1, 6, 11, 16]:
                plt.ylabel('Frequency')
            i_plot += 1
        title_txt = 'LocDiscScore of 20 stations (red vertical line) and their' 
        title_txt += '%s random-shuffled HS TSs (blue histogram) with'%EQRs_en_disc.shape[0]
        title_txt += ' optimal LocRad, div_y=%s, indicators='%div_y 
        title_txt += indi_txt_key + '\n(%s stns within top 20 pct, '%num_below20pct
        title_txt += '%s stns within top 50 pct, '%num_below50pct
        title_txt += 'mean=%.4f)'%np.mean(rate_mean_prod_tops)
        plt.suptitle(title_txt)
        save_txt = 'NewOptimalLocDiscScore_hist_div%s'%(div_y) +'_'+indi_txt_key+save_endtxt+EQ_txt
        plt.savefig(save_txt+'.png',dpi = 500)
        plt.close(fig)
    if save_dic == 'yes':
        with open('NewOptimalLocRadsDicts_div%s'%(div_y)+'_'+indi_txt_key+save_endtxt+EQ_txt+'.pickle', 
                  'wb') as f:
            pickle.dump(stns_dic, f)
    if return_dic == 'yes':
        return stns_dic
    else:
        return

def plot_OptiLocDiscScore_hist_fromDic(dic_name, new = 'no'):
    with open(dic_name, 'rb') as f:
        stns_dic = pickle.load(f)
    if 'New' in dic_name:
        new = 'yes'
    LocRads = stns_dic['_LocRads']
    div_y = stns_dic['_div_y']
    indi_txt_key = stns_dic['_indi_txt_key']
    save_endtxt = stns_dic['_save_endtxt']
    stn_txts = stns_dic['_stn_txts']
    try:
        EQ_txt = '_minEQ%s_mag%s'%(stns_dic['_min_EQ'], stns_dic['_mag_LB'])
    except:
        print('old version of stns_dic')
        EQ_txt = '_minEQ20_mag3'
    fig = plt.figure(figsize=(26,14))
    i_plot = 1
    for stn_txt in stn_txts:
        # unpack
        rate_mean_prod_tops_stn = stns_dic[stn_txt]['rate_mean_prod_tops_stn']
        confidences = 1 - np.array(rate_mean_prod_tops_stn)
        plt.subplot(4,5,i_plot)
        plt.plot(LocRads, confidences)
        plt.ylim(0,1)
        plt.title('Station ' + stn_txt)
        if i_plot > 15:
            plt.xlabel('LocRad')
        if i_plot in [1, 6, 11, 16]:
            plt.ylabel('Confidence Level')
        i_plot += 1
    num_shuffles = len(stns_dic[stn_txt]['mean_prods_en'])
    title_txt = "Stations's Confidence Level (compared with their their %s "%num_shuffles
    title_txt += "random-shuffled HS TSs) at different LocRads with div_y=%s"%div_y
    title_txt += ', indicators=' + indi_txt_key + ' EQs: ' + EQ_txt[1:]
    plt.suptitle(title_txt)
#    plt.savefig('ConfidenceVsLocRad_div%s'%(div_y) +'_'+indi_txt_key+save_endtxt+EQ_txt+'.png', 
#                dpi = 500)
    plt.savefig('ConfidenceVsLocRad'+dic_name[19:-7]+'.png', dpi = 500)
    plt.close(fig)
    # next plot
    fig = plt.figure(figsize=(26,14))
    rate_mean_prod_tops = []
    i_plot = 1
    num_below20pct = 0
    num_below50pct = 0
    for stn_txt in stn_txts:
        rate_mean_prod_top = stns_dic[stn_txt]['rate_mean_prod_top']
        mean_prods_en = stns_dic[stn_txt]['mean_prods_en']
        mean_prods = stns_dic[stn_txt]['mean_prods']
        plt.subplot(4,5,i_plot)
        sns.distplot(mean_prods_en)
        rate_mean_prod_top = np.sum(mean_prods_en > mean_prods)/num_shuffles
        rate_mean_prod_tops.append(rate_mean_prod_top)
        confidence_level = 1 - rate_mean_prod_top
        confidence_levels = 1 - np.array(rate_mean_prod_tops)
        if rate_mean_prod_top < 0.2:
            num_below20pct += 1
        if rate_mean_prod_top < 0.5:
            num_below50pct += 1
        txt = 'Station: %.1f%% confidence'%(confidence_level*100)
        plt.axvline(x=mean_prods, color = 'r', label = txt)
        plt.legend(loc = 'upper left')
        plt.title('Station ' + stn_txt + ' LocRad=%s'%stns_dic[stn_txt]['LocRad'])
        if i_plot > 15:
            plt.xlabel('LocDiscScore')
        if i_plot in [1, 6, 11, 16]:
            plt.ylabel('Frequency')
        i_plot += 1
    title_txt = 'LocDiscScore of 20 stations (red vertical line) and their ' 
    title_txt += '%s random-shuffled HS TSs (blue histogram) with'%num_shuffles
    title_txt += ' optimal LocRad, div_y=%s, indicators='%div_y 
    title_txt += indi_txt_key + ' EQs: ' + EQ_txt[1:]
    title_txt += '\n(%s stns above 80%% confidence, '%num_below20pct
    title_txt += '%s stns above 50%% confidence, '%num_below50pct
    title_txt += 'average=%.1f%%)'%np.mean(confidence_levels*100)
    plt.suptitle(title_txt)
#    plt.savefig('OptimalLocDiscScore_hist_div%s'%(div_y) +'_'+indi_txt_key+save_endtxt+EQ_txt+'.png',
#                dpi = 500)
    if new == 'yes':
        txt_new = 'New'
    else:
        txt_new = ''
    plt.savefig(txt_new+'OptimalLocDiscScore_hist'+dic_name[19:-7]+'.png', dpi = 500)
    plt.close(fig)
    return

def plot_ConfidenceCompare(dic_names, save = 'no', new = 'no'):
    stns_dics = []
    for dic_name in dic_names:
        with open(dic_name, 'rb') as f:
            stns_dic = pickle.load(f)
        stns_dics.append(stns_dic)
    if 'New' in dic_name:
        new = 'yes'
    stn_txts = stns_dic['_stn_txts']
    LocRads = stns_dic['_LocRads']
    fig = plt.figure(figsize=(26,14))
    i_plot = 1
    title_txt = "Comparison plots of 20 stations' Confidence Level vs LocRads "
    title_txt += "(div%s), for datasets: "%stns_dic['_div_y']
    save_txt = 'ComparePlots_ConfidenceVsLocRad_div%s'%stns_dic['_div_y']
    i_dic = 0
    for stn_txt in stn_txts:
        plt.subplot(4,5,i_plot)
        for stns_dic in stns_dics:
            try:
                EQ_txt = '_minEQ%s_mag%s'%(stns_dic['_min_EQ'], stns_dic['_mag_LB'])
            except:
                print('old version of stns_dic')
                EQ_txt = ''
            rate_mean_prod_tops_stn = stns_dic[stn_txt]['rate_mean_prod_tops_stn']
            confidences_stn = 1 - np.array(rate_mean_prod_tops_stn)
            indi_txt_key = stns_dic['_indi_txt_key']
            save_endtxt = stns_dic['_save_endtxt'] + '_div%s_minEQ%s'%(stns_dic['_div_y'],
                                  stns_dic['_min_EQ'])
            if i_plot == 1:
                plt.plot(LocRads, confidences_stn, label = dic_names[i_dic][20:-7])
                i_dic += 1
#                old: label = indi_txt_key + save_endtxt + EQ_txt
            else:
                plt.plot(LocRads, confidences_stn)
            plt.ylim(0,1)
            plt.title('Station ' + stn_txt)
            if i_plot > 15:
                plt.xlabel('LocRad')
            if i_plot in [1, 6, 11, 16]: 
                plt.ylabel('Confidence Level')
            if i_plot == 1:
                title_txt += indi_txt_key + save_endtxt + ', '
                save_txt += '_' + indi_txt_key + save_endtxt + EQ_txt
        if i_plot == 1:
            plt.legend()
        i_plot += 1
    plt.suptitle(title_txt[:-2])
    if save == 'yes':
        if new == 'yes':
            txt_new = 'New'
        else:
            txt_new = ''
        plt.savefig(txt_new+save_txt+'.png', dpi = 500)
        plt.close(fig)
    return


def plot_NewConfidenceCompare(dic_names, save = 'no', topAOC_rate = 0.2, new = 'no'):
    stns_dics = []
    for dic_name in dic_names:
        with open(dic_name, 'rb') as f:
            stns_dic = pickle.load(f)
        stns_dics.append(stns_dic)
    if 'New' in dic_name:
        new = 'yes'
    stn_txts = stns_dic['_stn_txts']
    LocRads = stns_dic['_LocRads']
    fig = plt.figure(figsize=(26,14))
    i_plot = 1
    title_txt = "Comparison plots of 20 stations' Confidence Level vs LocRads with "
    title_txt += 'topAUCrate=%s for datasets: '%topAOC_rate
    save_txt = 'NewComparePlots_ConfidenceVsLocRad_tAUCr%s'%topAOC_rate
    i_dic = 0
    AOCScores_StnByData = np.zeros((20,len(stns_dics)))
    for stn_txt in stn_txts:
        plt.subplot(4,5,i_plot)
        i_data = -1
        for stns_dic in stns_dics:
            i_data += 1
            rate_mean_prod_tops_stn = stns_dic[stn_txt]['rate_mean_prod_tops_stn']
            confidences_stn = 1 - np.array(rate_mean_prod_tops_stn)
            confidences_stn_sorted = np.sort(confidences_stn)
            AOC_ind_start = int(len(confidences_stn)*(1-topAOC_rate))
            AOCScore_stn = np.mean(confidences_stn_sorted[AOC_ind_start:])
            AOCScores_StnByData[i_plot-1, i_data] = AOCScore_stn
            AOCS_txt = 'AUCS=%.2f'%AOCScore_stn
            if i_plot == 1:
                plt.plot(LocRads, confidences_stn, 
                         label = dic_names[i_dic][20:-7] + '_' + AOCS_txt)
                i_dic += 1
            else:
                plt.plot(LocRads, confidences_stn, label = AOCS_txt)
            plt.ylim(0,1)
            plt.title('Station ' + stn_txt)
            if i_plot > 15:
                plt.xlabel('LocRad')
            if i_plot in [1, 6, 11, 16]: 
                plt.ylabel('Confidence Level')
        plt.legend(fontsize=9)
        i_plot += 1
    i_dic = 0
    current_title_len = len(title_txt)
    for dic_name in dic_names:
        AOCS_txt = '_meanAUCS = %.3f'%np.mean(AOCScores_StnByData[:, i_dic])
        title_txt += dic_names[i_dic][20:-7] +  AOCS_txt + ', '
        save_txt += '_' + dic_names[i_dic][20:-7]
        current_title_len += len(dic_names[i_dic][20:-7] +  AOCS_txt + ', ')
        i_dic += 1
        if current_title_len > 90:
            title_txt += '\n'
            current_title_len = 0
    plt.suptitle(title_txt[:-2])
    if save == 'yes':
        if new == 'yes':
            txt_new = 'New'
        else:
            txt_new = ''
        plt.savefig(txt_new+save_txt+'.png', dpi = 500)
        plt.close(fig)
    return

def plot_pValueCompare(dic_names, save = 'no', topAOC_rate = 0.2, new = 'no'):
    stns_dics = []
    for dic_name in dic_names:
        with open(dic_name, 'rb') as f:
            stns_dic = pickle.load(f)
        stns_dics.append(stns_dic)
    # if 'New' in dic_name:
    #     new = 'yes'
    stn_txts = stns_dic['_stn_txts']
    LocRads = stns_dic['_LocRads']
    plt.figure(figsize=(26,14))
    i_plot = 1
    title_txt = "Comparison plots of 20 stations' p-value vs Rs with "
    title_txt += 'topAUCrate=%s for datasets: '%topAOC_rate
    save_txt = 'ComparePlots_pValueVsR_tAUCr%s'%topAOC_rate
    i_dic = 0
    AOCScores_StnByData = np.zeros((20,len(stns_dics)))
    for stn_txt in stn_txts:
        plt.subplot(4,5,i_plot)
        i_data = -1
        for stns_dic in stns_dics:
            i_data += 1
            rate_mean_prod_tops_stn = stns_dic[stn_txt]['rate_mean_prod_tops_stn']
            ps_stn = np.array(rate_mean_prod_tops_stn)
            ps_stn_sorted = np.sort(ps_stn)[::-1]
            AOC_ind_start = int(len(ps_stn)*(1-topAOC_rate))
            AOCScore_stn = np.mean(ps_stn_sorted[AOC_ind_start:])
            AOCScores_StnByData[i_plot-1, i_data] = AOCScore_stn
            AOCS_txt = 'AUCS=%.2f'%AOCScore_stn
            if i_plot == 1:
                plt.plot(LocRads, ps_stn, 
                         label = dic_names[i_dic][23:-7] + '_' + AOCS_txt)
                i_dic += 1
            else:
                plt.plot(LocRads, ps_stn, label = AOCS_txt)
            plt.ylim(0,1)
            plt.title('Station ' + stn_txt)
            if i_plot > 15:
                plt.xlabel('R')
            if i_plot in [1, 6, 11, 16]: 
                plt.ylabel('p-value')
        plt.legend(fontsize=9)
        i_plot += 1
    i_dic = 0
    current_title_len = len(title_txt)
    for dic_name in dic_names:
        AOCS_txt = '_meanAUCS = %.3f'%np.nanmean(AOCScores_StnByData[:, i_dic])
        title_txt += dic_names[i_dic][23:-7] +  AOCS_txt + ', '
        save_txt += '_' + dic_names[i_dic][23:-7]
        current_title_len += len(dic_names[i_dic][23:-7] +  AOCS_txt + ', ')
        i_dic += 1
        if current_title_len > 90:
            title_txt += '\n'
            current_title_len = 0
    plt.suptitle(title_txt[:-2])
    if save == 'yes':
        # if new == 'yes':
        #     txt_new = 'New'
        # else:
        #     txt_new = ''
#        plt.savefig(txt_new+save_txt+'.png', dpi = 500)
        time_now = datetime.datetime.now()
        time_stamp = str(time_now)[:-7]
        plt.savefig('Comparison_pvalues_'+time_stamp+'.png', dpi = 500)
    return

import re

def plot_pValueCompare_0204(dic_names, save = 'no', search_syntax = ''):
    stns_dics = []
    for dic_name in dic_names:
        with open(dic_name, 'rb') as f:
            stns_dic = pickle.load(f)
        stns_dics.append(stns_dic)
    stn_txts = stns_dic['_stn_txts']
    LocRads = stns_dic['_LocRads']
    plt.figure(figsize=(26,14))
    i_plot = 1
    title_txt = "Comparison plots of 20 stations' p-value vs Rs for datasets: "
    i_dic = 0
    min_pvalue_StnByData = np.zeros((20,len(stns_dics)))
    for stn_txt in stn_txts:
        plt.subplot(4,5,i_plot)
        i_data = -1
        for stns_dic in stns_dics:
            i_data += 1
            rate_mean_prod_tops_stn = stns_dic[stn_txt]['rate_mean_prod_tops_stn']
            ps_stn = np.array(rate_mean_prod_tops_stn)
            min_pvalue_stn = np.nanmin(ps_stn)
            min_pvalue_StnByData[i_plot-1, i_data] = min_pvalue_stn
            AOCS_txt = 'min=%.3f'%min_pvalue_stn
            # if i_plot == 1:
            #     if len(search_syntax) > 0:
            #         label_str = re.search(search_syntax, stns_dic['_indi_txt']).group()
            #     else:
            #         label_str = dic_names[i_dic][23:-7] + '_'
            #     plt.plot(LocRads, ps_stn, label = label_str + AOCS_txt)
            #     i_dic += 1
            # else:
            #     plt.plot(LocRads, ps_stn, label = AOCS_txt)
            if len(search_syntax) > 0:
                label_str = re.search(search_syntax, stns_dic['_indi_txt']).group()
            else:
                label_str = dic_names[i_dic][23:-7] + '_'
            plt.plot(LocRads, ps_stn, label = label_str + AOCS_txt)
            i_dic += 1
            plt.ylim(0,1)
            plt.title('Station ' + stn_txt)
            if i_plot > 15:
                plt.xlabel('R')
            if i_plot in [1, 6, 11, 16]: 
                plt.ylabel('p-value')
        plt.legend(fontsize=9)
        i_plot += 1
    i_dic = 0
    current_title_len = len(title_txt)
    title_filled = 0
    for dic_name in dic_names:
        AOCS_txt = 'mean(minPvalue) = %.3f'%np.nanmean(min_pvalue_StnByData[:, i_dic])
        if title_filled == 0:
            label_str = dic_names[i_dic][23:-7] + '_'
            title_filled += 1
        else:
            if len(search_syntax) > 0:
                label_str = re.search(search_syntax, dic_name).group()
            else:
                label_str = dic_names[i_dic][23:-7] + '_'
        title_txt += label_str +  AOCS_txt + ', '
        current_title_len += len(label_str +  AOCS_txt + ', ')
        i_dic += 1
        if current_title_len > 90:
            title_txt += '\n'
            current_title_len = 0
    plt.suptitle(title_txt[:-2])
    if save == 'yes':
        time_now = datetime.datetime.now()
        time_stamp = str(time_now)[:-7]
        if len(search_syntax) > 0:
            time_stamp += '_(' + search_syntax + ')'
        plt.savefig('Comparison_minPvalues_'+time_stamp+'.png', dpi = 500)
    return