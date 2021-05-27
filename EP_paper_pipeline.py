#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:18:41 2021

@author: haoyuwen

Python scripts that were used to produce the research in paper 
'Hidden-State Modelling of a Cross-section of Geo-electric Time Series Data Can Provide 
Reliable Intermediate-term Earthquake Risk Assessments in Taiwan'

Required data in directory: 
    'D_XXXX_ti2_FCW500S2.pickle' where XXXX are 20 GEMS stations' names
    'tweqk28.pickle', which is the EQ catalogue data in the same time periods

Required modules in directory:
    EP_modA.py
    EP_modB_new.py
    BM_mod.py
    CM_mod.py
    
Note 1: Since the data is not available for public access, this script serves mostly as a recording 
of our procedures. 

Note 2: With data that is available for public access, Block #3 and #4 can be executed. To execute 
Block after #5, EQ catalogue data would be necessary. To excute Block #1 and #2, 0.5 Hz geo-electric 
data named 'D_XXXX_ti2_FCW500S2.pickle' data would be necessary.

Note 3: There are redundant functions in these modules that are not needed for running this script.
These functions were used during the early development stage. 

List of Contents (in code block numbers)
0. Run this block to load necessary modules, define necessary function and variables 
1. Run this block to apply Butter Worth Filtering on the 0.5-Hz geo-electric data
2. Run this block to compute C, V, S, K from the geo-electric TSs data, obtained at Block 1.
3. Run this block to clean outliers on the V TSs
4. Run this block to perform k-means clustering and run BWA 15 times with random initializations
and obtain 15 HMMs. 
5. Run this block to create 400 simulated models and save their information
6. Run this block to create save the information for the optimal empirical HMM, and plot the 
EQ frequency grid maps
7. Run this block to create, plot, and save the information of the grid maps of discrimination power
and discrimination reliability
8. Note on hyper-parameter optimization
9. Run this block to organize the information from the 28 pickle files described in Block 8.
10. Run this block to plot Fig. 13
11. Run this block to plot Fig. 14
12. Run this block to plot Fig. 2
13. Run this block to plot Fig. 3
14. Run this block to plot Fig. 4
15. Run this block to plot Fig. 6
16. Run this block to plot Fig. 7
17. Run this block to plot Fig. 8, 9, 11
18. Run this block to plot Fig. 10
19. Run this block to plot Fig. 12

"""

# In[0] 
'''
Run this block to load necessary modules, define necessary function and variables 
'''
from scipy.signal import butter, lfilter
import pickle
import EP_modB_new as mB
import BM_mod as bm
import CM_mod as cm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as pdate

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def BWfilter_data(dataname = 'D_FENL_ti2_FCW500S2', fs = 43200, lowcut = 8.64, 
                  highcut = 1536.43341, save_code = 'BW1'):
    with open(dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    new_Mat = data['Mat'].copy()
    new_Mat[:,1] = butter_bandpass_filter(new_Mat[:,1], lowcut, highcut, fs, order=3)
    new_Mat[:,2] = butter_bandpass_filter(new_Mat[:, 2], lowcut, highcut, fs, order=3)
    data['Mat'] = new_Mat
    data['fs'] = fs
    data['lowcut'] = lowcut
    data['highcut'] = highcut
    with open(dataname + '_' + save_code + '.pickle', 'wb') as f:
        pickle.dump(data, f)
    return

def get_EQS_dic(datanames, main_folder, ensemble_folder, indi_txt, min_DR, RW_day, L, 
                div_y = 16):
    '''
    in cm.all_stns_plot, for each station:
        EQRs_highdisc = the DR heatmap of empirical
        EQratios_disc = the DP heatmap of empirical
        EQRs_en_disc = the DP heatmap of simulated
    now, we also need the DR heatmap of simulated, call it EQRs_en_highdisc:
        EQRs_en_highdisc = the DR heatmap of simulated
    As for r_EQs, we name:
        r_EQS = the r_EQS of empirical 
        r_EQS_en = the r_EQS of simulated
    '''
    div_txt = '_div%s'%div_y
    EQRs_highdisc_dic = {}
    EQratios_dic = {}
    EQRs_en_disc_dic = {}
    EQRs_en_highdisc_dic = {}
    confidence_dic = {}
    r_EQS_dic = {}
    r_EQS_en_dic = {}
    for dataname in datanames:
        stn_txt = dataname[2:6]
        name_list = os.listdir(main_folder)
        for name in name_list:
            if '.pickle' in name:
                if indi_txt in name:
                    if div_txt in name:
                        if stn_txt in name:
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
        EQRs_en_highdisc = np.zeros((len(ensemble_names), div_y, div_y))
        for i_en in range(len(ensemble_names)):
            EQR_en_highdisc = np.zeros((div_y, div_y))
            EQR_en_highdisc[:] = np.nan
            for i in range(div_y):
                for j in range(div_y):
                    EQratio_disc_ien = EQRs_en_disc[i_en, i,j]
                    if ~np.isnan(EQratio_disc_ien):
                        EQR_en_disc = EQRs_en_disc[:, i, j]
                        EQR_en_highdisc[i,j] = np.sum(EQratio_disc_ien > EQR_en_disc)
            EQR_en_highdisc = EQR_en_highdisc/EQR_en_disc.shape[0]
            EQRs_en_highdisc[i_en, :, :] = EQR_en_highdisc
        # 
        EQRs_highdisc_dic[stn_txt] = EQRs_highdisc
        EQratios_dic[stn_txt] = EQratios
        EQRs_en_highdisc_dic[stn_txt] = EQRs_en_highdisc
        EQRs_en_disc_dic[stn_txt] = EQRs_en_disc
        # 
        EQS1 = plot_info['EQS1_plot']
        EQS1[np.isnan(EQS1)] = 0
        EQS2 = plot_info['EQS2_plot']
        EQS2[np.isnan(EQS2)] = 0
        nEQs = EQS1 + EQS2
        n_EQs = np.sum(nEQs)
        # r_EQS = ratio of EQs being in Satisfactory cells
        r_EQS = np.sum(nEQs[EQRs_highdisc>min_DR])/n_EQs
        r_EQS_en = np.zeros((len(ensemble_names),))
        for i_en in range(len(ensemble_names)):
            r_EQS_en[i_en] = np.sum(nEQs[EQRs_en_highdisc[i_en]>min_DR])/n_EQs
        confidence = np.sum(r_EQS_en < r_EQS)/len(ensemble_names)
        confidence_dic[stn_txt] = confidence
        r_EQS_dic[stn_txt] = r_EQS
        r_EQS_en_dic[stn_txt] = r_EQS_en
        # create NEQS_dic to save
        EQS_dic = {'EQRs_highdisc_dic': EQRs_highdisc_dic,
                   'EQratios_dic': EQratios_dic,
                   # 'EQRs_en_disc_dic': EQRs_en_disc_dic,
                   # 'EQRs_en_highdisc_dic': EQRs_en_highdisc_dic,
                   'confidence_dic': confidence_dic,
                   'r_EQS_dic': r_EQS_dic,
                   'r_EQS_en_dic': r_EQS_en_dic,
                   'div_y': div_y,
                   'main_folder': main_folder,
                   'ensemble_folder': ensemble_folder,
                   'indi_txt': indi_txt,
                   'min_DR': min_DR,
                   'RW_day': RW_day, 
                   'L': L}
    return EQS_dic

def create_folder_names(indi_txts):
    # function returns the corresponding folders
    main_folders = []
    ensemble_folders = []
    for indi_txt in indi_txts:
        if '0225' in indi_txt:
            main_folders.append('testF_L30_days_0225_main')
            ensemble_folders.append('testF_L30_days_0225_ReSimu')
        elif '0218' in indi_txt:
            main_folders.append('testF_L40_days_0218_main')
            ensemble_folders.append('testF_L40_days_0218_ReSimu')
        elif '0216' in indi_txt:
            main_folders.append('testF_0.03days_Ls_0216_main')
            ensemble_folders.append('testF_0.03days_Ls_0216_ReSimu')
        elif '0201' in indi_txt:
            main_folders.append('testF_L80_days_0201_main')
            ensemble_folders.append('testF_L80_days_0201_ReSimu')
        elif '0130' in indi_txt:
            main_folders.append('testF_0.1days_Ls_0130_main')
            ensemble_folders.append('testF_0.1days_Ls_0130_ReSimu')
        elif '0125' in indi_txt:
            main_folders.append('testF_L60_days_0125_main')
            ensemble_folders.append('testF_L60_days_0125_ReSimu')
    return main_folders, ensemble_folders

# In generate and save
def genNsave_EQS_dic_all(min_DR, indi_txts_L30, indi_txts_L40, indi_txts_L60, indi_txts_L80, 
                         RW_days, Ls, save_code = '_xbyx_0301'):
    indi_txts_all = [indi_txts_L30, indi_txts_L40, indi_txts_L60, indi_txts_L80]
    main_folders_L80, ensemble_folders_L80 = create_folder_names(indi_txts_L80)
    main_folders_L60, ensemble_folders_L60 = create_folder_names(indi_txts_L60)
    main_folders_L40, ensemble_folders_L40 = create_folder_names(indi_txts_L40)
    main_folders_L30, ensemble_folders_L30 = create_folder_names(indi_txts_L30)
    main_folders_all = [main_folders_L30, main_folders_L40, main_folders_L60, main_folders_L80]
    ensemble_folders_all = [ensemble_folders_L30, ensemble_folders_L40, ensemble_folders_L60, 
                            ensemble_folders_L80]
    # indi_txts_all[0][0]
    EQS_dic_all = indi_txts_all.copy()
    for i_L in range(4):
        for i_RW in range(7):
            indi_txt = indi_txts_all[i_L][i_RW]
            main_folder = main_folders_all[i_L][i_RW]
            ensemble_folder = ensemble_folders_all[i_L][i_RW]
            RW_day = RW_days[i_RW]
            L = Ls[i_L]
            EQS_dic = get_EQS_dic(datanames, main_folder, ensemble_folder, indi_txt, min_DR, 
                                  RW_day, L)
            EQS_dic_all[i_L][i_RW] = EQS_dic
    
    EQS_para = {'min_DR': min_DR,
                'RW_days': RW_days,
                'Ls': Ls,
                'indi_txts_all': indi_txts_all}
    EQS_dic_all.append(EQS_para)
    with open('EQSDic'+save_code+'_minDR%s.pickle'%min_DR, 'wb') as f:
        pickle.dump(EQS_dic_all, f)
    return

def plot_EQS_confidences(EQS_file, RW_days, Ls, datanames, mode = 'confidence'):
    EQS_id = EQS_file[12:-7]
    with open(EQS_file, 'rb') as f:
        EQS_dic_all = pickle.load(f)
    confidence_mat_dic = {}
    for dataname in datanames:
        stn_txt = dataname[2:6]
        confidence_mat = np.zeros((4,7))
        for i_L in range(4):
            for i_RW in range(7):
                confidence_mat[i_L, i_RW] = EQS_dic_all[i_L][i_RW][mode+'_dic'][stn_txt]
        confidence_mat_dic[stn_txt] = confidence_mat
    xticks = np.arange(7)+0.5
    yticks = np.arange(4)+0.5
    xtick_labels = RW_days
    ytick_labels = Ls
    i_plot = 1
    plt.figure(figsize=(20,10))
    fig_title = '20 '+mode+' heatmaps for ' + EQS_id
    # plt.suptitle(fig_title)
    xtick_labels_empty = ['']*7
    if mode == 'confidence':
        vmax = 1
    elif mode == 'r_EQS':
        vmax = 0.5
    for dataname in datanames:
        stn_txt = dataname[2:6]
        plt.subplot(4, 5, i_plot)
        ax = sns.heatmap(confidence_mat_dic[stn_txt], annot=True, annot_kws={"fontsize":8},
                         vmin = 0, vmax=vmax, cbar=False)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels_empty, rotation = 0)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0)
        plt.title(stn_txt)
        if i_plot in [1, 6, 11, 16]:
            plt.ylabel('Q')
        if i_plot >= 16:
            plt.xlabel('$L_{w}$ (days)')
            ax.set_xticklabels(xtick_labels, rotation = 0)
        i_plot += 1
    plt.savefig(fig_title + '_0507.tiff', dpi = 300, bbox_inches='tight')
    plt.close('all')
    return

def create_optiParas(EQS_file, RW_days, Ls):
    with open(EQS_file, 'rb') as f:
        EQS_dic_all = pickle.load(f)
    opti_params = {}
    for dataname in datanames:
        stn_txt = dataname[2:6]
        confidence_mat = np.zeros((4,7))
        for i_L in range(4):
            for i_RW in range(7):
                confidence_mat[i_L, i_RW] = EQS_dic_all[i_L][i_RW]['confidence_dic'][stn_txt]
        xys_opti = np.argwhere(confidence_mat == np.nanmax(confidence_mat))
        if xys_opti.shape[0] > 1: 
            r_EQSs = np.zeros((xys_opti.shape[0], ))
            for i in range(xys_opti.shape[0]):
                r_EQSs[i] = EQS_dic_all[xys_opti[i,0]][xys_opti[i,1]]['r_EQS_dic'][stn_txt]
            xys_opti = xys_opti[np.argmax(r_EQSs), :]
        else:
            xys_opti = xys_opti[0,:]
        i_L, i_RW = xys_opti[0], xys_opti[1]
        EQS_dic_opti = EQS_dic_all[i_L][i_RW]
        opti_params[stn_txt] = {'L': Ls[i_L],
                                'RW_day': RW_days[i_RW],
                                'ensemble_folder': EQS_dic_opti['ensemble_folder'],
                                'main_folder': EQS_dic_opti['main_folder'],
                                'indi_txt': EQS_dic_opti['indi_txt'],
                                'r_EQS': EQS_dic_opti['r_EQS_dic'][stn_txt],
                                'r_EQS_en': EQS_dic_opti['r_EQS_en_dic'][stn_txt],
                                'EQratios': EQS_dic_opti['EQratios_dic'][stn_txt],
                                'EQs_highdisc': EQS_dic_opti['EQRs_highdisc_dic'][stn_txt],
                                'confidence': confidence_mat[i_L, i_RW],
                                'min_DR': EQS_dic_opti['min_DR']
                                }
    with open('OptiParams_'+EQS_file, 'wb') as f:
        pickle.dump(opti_params, f)
    return

def plot_ttf_heatmap(ys, ttf, x_UB, y_LB, y_UB, x_div = 30, y_div = 20):
    y_bins = np.linspace(y_LB, y_UB, y_div+1)
    x_bins = np.linspace(0, x_UB, x_div+1)
    hists_norm = np.zeros((y_div, x_div))
    for i in range(x_div):
        x_start, x_end = x_bins[i], x_bins[i+1]
        ys_inbound = ys[np.logical_and(ttf<x_end, ttf>x_start)]
        hists = np.histogram(ys_inbound, bins = y_bins)
        hists_norm[:,i] = hists[0]/np.sum(hists[0])
    xtlbs = (x_bins[0:-1] + x_bins[1:])/2
    xtlbs = np.round(xtlbs*10)/10
    ytlbs = (y_bins[0:-1] + y_bins[1:])/2
    ytlbs = np.round(np.flipud(ytlbs)*1000)/1000
    sns.heatmap(np.flipud(hists_norm), cmap="YlGnBu", xticklabels = xtlbs, yticklabels = ytlbs)
    return

def ttf_histplot(col, t_eqs, taf_LB = 2, taf_UB = 0, save_info = str(), save = 'no',
                 plot_name = '', save_name = ''):
    '''
    in case of saving figure, the figure will not appear, but saved directly
    '''
    if len(save_name) == 0:
        save_name = plot_name
    fig = plt.figure(figsize=(20,12))
    if len(plot_name) == 0:
        plot_name = 'ttf_hist_'+col['dataname'][2:6] + '_' + save_info
        if taf_LB > 0:
            plot_name = plot_name + '_tafLB%s'%taf_LB
        if taf_UB > 0:
            plot_name = plot_name + '_tafUB%s'%taf_UB
    ttf = mB.time2failure(col['t_ind'][0:-1], t_eqs)
    taf = mB.time_after_failure(col['t_ind'][0:-1], t_eqs)
    ttf_notnan = ttf.copy()[~np.isnan(ttf)]
    ttf_notnan_sorted = np.sort(ttf_notnan)
    x_UB = ttf_notnan_sorted[-200]
    y_div = 20
    x_div = 30
    plt.subplot(2,2,1)
    indi_C = mB.taf_filter(col['Cs'], taf, lower_bound = taf_LB)
    indi_sorted = np.sort(indi_C[0,:])
    indi_sorted = indi_sorted[~np.isnan(indi_sorted)]
    y_UB = indi_sorted[int(len(indi_sorted)*0.995)]-0.001
    y_LB = indi_sorted[int(len(indi_sorted)*0.005)]
    plot_ttf_heatmap(indi_C[0,:], ttf, x_UB, y_LB, y_UB, x_div = x_div, y_div = y_div)
    plt.xlabel('TTF (days)')
    plt.ylabel('(a) Normalized PDF, C')
    plt.subplot(2,2,2)
    indi_V = mB.taf_filter(col['Vs'], taf, lower_bound = taf_LB)
    indi_sorted = np.sort(indi_V[0,:])
    indi_sorted = indi_sorted[~np.isnan(indi_sorted)]
    y_UB = indi_sorted[int(len(indi_sorted)*0.89)]
    if np.max(indi_sorted) < y_UB:
        y_UB = np.max(indi_sorted)
    y_LB = indi_sorted[int(len(indi_sorted)*0.0005)]
    plot_ttf_heatmap(indi_V[0,:], ttf, x_UB, y_LB, y_UB, x_div = x_div, y_div = y_div)
    plt.xlabel('TTF (days)')
    plt.ylabel('(b) Normalized PDF, V')
    plt.subplot(2,2,3)
    indi_S = mB.taf_filter(col['Ss'], taf, lower_bound = taf_LB)
    indi_sorted = np.sort(indi_S[0,:])
    indi_sorted = indi_sorted[~np.isnan(indi_sorted)]
    y_UB = indi_sorted[int(len(indi_sorted)*0.98)]
    y_LB = indi_sorted[int(len(indi_sorted)*0.02)]
    plot_ttf_heatmap(indi_S[0,:], ttf, x_UB, y_LB, y_UB, x_div = x_div, y_div = y_div)
    plt.xlabel('TTF (days)')
    plt.ylabel('(c) Normalized PDF, S')
    plt.subplot(2,2,4)
    indi_K = mB.taf_filter(col['Ks'], taf, lower_bound = taf_LB)
    indi_sorted = np.sort(indi_K[0,:])
    indi_sorted = indi_sorted[~np.isnan(indi_sorted)]
    y_UB = indi_sorted[int(len(indi_sorted)*0.89)]
    y_LB = indi_sorted[int(len(indi_sorted)*0.005)]
    plot_ttf_heatmap(indi_K[0,:], ttf, x_UB, y_LB, y_UB, x_div = x_div, y_div = y_div)
    plt.xlabel('TTF (days)')
    plt.ylabel('(d) Normalized PDF, K')
    if save == 'no':
        pass
    else:
        plt.savefig(save_name+'.tiff', dpi = 300, bbox_inches='tight')
        plt.close(fig)
    return

def plot_n_save(dataname, mag = 3, mag_UB = 0, taf_LB = 2, taf_UB = 0, radius = 0.8, 
                save = 'no'):
    '''
    trendname can be '_resBW600' or '_trendBW600'
    '''
    with open(dataname+'.pickle', 'rb') as f:
        data = pickle.load(f)
    eqks = mB.eqk()
    eqks.select(dataname[2:], mag, radius = radius)
    t_eqs = eqks.selected[:,0]
    if mag_UB > mag:
        t_eqs = t_eqs[eqks.selected[:,1] < mag_UB]
        print('Mag_UB applied, new #EQs = %s'%len(t_eqs))
    pn1 = 'Heatmap of normalized distribution of geo-electric TS indexes at different '
    pn1 += 'Time-to-Failures, for EQs of M >= %s within %s long-lat degrees '%(mag, radius)
    pn1 += 'from '+dataname[2:6]+' station (NS direction)'
    pn2 = 'Heatmap of normalized distribution of geo-electric TS indexes at different '
    pn2 += 'Time-to-Failures, for EQs of M >= %s within %s long-lat degrees '%(mag, radius)
    pn2 += 'from '+dataname[2:6]+' station (EW direction)'
    sn1 = 'Heatmap_TTFvIndex_trV_M%s_Rad%s_'%(mag, radius) + dataname[2:6] + '_NS'
    sn2 = 'Heatmap_TTFvIndex_trV_M%s_Rad%s_'%(mag, radius) + dataname[2:6] + '_EW'
    ttf_histplot(data['col1'], t_eqs, taf_LB = taf_LB, taf_UB = taf_UB, 
                    save_info = 'col1', save = save, plot_name = pn1, save_name = sn1)
    ttf_histplot(data['col2'], t_eqs, taf_LB = taf_LB, taf_UB = taf_UB, 
                    save_info = 'col2', save = save, plot_name = pn2, save_name = sn2)
    plt.close('all')
    return

datanames = ['D_FENG_ti2_FCW500S2',
             'D_KUOL_ti2_FCW500S2',
             'D_TOCH_ti2_FCW500S2',
             'D_LIOQ_ti2_FCW500S2',
             'D_YULI_ti2_FCW500S2',
             'D_SIHU_ti2_FCW500S2',
             'D_PULI_ti2_FCW500S2',
             'D_HERM_ti2_FCW500S2',
             'D_FENL_ti2_FCW500S2',
             'D_CHCH_ti2_FCW500S2',
             'D_WANL_ti2_FCW500S2',
             'D_SHCH_ti2_FCW500S2',
             'D_LISH_ti2_FCW500S2',
             'D_KAOH_ti2_FCW500S2',
             'D_HUAL_ti2_FCW500S2',
             'D_ENAN_ti2_FCW500S2',
             'D_DAHU_ti2_FCW500S2',
             'D_DABAXti2_FCW500S2',
             'D_RUEY_ti2_FCW500S2',
             'D_SHRL_ti2_FCW500S2']

# In[1] 
'''
Run this block to apply Butter Worth Filtering on the 0.5-Hz geo-electric data
Required data: 'D_XXXX_ti2_FCW500S2.pickle' in directory
Output data: 'D_XXXX_ti2_FCW500S2_BW0813.pickle' saved to directory

'''
for dataname in datanames:
    BWfilter_data(dataname, save_code = 'BW0813')
    
# In[2]
'''
Run this block to compute C, V, S, K from the geo-electric TSs data, obtained at Block 1.
Computation parameters: 'RW_days', the length of non-overlapping time window (days) to compute 
                        those indexes, noted by Lw in paper. 
Required data: 'D_XXXX_ti2_FCW500S2_BW0813.pickle' in directory
Output data: 'D_XXXX_ti2_FCW500S2_BW0813_RWday0.1_05JAN.pickle' saved to directory
'''
RW_days = [0.1]
for RW_day in RW_days:
    for dataname in datanames:
        dataname = dataname + '_BW0813'
        _ = mB.indicator_map_eqdata_day(dataname, extra_savename = '_05JAN', 
                                        RW_day = RW_day)
        
# In[3]
'''
Run this block to clean outliers on the V TSs
Required data: 'D_XXXX_ti2_FCW500S2_BW0813_RWday0.1_05JAN.pickle' in directory
Output data: 'D_XXXX_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV.pickle' saved to directory
'''
for RW_day in RW_days:
    for dataname in datanames:
        dataname_CVSK = dataname + '_BW0813_RWday%s_05JAN'%RW_day
        mB.clean_V(dataname_CVSK, mode = 'tr')
        
# In[4]
'''
Run this block to perform k-means clustering and run BWA 15 times with random initializations
and obtain 15 HMMs. 
Computation parameters: 'n_labelss', which is the number of labels for the k-means clustering, 
                         noted by Q in the paper.
Required data: 'D_XXXX_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV.pickle' in directory
Output data: 'CHMMMI_XXXX_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV_CVSK_R15L40_0218NA.pickle' 
              saved to directory
'''
datanames = ['D_FENG_ti2_FCW500S2_BW0813',
             'D_KUOL_ti2_FCW500S2_BW0813',
             'D_TOCH_ti2_FCW500S2_BW0813',
             'D_LIOQ_ti2_FCW500S2_BW0813',
             'D_YULI_ti2_FCW500S2_BW0813',
             'D_SIHU_ti2_FCW500S2_BW0813',
             'D_PULI_ti2_FCW500S2_BW0813',
             'D_HERM_ti2_FCW500S2_BW0813',
             'D_FENL_ti2_FCW500S2_BW0813',
             'D_CHCH_ti2_FCW500S2_BW0813',
             'D_WANL_ti2_FCW500S2_BW0813',
             'D_SHCH_ti2_FCW500S2_BW0813',
             'D_LISH_ti2_FCW500S2_BW0813',
             'D_KAOH_ti2_FCW500S2_BW0813',
             'D_HUAL_ti2_FCW500S2_BW0813',
             'D_ENAN_ti2_FCW500S2_BW0813',
             'D_DAHU_ti2_FCW500S2_BW0813',
             'D_DABAXti2_FCW500S2_BW0813',
             'D_RUEY_ti2_FCW500S2_BW0813',
             'D_SHRL_ti2_FCW500S2_BW0813']

n_labelss = [40]
n_runs = 15
# indi_namess = [['Cs', 'Vs', 'Ss', 'Ks']]
# indi_txts = ['CVSK']

data_family = '_RWday0.1_05JAN_trV'
indi_txt = data_family + '_CVSK'
mid_name = '_R%sL%s_0218NA'%(n_runs, n_labelss[0])

for i in range(len(datanames)):
    indi_names = ['Cs', 'Vs', 'Ss', 'Ks']
    dataname_CVSK_DT = datanames[i] + indi_txt
    save_name = 'CHMMMI_' + dataname_CVSK_DT[2:] + mid_name + '.pickle'
    bm.sys_BM_CHMMMI_day_0125(indi_names, n_labelss, n_runs, save_name, 
                              dataname = datanames[i][2:] + data_family)
        
# In[5]
'''
Run this block to create 400 simulated models and save their information
Required data: 'CHMMMI_XXXX_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV_CVSK_R15L40_0218NA.pickle' 
                in directory
Output data: 'CHMMMI_XXXX_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV_CVSK_R15L40_0218NA_minEQ1
              _magLB3_div16_info_S1ensemble399.pickle' 
                saved to a folder, being subdirectory '/testF_L40_days_0218_ReSimu/'
'''

folder_family = 'testF_L40_days_0218'
save_dir = folder_family + '_ReSimu/'
after_fix = indi_txt + mid_name + '.pickle'
shuffle_perstn = 400 # number of simulated counterparts
min_EQ, mag_LB = 1, 3 # this setting don't need to change
div_y = 16 # 16-by-16 grid map
cm.create_bootstrap_plot_info(datanames, after_fix, min_EQ, mag_LB, div_y, 
                              save_dir = save_dir, shuffle_perstn = shuffle_perstn,
                              ind_start = 0, version = '0125', ReSimu = 'yes')

# In[6]
'''
Run this block to create save the information for the optimal empirical HMM, and plot the 
EQ frequency grid maps
Required data: 'CHMMMI_XXXX_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV_CVSK_R15L40_0218NA.pickle' 
                in directory
Output data: 'EQGM_S1_CHMMMI_CHCH_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV_CVSK_R15L40_0218NA_
              minEQ1_magLB3_div16_info.pickle' 
              saved to a folder, being subdirectory '/testF_L40_days_0218_main/'
'''
main_folder = folder_family + '_main'
for file_name in datanames:
    file_name = 'CHMMMI_' + file_name[2:] + after_fix
    bm.EQGM_5plot(file_name, save = 'no', div_y = div_y, EQrate_mode = 'S1', 
                  min_EQ = min_EQ, mag_LB = mag_LB, save_info = 'yes', 
                  save_dir = main_folder + '/', plot = 'yes', version = '0125')

# In[7]
'''
Run this block to create, plot, and save the information of the grid maps of discrimination power
and discrimination reliability
Required data: folders 'testF_L40_days_0218_main' and 'testF_L40_days_0218_ReSimu' in directory
Output data: 'OptimalLocRadsDicts_div16_CVSK_UFD_trV_ReSimu_minEQ1_mag3_RWday0.1_05JAN_trV_CVSK
              _R15L40_0218NA.pickle' in directory
'''

ensemble_folder = folder_family + '_ReSimu'
indi_txtF = indi_txt + mid_name
indi_txt_key = 'CVSK' # this setting don't need to change
EQR_info = cm.all_stns_plot(min_EQ = min_EQ, mag_LB = mag_LB, div_y = div_y, 
                            main_folder = main_folder, ensemble_folder = ensemble_folder, 
                            indi_txt = indi_txtF, save_extratxt = '_ReSimu', plot = 'yes', 
                            plot_Disc = 'yes', indi_txt_key = indi_txt_key, shift = 0, 
                            return_dic = 'yes', paratxt = indi_txtF)
cm.plot_OptiLocDiscScore_hist(EQR_info, div_y=div_y, min_EQ = min_EQ, indi_txt = indi_txtF, 
                              indi_txt_key = indi_txt_key, plot = 'yes', shift = 0, 
                              return_dic = 'no', save_dic = 'yes',paratxt = indi_txtF, 
                              save_extratxt = '_ReSimu')

# In[8]
'''
Note on hyper-parameter optimization
There are 2 key hyper-parameters here: n_labelss and RW_days, their values for the demo code 
above are:
n_labelss = [40]
RW_days = [0.1]
In the paper, we denoted n_labelss as 'Q' and RW_days as'Lw', and used a 4-by-7 grid search:
n_labelss = [30, 40, 60, 80]
RW_days = [0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.25]

In order to implement the grid search, simply run code blocks 4, 5, 6, 7 while replacing the
values of n_labelss and RW_days accordingly.

At the end of all these, the following files will be available at the directory:
OptimalLocRadsDicts_div16_CVSK_UFD_trV_ReSimu_minEQ1_mag3_XXXXXXXXX_MMDDNA.pickle,
where 'MMDDNA' can be arbtrary name or be ignored, we used month-day-'NA' format for our case
and 'XXXXXXXXX' are:

RWday0.02_05JAN_trV_CVSK_R15L80
RWday0.03_05JAN_trV_CVSK_R15L80
RWday0.04_05JAN_trV_CVSK_R15L80
RWday0.05_05JAN_trV_CVSK_R15L80
RWday0.1_05JAN_trV_CVSK_R15L80
RWday0.2_05JAN_trV_CVSK_R15L80
RWday0.25_05JAN_trV_CVSK_R15L80
RWday0.02_05JAN_trV_CVSK_R15L60
RWday0.03_05JAN_trV_CVSK_R15L60
RWday0.04_05JAN_trV_CVSK_R15L60
RWday0.05_05JAN_trV_CVSK_R15L60
RWday0.1_05JAN_trV_CVSK_R15L60
RWday0.2_05JAN_trV_CVSK_R15L60
RWday0.25_05JAN_trV_CVSK_R15L60
RWday0.02_05JAN_trV_CVSK_R15L40
RWday0.03_05JAN_trV_CVSK_R15L40
RWday0.04_05JAN_trV_CVSK_R15L40
RWday0.05_05JAN_trV_CVSK_R15L40
RWday0.1_05JAN_trV_CVSK_R15L40
RWday0.2_05JAN_trV_CVSK_R15L40
RWday0.25_05JAN_trV_CVSK_R15L40
RWday0.02_05JAN_trV_CVSK_R15L30
RWday0.03_05JAN_trV_CVSK_R15L30
RWday0.04_05JAN_trV_CVSK_R15L30
RWday0.05_05JAN_trV_CVSK_R15L30
RWday0.1_05JAN_trV_CVSK_R15L30
RWday0.2_05JAN_trV_CVSK_R15L30
RWday0.25_05JAN_trV_CVSK_R15L30

Note1: the total computational time can be very long. For my 4 GHz i7 quad-core desktop, it took 
more than 2 weeks' total computational time to obtain all these results.

Note2: we initially set n_runs = 30, but later found out that n_runs = 15 to be good enough. 
Therefore some of our data are labeled with 'R30LXX'. But we took care of it by only using half 
of the models, so their are effectively 'R15LXX'. 

After having obtained all those files, we used the following codes (Block 9-11) to plot Fig. 13 
and 14 in the manuscript, which are the R_EQS values and GCL values for the grid search

'''
# In[9]
'''
Run this block to organize the information from the 28 pickle files described in Block 8.
Required data: 28 pickle files in directory
Output data: 'EQSDic_4by7_0308_minDR0.95.pickle' in directory
'''

indi_txts_L80 = ['RWday0.02_05JAN_trV_CVSK_R30L80_0201NA',
                 'RWday0.03_05JAN_trV_CVSK_R30L80_0201NA',
                 'RWday0.04_05JAN_trV_CVSK_R30L80_0201NA',
                 'RWday0.05_05JAN_trV_CVSK_R30L80_0201NA',
                 'RWday0.1_05JAN_trV_CVSK_R30L80_0130NA',
                 'RWday0.2_05JAN_trV_CVSK_R30L80_0201NA',
                 'RWday0.25_05JAN_trV_CVSK_R30L80_0201NA']

indi_txts_L60 = ['RWday0.02_05JAN_trV_CVSK_R30L60_0125NA',
                 'RWday0.03_05JAN_trV_CVSK_R30L60_0125NA',
                 'RWday0.04_05JAN_trV_CVSK_R30L60_0125NA',
                 'RWday0.05_05JAN_trV_CVSK_R30L60_0125NA',
                 'RWday0.1_05JAN_trV_CVSK_R30L60_0125NA',
                 'RWday0.2_05JAN_trV_CVSK_R30L60_0125NA',
                 'RWday0.25_05JAN_trV_CVSK_R30L60_0125NA']

indi_txts_L40 = ['RWday0.02_05JAN_trV_CVSK_R15L40_0218NA',
                 'RWday0.03_05JAN_trV_CVSK_R15L40_0216NA',
                 'RWday0.04_05JAN_trV_CVSK_R15L40_0218NA',
                 'RWday0.05_05JAN_trV_CVSK_R15L40_0218NA',
                 'RWday0.1_05JAN_trV_CVSK_R15L40_0218NA',
                 'RWday0.2_05JAN_trV_CVSK_R15L40_0218NA',
                 'RWday0.25_05JAN_trV_CVSK_R15L40_0218NA']

indi_txts_L30 = ['RWday0.02_05JAN_trV_CVSK_R15L30_0225NA',
                 'RWday0.03_05JAN_trV_CVSK_R15L30_0216NA',
                 'RWday0.04_05JAN_trV_CVSK_R15L30_0225NA',
                 'RWday0.05_05JAN_trV_CVSK_R15L30_0225NA',
                 'RWday0.1_05JAN_trV_CVSK_R30L30_0130NA',
                 'RWday0.2_05JAN_trV_CVSK_R15L30_0225NA',
                 'RWday0.25_05JAN_trV_CVSK_R15L30_0225NA']

RW_days = [0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.25]
Ls = [30, 40, 60, 80]
min_DR = 0.95 # minimum discrimination reliabliilty. equivalent to p = 0.05 (see paper)

genNsave_EQS_dic_all(min_DR, indi_txts_L30, indi_txts_L40, indi_txts_L60, indi_txts_L80, 
                     RW_days, Ls, save_code = '_4by7_0308')

# In[10]
'''
Run this block to plot Fig. 13
Required data: 28 pickle files in directory
Output data: 'EQSDic_4by7_0308_minDR0.95.pickle' in directory
'''
RW_days = [0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.25]
Ls = [30, 40, 60, 80]
EQS_file = 'EQSDic_4by7_0308_minDR0.95.pickle'

plot_EQS_confidences(EQS_file, RW_days, Ls, datanames, mode = 'r_EQS')

# In[11]
'''
Run this block to plot Fig. 14
Required data: 28 pickle files in directory
Output data: 'EQSDic_4by7_0308_minDR0.95.pickle' in directory

Note: after this, you can already decide for each of the 20 stations, which hyper-parameter is
the optimal one for that station. Therefore, the HMM obtained with the optimal hyper-parameter
can be concluded as the optimal HMM for that station. 
For example, it can be
CHMMMI_CHCH_ti2_FCW500S2_BW0813_RWday0.03_05JAN_trV_CVSK_R15L30_0216NA.pickle
'''
RW_days = [0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.25]
Ls = [30, 40, 60, 80]
EQS_file = 'EQSDic_4by7_0308_minDR0.95.pickle'

plot_EQS_confidences(EQS_file, RW_days, Ls, datanames)

'''
For the following codes, we show how we obtained the figures in the manuscript.
'''
# In[12]
'''
Run this block to plot Fig. 2
Required data: 'D_KAOH_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV.pickle' in directory
'''
file_name = 'D_KAOH_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV'
mags = [4]
radiuss = [2]
for mag in mags:
    for radius in radiuss:
        plot_n_save(file_name, mag = mag, radius = radius, save = 'yes')

# In[13]
'''
Run this block to plot Fig. 3
Required data: 'D_KAOH_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV.pickle' in directory
'''

dataname = 'KAOH_ti2_FCW500S2_BW0813_RWday0.1_05JAN_trV'
col_name = 'col1' # col1 for NS, col2 for EW direction for each station
min_it = 30 # minimum number of iterations 
# n_labels = 50 
# n_states = 2
n_bins = 30
gamma_B = 'no' # not important
mutate = 0  # not important
good_info_scores = {'Cs': -np.inf, 'Vs': -np.inf, 'Ss': -np.inf, 'Ks': -np.inf}
good_infos = {'Cs': None, 'Vs': None, 'Ss': None, 'Ks': None}

for indi_name in ['Cs', 'Vs', 'Ss', 'Ks']:
    for i in range(8):
        indi_names = [indi_name]
        info = bm.BM_CHMM_day(dataname, col_name, indi_name, n_bins, gamma_B, min_it, mutate)
        if info['model_trained'] == 'no':
            try_num = 0
            while (info['model_trained'] == 'no'):
                if try_num > 16:
                    print('Failed to get good HMM for run%s.'%i)
                    break
                info = bm.BM_CHMM_day(dataname, col_name, indi_name, n_bins, gamma_B, min_it, mutate)
                try_num += 1
            print(indi_name + ', i=%s, score=%s'%(i, np.int(info['model_scores'][-1])))
            if info['model_scores'][-1] > good_info_scores[indi_name]:
                good_info_scores[indi_name] = info['model_scores'][-1]
                good_infos[indi_name] = info
      
indi_names = ['Cs', 'Vs', 'Ss', 'Ks']
plot_bases = [1, 3, 5, 7]
alphabets = ['x', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
fig = plt.figure(figsize=(20, 9.5))
for i_indi in range(4):
    indi_name = indi_names[i_indi]
    info = good_infos[indi_name]
    plt.subplot(2,4,plot_bases[i_indi])
    alphabet = alphabets[plot_bases[i_indi]]
    plt.title('('+alphabet+') Emission Probability Distributions for '+indi_name[0])
    plt.plot(info['bin_centers'], info['B'][0,:], label = 'S1', lw = 1)
    plt.plot(info['bin_centers'], info['B'][1,:], label = 'S2', lw = 1)
    plt.legend()
    plt.xlabel(indi_name[0])
    plt.subplot(2,4,plot_bases[i_indi]+1)
    plt.plot(pdate.num2date(info['t_segs'][1:]), info['Gamma'][0,:], lw = 0.5)
    alphabet = alphabets[plot_bases[i_indi]+1]
    plt.title('('+alphabet+') Posterior Probability P(S1), index = '+indi_name[0])
    plt.xlabel("Time")
figure_name = 'CHMMPlot_new_'+ dataname
plt.savefig(figure_name +'_F3.tiff', dpi = 300, bbox_inches='tight')

# In[14]
'''
Run this block to plot Fig. 4
Required data: 'CHMMMI_CHCH_ti2_FCW500S2_BW0813_RWday0.03_05JAN_trV_CVSK_R15L30_0216NA.pickle' 
                in directory
'''
filenames = ['CHMMMI_CHCH_ti2_FCW500S2_BW0813_RWday0.03_05JAN_trV_CVSK_R15L30_0216NA']

for file_name in filenames:
    bm.EQGM_5plot_pubF4(file_name + '.pickle', save = 'yes', div_y = 16, EQrate_mode = 'S1', 
                        min_EQ = 1, mag_LB = 3, save_info = 'no', plot = 'sub', 
                        version = '0125', max_models = 15)

# In[15]
'''
Run this block to plot Fig. 6
Required data: none
'''
eqks = mB.eqk()
eqks.makeGM(div_y = 16, plot = 'yes')

# In[16] 
'''
Run this block to plot Fig. 7
Required data: 'CHMMMI_CHCH_ti2_FCW500S2_BW0813_RWday0.03_05JAN_trV_CVSK_R15L30_0216NA.pickle' 
                in directory
'''
filenames = ['CHMMMI_CHCH_ti2_FCW500S2_BW0813_RWday0.03_05JAN_trV_CVSK_R15L30_0216NA']

for file_name in filenames:
    bm.EQGM_5plot_pubF7(file_name + '.pickle', save = 'yes', div_y = 16, EQrate_mode = 'S1', 
                        min_EQ = 1, mag_LB = 3, save_info = 'no', plot = 'sub', 
                        version = '0125', max_models = 15)

# In[17] 
'''
Run this block to plot Fig. 8, 9, 11
Required data: 'OptiParams_EQSDic_4by7_0308_minDR0.95.pickle' in directory
'''
def plot_3_20heatmaps(optiPara_file):
    with open(optiPara_file, 'rb') as f:
        opti_params = pickle.load(f)
    div_y = 16
    eqks = mB.eqk()
    eqks.makeGM(div_y = div_y, div_x = div_y) 
    x_cuts, y_cuts = eqks.x_cuts, eqks.y_cuts
    x_cuts = np.round(x_cuts*100)/100
    y_cuts = np.round(y_cuts*100)/100
    xtick_labels = x_cuts
    ytick_labels = np.flipud(y_cuts)
    x_cuts, y_cuts = x_cuts[1:-1], y_cuts[1:-1]
    xtick_labels, ytick_labels = xtick_labels[1:-1], ytick_labels[1:-1]
    xticks = np.arange(0, len(x_cuts),1)-0.5
    yticks = np.arange(0, len(y_cuts),1)-0.5
    # plot EQRs
    i_plot = 1
    plt.figure(figsize=(20,14))
    for dataname in datanames:
        stn_txt = dataname[2:6]
        opti_param = opti_params[stn_txt]
        EQRs = opti_param['EQratios'][1:div_y-1, 1:div_y-1]
        ax = plt.subplot(4,5, i_plot)
        plt.imshow(EQRs, cmap="PiYG", vmin=0, vmax=1)
        for iy in range(div_y-2):
            for ix in range(div_y-2):
                if ~np.isnan(EQRs[iy, ix]):
                    entry = EQRs[iy, ix]*100
                    if (entry > 25) & (entry < 75):
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                            fontsize=5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
        stn_coord = eqks.position[stn_txt]
        ax.scatter(bm.get_tick(stn_coord[1],xtick_labels)-0.5,bm.get_tick(stn_coord[0],ytick_labels)-0.5, 
                   marker = '*', color='b', s = 120, label = stn_txt)
        plt.legend(loc = 'upper left')
        i_plot += 1
    fig_name = 'EQR heatmaps for ' + optiPara_file[:-7]
    plt.savefig(fig_name + '.tiff', dpi = 300, bbox_inches='tight')
    plt.close('all')
    # In plot EQRs
    i_plot = 1
    plt.figure(figsize=(20,14))
    for dataname in datanames:
        stn_txt = dataname[2:6]
        opti_param = opti_params[stn_txt]
        EQRs = opti_param['EQratios'][1:div_y-1, 1:div_y-1]
        Discs = np.abs(EQRs-0.5)
        ax = plt.subplot(4,5, i_plot)
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
                            fontsize=5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
        stn_coord = eqks.position[stn_txt]
        ax.scatter(bm.get_tick(stn_coord[1],xtick_labels)-0.5,bm.get_tick(stn_coord[0],ytick_labels)-0.5, 
                   marker = '*', color='b', s = 120, label = stn_txt)
        plt.legend(loc = 'upper left')
        i_plot += 1
    fig_name = 'Discrimination Power heatmaps for ' + optiPara_file[:-7]
    plt.savefig(fig_name + '.tiff', dpi = 300, bbox_inches='tight')
    plt.close('all')
    # In plot discrimination reliability R_D
    i_plot = 1
    plt.figure(figsize=(20,14))
    for dataname in datanames:
        stn_txt = dataname[2:6]
        opti_param = opti_params[stn_txt]
        RDs = opti_param['EQs_highdisc'][1:div_y-1, 1:div_y-1]
        ax = plt.subplot(4,5, i_plot)
        plt.imshow(RDs, cmap="Reds", vmin=0, vmax=1)
        for iy in range(div_y-2):
            for ix in range(div_y-2):
                if ~np.isnan(RDs[iy, ix]):
                    entry = RDs[iy, ix]*100
                    if entry < 50:
                        c = 'k'
                    else:
                        c = 'w'
                    ax.text(ix, iy, '%.1f'%(entry), ha="center", va="center", color=c, 
                            fontsize=5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation = 45, fontsize=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, rotation = 0, fontsize=8)
        stn_coord = eqks.position[stn_txt]
        ax.scatter(bm.get_tick(stn_coord[1],xtick_labels)-0.5,bm.get_tick(stn_coord[0],ytick_labels)-0.5, 
                   marker = '*', color='b', s = 120, label = stn_txt)
        plt.legend(loc = 'upper left')
        i_plot += 1
    fig_name = 'Discrimination Reliability heatmaps for ' + optiPara_file[:-7]
    plt.savefig(fig_name + '.tiff', dpi = 300, bbox_inches='tight')
    plt.close('all')
    return

optiPara_file = 'OptiParams_EQSDic_4by7_0308_minDR0.95.pickle'
plot_3_20heatmaps(optiPara_file)

# In[18]
'''
Run this block to plot Fig. 10
Required data: see below
'''
file_names = ['CHMMMI_YULI_ti2_FCW500S2_BW0813_RWday0.02_05JAN_trV_CVSK_R30L80_0201NA.pickle',
              'CHMMMI_SHRL_ti2_FCW500S2_BW0813_RWday0.04_05JAN_trV_CVSK_R15L40_0218NA.pickle',
              'CHMMMI_CHCH_ti2_FCW500S2_BW0813_RWday0.03_05JAN_trV_CVSK_R15L30_0216NA.pickle',
              'CHMMMI_SIHU_ti2_FCW500S2_BW0813_RWday0.05_05JAN_trV_CVSK_R30L80_0201NA.pickle']

fig = plt.figure(figsize=(20,12))
i_plot = 1
pre_fixes = ['a', 'b', 'c', 'd',]

for file_name in file_names:
    info = bm.HSVA_getinfo_0125(file_name, keep_models = 'yes', max_models = 15)
    pS1s_voted, A = info['pS1s_voted'], info['A']
    ensembles = cm.create_HSensembles_wModelPara(info, num_ensemble = 10, flip = 'yes')
    plt.subplot(2,2,i_plot)
    ax = sns.heatmap(ensembles, cbar = True)
    title = '(' + pre_fixes[i_plot-1] +  ') 10 simulated HS TSs for ' + file_name[7:11] 
    plt.title(title + ', and its empirical HS TS')
    if i_plot >= 3:
        plt.xlabel('Time')
    if (i_plot == 1) |(i_plot == 3):
        plt.ylabel('Index of TSs')
    plt.yticks(np.arange(11)+0.5)
    ytls = list(range(-1,50))
    ytls[0] = 'Empirical TS'
    ax.set_yticklabels(ytls, rotation = 0)
    # deal with xticks
    time_axis = pdate.num2date(info['t_segs'])
    time_axis_short = time_axis.copy()
    for i in range(len(time_axis)):
        time_axis_short[i] = time_axis_short[i].strftime("%b %Y")
    xtick_points = np.int64(np.linspace(0,len(time_axis), 10))
    xtick_points = xtick_points[1:-1]
    xtick_pt_labels = []
    for i in range(len(xtick_points)):
        xtick_pt_labels.append(time_axis_short[xtick_points[i]])
    plt.xticks(xtick_points, xtick_pt_labels, rotation='horizontal')
    plt.xlabel('Time')
    i_plot += 1
plt.savefig('YULI_SHRL_CHCH_SIHU_10shuffles_ReSimu.tiff', dpi = 300, bbox_inches='tight')

# In[19]
'''
Run this block to plot Fig. 12
Required data: 'OptiParams_EQSDic_4by7_0308_minDR0.95.pickle' in directory
'''
def plot_rEQS_hist(optiPara_file):
    with open(optiPara_file, 'rb') as f:
        opti_params = pickle.load(f)
    i_plot = 1
    plt.figure(figsize=(20,13))
    for dataname in datanames:
        stn_txt = dataname[2:6]
        opti_param = opti_params[stn_txt]
        param_id = '[$L_{w}$,Q]=[%s,%s]'%(opti_param['RW_day'], opti_param['L'])
        r_EQS_en = opti_param['r_EQS_en']
        r_EQS = opti_param['r_EQS']
        confidence = opti_param['confidence']
        plt.subplot(4,5,i_plot)
        sns.distplot(r_EQS_en)
        txt = '$R_{EQS}$=%.3f,GCL=%.3f'%(r_EQS, confidence)
        plt.axvline(x=r_EQS, color = 'r', label = txt)
        plt.legend(loc = 'upper left')
        plt.title(stn_txt + ' with ' +param_id)
        if i_plot >= 16:
            plt.xlabel('$R_{EQS}$')
        if i_plot in [1, 6, 11, 16]:
            plt.ylabel('PDF')
        i_plot += 1
    plt.savefig('r_EQS_Histogram_' + optiPara_file[:-7] + '_0507.tiff', dpi = 300, 
                bbox_inches='tight')
    plt.close('all')
    return

plot_rEQS_hist('OptiParams_EQSDic_4by7_0308_minDR0.95.pickle')

