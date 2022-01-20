#%% prepare data for training tvae
import numpy as np
import os

#datadir = 'Y:/Parkerlab/Behavior/Clozapine'
datadir = '/Volumes/fsmresfiles/BasicSciences/Phys/Kennedylab/Parkerlab/Behavior/Clozapine'
fileList = ['HighDose/Amph/output_v1_8/20191207_m085_clo04_amph/20191207_m085_clo04_amph_raw_feat_top_v1_8.npz',
            'HighDose/Amph/output_v1_8/20191207_m971_clo04_amph/20191207_m971_clo04_amph_raw_feat_top_v1_8.npz',
            'HighDose/Control/output_v1_8/20191207_m085_clo04/20191207_m085_clo04_raw_feat_top_v1_8.npz',
            'HighDose/Control/output_v1_8/20191207_m971_clo04/20191207_m971_clo04_raw_feat_top_v1_8.npz',
            'LowDose/Amph/output_v1_8/20191206_m085_clo05_amph/20191206_m085_clo05_amph_raw_feat_top_v1_8.npz',
            'LowDose/Amph/output_v1_8/20191206_m971_clo05_amph/20191206_m971_clo05_amph_raw_feat_top_v1_8.npz',
            'LowDose/Control/output_v1_8/20191206_m085_clo05/20191206_m085_clo05_raw_feat_top_v1_8.npz',
            'LowDose/Control/output_v1_8/20191206_m971_clo05/20191206_m971_clo05_raw_feat_top_v1_8.npz',
            'Vehicle/Amph/output_v1_8/20191205_m085_clo01_amph/20191205_m085_clo01_amph_raw_feat_top_v1_8.npz',
            'Vehicle/Amph/output_v1_8/20191205_m971_clo01_amph/20191205_m971_clo01_amph_raw_feat_top_v1_8.npz',
            'Vehicle/Control/output_v1_8/20191205_m085_clo01/20191205_m085_clo01_raw_feat_top_v1_8.npz',
            'Vehicle/Control/output_v1_8/20191205_m971_clo01/20191205_m971_clo01_raw_feat_top_v1_8.npz']
            #read two file from each condition. same mice
for fileName in fileList
    



