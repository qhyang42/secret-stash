#%% prepare data for training tvae
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

def prep_data_tvae(datadir, filelist, slength = 10):
    out = np.array([])
    for fileName in filelist:
        data = np.load(os.path.join(datadir, fileName))
        keypoints = data['keypoints']
        keypoints = keypoints[:, 0, :]
        x_kps = keypoints[:, 0]
        y_kps = keypoints[:, 1]
        out_this = np.zeros(keypoints.shape)
        out_this = out_this.reshape(out_this.shape[0], -1)
        out_this[:, ::2] = x_kps
        out_this[:, 1::2] = y_kps
        if out.size == 0:
            out = out_this
        else:
            out = np.row_stack([out, out_this])

    # reshape ts
    rs_out = np.zeros([out.shape[0], slength, out.shape[1]])
    for i in range(out.shape[0] - slength):
        rs_out[i, :, :] = out[i:i + slength, :]
    return rs_out
#%%
# datadir = 'Y:/Parkerlab/Behavior/Clozapine'
# datadir = '/Volumes/fsmresfiles/BasicSciences/Phys/Kennedylab/Parkerlab/Behavior/Clozapine'
datadir = '/run/user/1006/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/Phys/Kennedylab/Parkerlab/Behavior/Clozapine/HighDose/Amph/output_v1_8'
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

rs_out = prep_data_tvae(datadir, fileList)

# save
# np.savez('data_in.npz', data = rs_out)


#%% plit data into training and testing
import numpy as np
import os
import matplotlib.pyplot as plt

datadir = 'Y:/Parkerlab/Behavior/Clozapine'
#datadir = '/Volumes/fsmresfiles/BasicSciences/Phys/Kennedylab/Parkerlab/Behavior/Clozapine'
fileList_train = ['HighDose/Amph/output_v1_8/20191207_m085_clo04_amph/20191207_m085_clo04_amph_raw_feat_top_v1_8.npz',
            'HighDose/Amph/output_v1_8/20191207_m971_clo04_amph/20191207_m971_clo04_amph_raw_feat_top_v1_8.npz',
            'HighDose/Control/output_v1_8/20191207_m085_clo04/20191207_m085_clo04_raw_feat_top_v1_8.npz',
            'HighDose/Control/output_v1_8/20191207_m971_clo04/20191207_m971_clo04_raw_feat_top_v1_8.npz',
            'LowDose/Amph/output_v1_8/20191206_m971_clo05_amph/20191206_m971_clo05_amph_raw_feat_top_v1_8.npz',
            'LowDose/Control/output_v1_8/20191206_m085_clo05/20191206_m085_clo05_raw_feat_top_v1_8.npz',
            'Vehicle/Amph/output_v1_8/20191205_m085_clo01_amph/20191205_m085_clo01_amph_raw_feat_top_v1_8.npz',
            'Vehicle/Amph/output_v1_8/20191205_m971_clo01_amph/20191205_m971_clo01_amph_raw_feat_top_v1_8.npz',
            'Vehicle/Control/output_v1_8/20191205_m971_clo01/20191205_m971_clo01_raw_feat_top_v1_8.npz']
            #read two file from each condition. same mice
rs_out_train = prep_data_tvae(datadir, fileList_train)


fileList_test = [ 'LowDose/Amph/output_v1_8/20191206_m085_clo05_amph/20191206_m085_clo05_amph_raw_feat_top_v1_8.npz',
                'LowDose/Control/output_v1_8/20191206_m971_clo05/20191206_m971_clo05_raw_feat_top_v1_8.npz',
                'Vehicle/Control/output_v1_8/20191205_m085_clo01/20191205_m085_clo01_raw_feat_top_v1_8.npz',]
                # read two file from each condition. same mice
rs_out_test = prep_data_tvae(datadir, fileList_test)

# save
np.savez('data_in_split.npz', data_train = rs_out_train, data_test = rs_out_test)

#%% single video. dataloader now require data_train and data_test in data.npz
#mouse = 'm972'
mouse = 'm485'
datadir = '/run/user/1006/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/Phys/Kennedylab/Parkerlab/Behavior/Clozapine/'
filelist_test_single = glob.glob(datadir+'*/*/*/*'+mouse+'*/*v1_8.npz') #run for m972 in each conditions

for filedir in iter(filelist_test_single):
    data_test = prep_data_tvae(datadir='', filelist=filelist_test_single)
    filename = os.path.basename(filedir).split('.')[0]
    np.savez('/media/storage/qiaohan/tvae/test_input/test_data_in_' + filename + '.npz', data_train = data_test, data_test = data_test )
    np.savez('/media/storage/qiaohan/tvae/test_input/test_data_out_' + filename + '.npz', data_train = data_test, data_test = data_test )


#%% test
# data1 = np.load(os.path.join(datadir, fileList[1]))
# data2 = np.load(os.path.join(datadir, fileList[2]))
#
# keypoints1 = data1['keypoints']
# keypoints1 = keypoints1[:, 0, :]
# x_kps = keypoints1[:, 0]
# y_kps = keypoints1[:, 1]
# out1 = np.zeros(keypoints1.shape)
# out1 = out1.reshape(out1.shape[0], -1)
# out1[:, ::2] = x_kps
# out1[:, 1::2] = y_kps
#
# keypoints2 = data2['keypoints']
# keypoints2 = keypoints2[:, 0, :]
# x_kps = keypoints2[:, 0]
# y_kps = keypoints2[:, 1]
# out2 = np.zeros(keypoints2.shape)
# out2 = out2.reshape(out2.shape[0], -1)
# out2[:, ::2] = x_kps
# out2[:, 1::2] = y_kps
# np.row_stack