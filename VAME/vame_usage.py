#%%
import os.path

import matplotlib.pyplot as plt
import numpy as np
import vame
import glob
from itertools import groupby

#%%


def vame_fetch_video(datadir, mouse: list):  # read data from mouse.
    # create_training.py assign last n% frames to testing so no need to split here
    videos = []
    for mouse_name in mouse:
        videos += glob.glob(datadir + '*/*/*' + mouse_name + '*.avi')
    if len(mouse)==2:
        videos_new = []  # alternate between two mice.
        for i in range(int(len(videos) / 2)):
            videos_new.append(videos[i])
            videos_new.append(videos[i + 6])
    else:
        videos_new = videos
    return videos_new


def vame_load_tracking(config, datadir, mouse: list, center_data = False):
    # load keypoint data
    # reshape to n_sample*coordinates, alternate x y
    # center the keypoint?
    # save to wkdir/project/data/<video>/<video>-PE-seq.npy
    from pathlib import Path
    from vame.util.auxiliary import read_config

    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    path_to_file = cfg['project_path']

    # get mouse keypoint data
    files = []
    for mouse_name in mouse:
        files += glob.glob(datadir+'*/*/*/*'+mouse_name+'*/*v1_8.npz')

    for file in files:
        # get filename
        bname = os.path.basename(file).split('.')[0]
        bname = bname.replace('raw_feat_top_v1_8', 'Top')
        keypoints = np.load(file)['keypoints']
        keypoints = keypoints[:, 0, :]
        x_kps = keypoints[:, 0]
        y_kps = keypoints[:, 1]
        out = np.zeros(keypoints.shape)
        out = out.reshape(out.shape[0], -1)
        out[:, ::2] = x_kps
        out[:, 1::2] = y_kps
        out = out.transpose()

        np.save(os.path.join(path_to_file, 'data', bname, bname+'-PE-seq-raw.npy'), out)


def compute_bout_length(seq):
    bout_length = np.array([len(list(g)) for i, g in groupby(seq)])
    return bout_length.mean(), bout_length.std(), bout_length



#%% initiate project
# BEFORE USE: comment out video copying part of ./vame/initialize_project/new.py
#[98]     # for src, dst in zip(videos, destinations):
# [99]        # shutil.copy(os.fspath(src),os.fspath(dst))

working_directory = '/home/roton/PycharmProjects/VAME/'
project='VAME_run02_qy_aligned'
datadir = '/run/user/1006/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/' \
          'Phys/Kennedylab/Parkerlab/Behavior/Clozapine/'
mouse = ['m085', 'm971']
videos = vame_fetch_video(datadir, mouse)


vame.init_new_project(project=project, videos=videos, working_directory=working_directory, videotype='.avi')
config = working_directory + project + '-Feb28-2022/config.yaml'  # change the date here
vame_load_tracking(config=config, datadir=datadir, mouse=mouse)
#%% creating training and train
vame.egocentric_alignment(config, pose_ref_index=[0, 6], crop_size=(300, 300), use_video = False, video_format='avi', check_video=False)
# only nose to tail alignment works. check why
vame.create_trainset(config)  # make sure num_features is set to match number of keypoints
vame.train_model(config)
vame.evaluate_model(config)
vame.pose_segmentation(config)
#%% check bout length and umap visualization
latent_vec = np.load('./VAME_run01_qy-Feb18-2022/results/20191206_m971_clo05_amph_Top/VAME/kmeans-15/latent_vector_20191206_m971_clo05_amph_Top.npy')
labels = np.load('./VAME_run01_qy-Feb18-2022/results/20191206_m971_clo05_amph_Top/VAME/kmeans-15/15_km_label_20191206_m971_clo05_amph_Top.npy')
import umap
import seaborn as sns

#%%
plt.figure()
plt.hist(compute_bout_length(labels)[2], bins=100, log=True)
plt.title('m971_clo05_amph, aligned')
plt.show()

#%% plot latent_vec in umap
umap_embed = umap.UMAP(n_neighbors=30, min_dist=0.01).fit(latent_vec)
umap_plot = umap_embed.transform(latent_vec)
palette = np.array(sns.color_palette('hls', 15))
classes = [str(i)for i in range(15)]
x = umap_plot[:, 0]
y = umap_plot[:, 1]

plt.figure()
for i, j in enumerate(classes):
    xi = [x[u] for u in range(x.size) if labels[u].astype(str) == j]
    yi = [y[u] for u in range(x.size) if labels[u].astype(str) == j]
    plt.scatter(x=xi, y=yi, s=1, color=palette[i], label='label'+j)
    # plt.scatter(x=umap_plot[:, 0], y=umap_plot[:, 1], s=5, c=palette[labels.astype(int)])
plt.legend()
plt.xlim([x.min()*1.1, x.max()*1.5])
plt.title('m971_clo05_amph, aligned')
plt.show()

#%% evaluate model in testing mouse
# initiate eval project
working_directory = '/home/roton/PycharmProjects/VAME/'
project='VAME_eval02_aligned'
datadir = '/run/user/1006/gvfs/smb-share:server=fsmresfiles.fsm.northwestern.edu,share=fsmresfiles/Basic_Sciences/' \
          'Phys/Kennedylab/Parkerlab/Behavior/Clozapine/'
mouse_test = ['m485']
video_test = vame_fetch_video(datadir, mouse_test)
vame.init_new_project(project=project, videos=video_test, working_directory=working_directory, videotype='.avi')
config = working_directory + project + '-Mar1-2022/config.yaml'
vame_load_tracking(config=config, datadir=datadir, mouse=mouse_test)
#%%
# COPY THE TRAINED MODEL, model_losses, edit num_features in config
vame.egocentric_alignment(config, pose_ref_index=[0, 6], crop_size=(300, 300), use_video = False, video_format='avi', check_video=False)
vame.create_trainset(config)
# ADD seq_mean.npy and seq_std.npy
train_seq = np.load('./VAME_eval02_aligned-Mar1-2022/data/train/train_seq.npy')
train_mean = np.mean(train_seq)
train_std = np.std(train_seq)
np.save('./VAME_eval02_aligned-Mar1-2022/data/train/seq_mean.npy', train_mean)
np.save('./VAME_eval02_aligned-Mar1-2022/data/train/seq_std.npy', train_std)

vame.evaluate_model(config)  # dunno if this is necessary
vame.pose_segmentation(config)

#%%
latent_vec = np.load('./VAME_eval02_aligned-Mar1-2022/results/20200916_m485_clo01_amph_Top/VAME/kmeans-15/latent_vector_20200916_m485_clo01_amph_Top.npy')
labels = np.load('./VAME_eval02_aligned-Mar1-2022/results/20200916_m485_clo01_amph_Top/VAME/kmeans-15/15_km_label_20200916_m485_clo01_amph_Top.npy')
#%%
plt.figure()
plt.hist(compute_bout_length(labels)[2], bins=100, log=True)
plt.title('m485_clo01_amph, aligned')
plt.show()
#%%
umap_embed = umap.UMAP(n_neighbors=10, min_dist=0.01).fit(latent_vec)
# umap_embed = umap.UMAP().fit(latent_vec)
umap_plot = umap_embed.transform(latent_vec)
palette = np.array(sns.color_palette('hls', 15))
classes = [str(i)for i in range(15)]
x = umap_plot[:, 0]
y = umap_plot[:, 1]

plt.figure()
for i, j in enumerate(classes):
    xi = [x[u] for u in range(x.size) if labels[u].astype(str) == j]
    yi = [y[u] for u in range(x.size) if labels[u].astype(str) == j]
    plt.scatter(x=xi, y=yi, s=1, color=palette[i], label='label'+j)
    # plt.scatter(x=umap_plot[:, 0], y=umap_plot[:, 1], s=5, c=palette[labels.astype(int)])
# plt.plot(umap_plot[:1000, 0], umap_plot[:1000, 1], 'k-')
plt.legend(markerscale=3)
plt.xlim([x.min()*1.1, x.max()*1.5])
plt.title('m485_clo01_amph, aligned')
plt.show()

#%% save labels
np.save('/media/storage/qiaohan/vame_m485_clo01_amph_aligned_label.npy', labels)

#%% plot bout length distribution for testing set

for file in video_test:
    filename = os.path.basename(file).split('.')[0]
    labels = np.load(os.path.join('./VAME_eval02_aligned-Mar1-2022/results', filename, 'VAME/kmeans-15', '15_km_label_'+filename+'.npy'))
    blength = compute_bout_length(labels)[2]
    plt.hist(blength, bins=100, log=True)
plt.legend([os.path.basename(file).split('.')[0] for file in video_test])
plt.show()

# length of the first bout
blength_plot = []
for file in video_test:
    filename = os.path.basename(file).split('.')[0]
    labels = np.load(os.path.join('./VAME_eval02_aligned-Mar1-2022/results', filename, 'VAME/kmeans-15', '15_km_label_'+filename+'.npy'))
    blength = compute_bout_length(labels)[2]
    blength_plot.append(blength[0])
plt.hist(blength_plot)
plt.show()

#%% check occurance of each label
labelcount = []
for i in range(labels.min(), labels.max()):
    labelcount.append(np.sum(labels.astype(int)==i))

#%% raster plot for bouts
beh_palette = sns.color_palette('hls', 14)
i=0
plt.figure(figsize=(15, 4))
for beh_idx in np.unique(labels):
    beh_pos = np.where(labels == beh_idx)
    plt.eventplot(beh_pos, colors=beh_palette[i])
    i += 1
plt.legend(['label' + str(k) for k in range(1, 15)], ncol=6)
plt.yticks([])
plt.xlabel('frames')
plt.title('m485_clo01_amph, VAME label')
plt.show()
