#%%
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import pickle
import seaborn as sns
import umap



def compute_bout_length(seq):
    bout_length = np.array([len(list(g)) for i, g in groupby(seq)])
    return bout_length.mean(), bout_length.std(), bout_length

def connect_dots(cord):
    cord = np.insert(cord, 3, cord[0]).reshape(-1, 1)
    cord = np.insert(cord, 4, cord[2]).reshape(-1, 1)
    cord = np.insert(cord, 6, cord[1]).reshape(-1, 1)
    cord = np.insert(cord, 7, cord[5]).reshape(-1, 1)
    cord = np.insert(cord, 10, cord[5]).reshape(-1, 1)
    cord = np.insert(cord, 11, cord[8]).reshape(-1, 1)
    cord = np.insert(cord, 13, cord[9]).reshape(-1, 1)
    return cord
#%% plot eval nll and kld
datadir = 'C:/Users/qyx1327/Documents/results/tvae/loss_plot_data'
files_nll = glob.glob(datadir+'/*nll*')
files_kld = glob.glob(datadir+'/*kld*')
labels = ['z = 32, aligned', 'z = 32, not aligned', 'z = 8, not aligned']
# plot nll across runs
plt.figure()
for filename in files_nll:
    with open(filename, 'r') as fhandle:
        data = json.load(fhandle)
    y = np.array([])
    for i in range(len(data)):
        j = len(data)-1-i
        y = np.append(y, np.array(data[j]['y']))
    # check for outliers and replace
    # step = y[1:]-y[0:-1]
    # if step.max() > 10000:  # arbitrary
    #     y[step.argmax()+1] = 0
    y = y[y < y[0]+1]
    x = np.arange(0, y.shape[0]*10, 10)
    plt.plot(x, y)
plt.legend(labels)
plt.title('Negative log likelihood')
plt.xlabel('eval epoch')
plt.show()

plt.figure()
for filename in files_kld:
    with open(filename, 'r') as fhandle:
        data = json.load(fhandle)
    y = np.array([])
    for i in range(len(data)):
        j = len(data) - 1 - i
        y = np.append(y, np.array(data[j]['y']))
    x = np.arange(0, y.shape[0]*10, 10)
    plt.plot(x, y)
plt.legend(labels)
plt.title('KL divergence')
plt.xlabel('eval epoch')
plt.show()

#%% plot distribution of reconstruction rsq. done w/ calculate rsq.py
#%% plot umap embeddings
with open('centered_gmm_model.pickle', 'rb') as fhandle:
    gm = pickle.load(fhandle)

current_model = gm[20] # n_component = 25
file_list = glob.glob('C:/Users/qyx1327/Documents/results/tvae/test_video_embeddings/m485/z32_aligned/*')
embed_all = np.array([])
labels_all = np.array([])
i = 0

for file in file_list:
    embed = load_split_data(file, split=False)[0] # take either one
    labels = current_model.predict(embed)
    if i == 0:
        embed_all = embed
    else:
        embed_all = np.row_stack([embed_all, embed])
    labels_all = np.append(labels_all, labels)
    i += 1


#%% downsampling
chooseidx = np.random.permutation(np.arange(embed_all.shape[0]))[0: 10000]  # plot ramdomly selected 10000 points
embed_all = embed_all[chooseidx]
labels_all = labels_all[chooseidx]
#%%
umap_embed = umap.UMAP(n_neighbors=50, min_dist=0.01).fit(embed_all)
# umap_embed = umap.UMAP().fit(latent_vec)
umap_plot = umap_embed.transform(embed_all)
palette = np.array(sns.color_palette('hls', 25))
classes = [str(i)for i in range(25)]
labels_all = labels_all.astype(int)
x = umap_plot[:, 0]
y = umap_plot[:, 1]

plt.figure()
for i, j in enumerate(classes):
    xi = [x[u] for u in range(x.size) if labels_all[u].astype(str) == j]
    yi = [y[u] for u in range(x.size) if labels_all[u].astype(str) == j]
    plt.scatter(x=xi, y=yi, s=1, color=palette[i], label='label'+j)
    # plt.scatter(x=umap_plot[:, 0], y=umap_plot[:, 1], s=5, c=palette[labels.astype(int)])
# plt.plot(umap_plot[:1000, 0], umap_plot[:1000, 1], 'k-')
plt.legend(ncol=2, markerscale=3)
plt.xlim([x.min()*0.7, x.max()*1.6])
plt.title('m485, z=32, aligned')
plt.show()

#%% plot bout length histogram

bout_length = compute_bout_length(labels_all)[2] # use the not downsampled ones

plt.figure()
plt.hist(bout_length)
plt.title('z=32, aligned, n_component = 25')
plt.xlabel('bout length')
plt.ylabel('count')
plt.show()

#%% plot bout raster plot
# bent_dict = parse_annotations('C:/Users/qyx1327/Documents/results/tvae/20200916_m485_clo01_amph_bento.annot')
beh_labels = bent_dict['behs']
beh_all = []
palette_beh = sns.color_palette()
i=0
plt.figure(figsize=(15,4))
for label in beh_labels:
    bout_pos = bouts['channel'][label]
    event_arr = np.array([])
    for bidx in range(bout_pos.shape[0]):
        event_arr = np.append(event_arr, np.arange(bout_pos[bidx, 0], bout_pos[bidx, 1]))
    # beh_all.append(event_arr)
    plt.eventplot(event_arr, colors=palette_beh[i])
    i += 1
# plt.eventplot(beh_all)
plt.yticks([])
plt.xlabel('frames')
plt.legend(beh_labels, ncol=3)
plt.title('m485_clo01_amph, manual annotation')
plt.show()

#%% reconstruction example
#%% load data
orig = np.load('C:/Users/qyx1327/Documents/results/tvae/embeddings/result_centered.npz')['originals']
recon = np.load('C:/Users/qyx1327/Documents/results/tvae/embeddings/result_centered.npz')['reconstructions']
#%% plot 0 ,4, 9
seqidx = 10451
recon_pal = sns.color_palette('Blues')[2:5]
orig_pal = sns.color_palette('Oranges')[2:5]

for frmidx in [0, 4, 9]:
    xorig = orig[seqidx, frmidx, ::2]
    yorig = orig[seqidx, frmidx, 1::2]
    xorig_l = connect_dots(xorig)
    yorig_l = connect_dots(yorig)
    xrecon = recon[seqidx, frmidx, ::2]
    yrecon = recon[seqidx, frmidx, 1::2]
    xrecon_l = connect_dots(xrecon)
    yrecon_l = connect_dots(yrecon)
    plt.scatter(x=xorig, y=yorig, s=100, color=orig_pal[int(frmidx/4)])
    plt.scatter(x=xrecon, y=yrecon, s=100, color=recon_pal[int(frmidx/4)])
    plt.plot(xorig_l, yorig_l, color=orig_pal[int(frmidx/4)])
    plt.plot(xrecon_l, yrecon_l, color=recon_pal[int(frmidx/4)])
plt.xlim([-150, 400])
plt.ylim([-150, 350])
plt.legend(['original', 'reconstruction'])
plt.title('sequence '+ str(seqidx) + ', z=32, aligned')
plt.show()

#%% plot confusion matrix for manual and VAME labels
bent_dict = parse_annotations('C:/Users/qyx1327/Documents/results/tvae/20200916_m485_clo01_amph_bento.annot')
behs_frame = bent_dict['behs_frame']
manual_labels = np.zeros(len(behs_frame))

for i, beh in enumerate(behs_frame):
    if beh == 'Forward':
        manual_labels[i] = 0
    elif beh == 'Grooming':
        manual_labels[i] = 1
    elif beh == 'LeftTurn':
        manual_labels[i] = 2
    elif beh == 'RearingUp':
        manual_labels[i] = 3
    elif beh == 'RightTurn':
        manual_labels[i] = 4
    else:
        manual_labels[i] = 5  # other

#%%
VAME_labels = np.load('C:/Users/qyx1327/Documents/results/tvae/m485_clo01_amph_VAME_label.npy')
# check length difference. VAME embedding works on 30 frame time window. so remove last 30 frm of manual label
manual_labels = manual_labels[:-30]

confmat = np.zeros([6,15])
# confusion matrix
for frmidx in range(manual_labels.size):
    vlabel = VAME_labels[frmidx]
    mlabel = int(manual_labels[frmidx])
    confmat[mlabel, vlabel] += 1
# normalized
confmat_norm = np.zeros([6, 15])
for ridx in range(confmat.shape[0]):
    r = confmat[ridx, :]
    rtotal = sum(r)
    confmat_norm[ridx, :] = r/rtotal

# normalized
confmat_cnorm = np.zeros([6, 15])
for cidx in range(1, confmat.shape[1]):
    r = confmat[:, cidx]
    rtotal = sum(r)
    confmat_cnorm[:, cidx] = r/rtotal



#%%
# beh_labels.append('Other')
plt.imshow(confmat)
plt.yticks(np.arange(6), beh_labels)
plt.xlabel('VAME label')
plt.xticks(range(0, 15))
plt.title('manual vs. VAME annotation, raw')
plt.colorbar(shrink=0.45, aspect=15)
plt.show()







