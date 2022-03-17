#%% cluster embedding with GMM
import os.path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import glob
from itertools import groupby

# load and split data
def load_split_data(path: str, split = True, valprop = 0.2):
    embed = np.load(path)['embeddings']
    if split is False:
        embed_train = embed
        embed_test = embed
    else:
        idx = np.random.choice(embed.shape[0], size=np.int32(np.floor(embed.shape[0] * valprop)), replace=False)
        embed_test = embed[idx, :]
        mask = np.ones(embed.shape[0], dtype=bool)
        mask[idx] = False
        embed_train = embed[mask, :]
    return embed_train, embed_test

# calculate bout length
def compute_bout_length(seq):
    bout_length = np.array([len(list(g)) for i, g in groupby(seq)])
    return bout_length.mean(), bout_length.std(), bout_length

# gm_bic = np.array([])
#
# for n_comp in range(140, 150):
#     gm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=1000).fit(embed_train)
#     gm_bic = np.append(gm_bic, gm.bic(embed_test))

#%% load data
file_path = 'C:/Users/qyx1327/Documents/results/tvae/embeddings/result_msplit.npz'
embed_train, embed_test = load_split_data(file_path)
#%% train and save the gmms, calculate bic

gm = []
gm_bic = np.array([])
for n_comp in range(5, 100):
    gm.append(GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=1000).fit(embed_train))
    gm_bic = np.append(gm_bic, gm[-1].bic(embed_test))

with open('z32_gmm_model.pickle', 'wb') as fhandle:
    pickle.dump(gm, fhandle)

np.save('z32_bic.npy', gm_bic)

#%% plot
xaxis = np.arange(5, 100)
plt.plot(xaxis, gm_bic_32)
plt.plot(xaxis, gm_bic_8)
plt.plot(xaxis, gm_bic_centered)
plt.xlim([5, 100])
plt.xticks(np.arange(5, 100, 10))
plt.xlabel('n_component')
plt.ylabel('bic')
plt.title('BIC of GMMs')
plt.legend(['z=32, not aligned', 'z=8, not aligned', 'z=32, aligned'])
plt.show()

# np.save('C:/Users/qyx1327/Documents/results/msplit_bic.npy', gm_bic)

#%% predict and save labels for testing videos. view labels with video in bento
# choose n_component = 25
with open('centered_gmm_model.pickle', 'rb') as fhandle:
    gm = pickle.load(fhandle)

current_model = gm[20]
file_list = glob.glob('C:/Users/qyx1327/Documents/results/tvae/test_video_embeddings/m485/*')
for file in iter(file_list):
    embed = load_split_data(file, split=False)[0] # take either one
    labels = current_model.predict(embed)
    labels = labels+1
    labels = np.pad(labels, (4,5))
    file_name = os.path.basename(file).split('.')[0]
    # np.save(file_name+'_labels', labels)
    label_list = ['label' + str(number) for number in range(15)] # same length as number of nonzero clusters
    # dump_labels_bento(labels=labels, filename='C:/Users/qyx1327/Documents/results/tvae/test_video_embeddings/m485/dino/'+file_name+'_bent.annot', beh_list=label_list)



# file_list = glob.glob('C:/Users/qyx1327/Documents/results/tvae/test_video_embeddings/m972/*.npy')
# for file in iter(file_list):
#     labels = np.load(file)
#     file_name = os.path.basename(file).split('.')[0]
#     dump_labels_bento(labels=labels, filename='C:/Users/qyx1327/Documents/results/tvae/test_video_embeddings/m485/dino/'+file_name+'_bent.annot', beh_list=label_list)
#     # np.save(file_name+'_labels', labels)

#%% plot embeddings in UMAP
import umap

# file = 'C:/Users/qyx1327/Documents/results/tvae/test_video_embeddings/results_1.npz'
file = 'C:/Users/qyx1327/Documents/results/tvae/embeddings/result_centered.npz'
embed = np.load(file)['embeddings']
# labels = np.load('C:/Users/qyx1327/Documents/results/tvae/test_video_embeddings/results_1_labels.npy')
# labels = labels[4:-5]
# plt.scatter(x = test[:, 0], y = test[:, 1])
labels = gm[0].predict(embed)

umap_embed = umap.UMAP(n_neighbors=100, min_dist=0.01).fit(embed)
umap.plot.points(umap_embed, labels=labels)
plt.legend('')
plt.title('z=8')
plt.show()

#%% use my own code for plotting

umap_embed = umap.UMAP(n_neighbors=10, min_dist=0.01).fit(embed)
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
plt.legend()
plt.xlim([x.min()*1.1, x.max()*1.5])
plt.title('m485_clo01_amph, aligned')
plt.show()


#%% cluster the umap embedding
umap_tf = umap_embed.transform(embed)
from sklearn.cluster import KMeans

km = KMeans(n_clusters=15, random_state=0).fit(umap_tf)
km_labels = km.predict(umap_tf)

# try gmm on umap embedding
umap_gm = GaussianMixture(n_components=4).fit(umap_tf)
gm_labels = umap_gm.predict(umap_tf)

umap.plot.points(umap_embed, labels=gm_labels)
# plt.legend('')
plt.title('umap cluster, gmm')
plt.show()
#%% try tsne
from sklearn.manifold import TSNE
import seaborn as sns

tsne_embed = TSNE(perplexity=70).fit_transform(embed)
palette = np.array(sns.color_palette('hls', 40))
labels = labels-1
plt.figure()
plt.scatter(x=tsne_embed[:, 0], y=tsne_embed[:, 1], c=palette[labels.astype(int)])
plt.title('perplexity=70')
plt.show()

#%%
plt.plot(embed[100:200, 0])
plt.show()


plt.hist(compute_bout_length(labels)[2])
plt.show()