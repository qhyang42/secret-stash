#%% cluster embedding with GMM
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import glob

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


#%%
gm_bic = np.array([])

for n_comp in range(140, 150):
    gm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=1000).fit(embed_train)
    gm_bic = np.append(gm_bic, gm.bic(embed_test))


#%% save the gmms

# gm_aic = np.array([])
gm = []
gm_bic = np.array([])
for n_comp in range(5, 100):
    gm.append(GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=1000).fit(embed_train))
    gm_bic = np.append(gm_bic, gm[-1].bic(embed_test))

with open('msplit_gmm_model.pickle', 'wb') as fhandle:
    pickle.dump(gm, fhandle)

#%% plot
# xaxis = np.arange(5, 100)
plt.plot(gm_bic)
# plt.xlim([5, 100])
# plt.xticks(np.arange(5, 100, 20))
plt.xlabel('n_component')
plt.ylabel('bic')
plt.title('manual split training/testing')
plt.show()

# np.save('C:/Users/qyx1327/Documents/results/msplit_bic.npy', gm_bic)
#%%
with open('gmm_model.pickle', 'rb') as fhandle:
    gm = pickle.load(fhandle)

gm_bic = np.array([])
for m_idx in range(95):
    gm_bic = np.append(gm_bic, gm[m_idx].bic(embed_test))

#%% predict and save labels. run for one additional video and compare with annotation
# choose n_component = 40
with open('msplit_gmm_model.pickle', 'rb') as fhandle:
    gm = pickle.load(fhandle)
current_model = gm[35]
file_list = glob.glob('C:/Users/qyx1327/Documents/results/tvae/test_video_embeddings/*')
for file in iter(file_list):
    embed = load_split_data(file, split=False)[0] # take either one
    labels = current_model.predict(embed)

