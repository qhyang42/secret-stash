#%% cluster embedding with GMM
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# load data
embed = np.load('C:/Users/qyx1327/Documents/results/tvae/result_msplit.npz')['embeddings']
#split
idx = np.random.choice(embed.shape[0], size=np.int32(np.floor(embed.shape[0]*0.2)), replace=False)
embed_test = embed[idx, :]
mask = np.ones(embed.shape[0], dtype=bool)
mask[idx] = False
embed_train = embed[mask, :]

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
plt.show()

#%%
with open('gmm_model.pickle', 'rb') as fhandle:
    gm = pickle.load(fhandle)

gm_bic = np.array([])
for m_idx in range(95):
    gm_bic = np.append(gm_bic, gm[m_idx].bic(embed_test))

#%% predict and save labels. run for one additional video and compare with annotation
# choose n_component = 40
current_model = gm[35]