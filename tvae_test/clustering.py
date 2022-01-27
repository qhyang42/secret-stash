#%% cluster embedding with GMM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# load data
embed = np.load('C:/Users/qyx1327/Documents/results/tvae/results.npz')['embeddings']
#split
idx = np.random.choice(embed.shape[0], size=np.int32(np.floor(embed.shape[0]*0.2)), replace=False)
embed_test = embed[idx, :]
mask = np.ones(embed.shape[0], dtype=bool)
mask[idx] = False
embed_train = embed[mask, :]

gm_bic = np.array([])

for n_comp in range(140, 150):
    gm = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=1000).fit(embed_train)
    gm_bic = np.append(gm_bic, gm.bic(embed_test))


#%% try aic

# gm_aic = np.array([])
gm = []

for n_comp in range(5, 100):
    gm.append(GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=1000).fit(embed_train))
    # gm_aic = np.append(gm_aic, gm[-1].aic(embed_test))



#%% todo predict and save labels 
#%% plot
# xaxis = np.arange(5, 150)
plt.plot(gm_aic)
# plt.xlim([5, 150])
# plt.xticks(np.arange(5, 150, 20))
plt.xlabel('n_component')
plt.ylabel('aic')
plt.show()