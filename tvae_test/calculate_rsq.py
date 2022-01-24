#%% calculate rsq between original and reconstruct

import numpy as np
import matplotlib.pyplot as plt
data = np.load('C:/Users/qyx1327/Documents/results/tvae/results.npz')
recon = data['reconstructions']
orig  = data['originals']

# for 1st frame of all sequences
rsq = np.array([])
for i in range(orig.shape[0]):
    for j in range(orig.shape[1]):
        orig_cord = orig[i, j, :]
        recon_cord = recon[i, j, :]
        #if np.sum(orig_cord - orig_cord.mean()) != 0:
        rsd = np.sum(np.square(recon_cord-orig_cord))/np.sum(np.square(recon_cord-recon_cord.mean()))
        #rsd = np.sum(np.square(orig_cord - recon_cord)) / np.sum(np.square(orig_cord - orig_cord.mean()))
        rsq = np.append(rsq, 1-rsd)

# todo check for (0,0) in keypoints

#%% plot

plt.hist(rsq, bins=10, log=True)
#plt.xlim([0.9 ,1])
plt.title('r_squared')
plt.show()




#%% test plot coordinate
test_cord1 = orig[i, 1, :]
test_cord2 = recon[i, 1, :]

plt.scatter(x = test_cord1[::2], y = test_cord1[1::2], c = 'r')
plt.scatter(x = test_cord2[::2], y = test_cord2[1::2], c = 'b')
plt.legend(['orig', 'recon'])
plt.show()

plt.plot(test_cord1)
plt.plot(test_cord2)
plt.legend(['orig', 'recon'])
plt.show()