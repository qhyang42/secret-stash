#%%
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob

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

#%% plot