# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


#%%
import numpy as np
data = np.load('/Users/yang/Desktop/parker_lab_data/20191207_m085_clo04_amph_raw_feat_top_v1_8.npz', allow_pickle=[True])
keypoints = data['keypoints']

#%%
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x=keypoints[0,0,0], y=keypoints[0,0,1], c='b', label='mouse_1')
plt.scatter(x=keypoints[0,1,0], y=keypoints[0,1,1], c='r', label='mouse_2')
plt.xlim([0,1024])
plt.ylim([0,570])
plt.show()

#%%
keypoints = keypoints[:,0,:]
x_kps = keypoints[:,0]
y_kps = keypoints[:,1]
out = np.zeros(keypoints.shape)
out = out.reshape(out.shape[0], -1)
out[:,::2] = x_kps
out[:,1::2] = y_kps

#%% 
plt.figure()
plt.scatter(x=out[0, ::2], y=out[0,1::2], c='b')
plt.scatter(x=out[1, ::2], y=out[1,1::2], c='r')
plt.show()

#%% compute autocorrelation of velocity
data_smooth = data['data_smooth']
speed = data_smooth[0, :, 34] # speed
# try nose
xpos = keypoints[:, 0, 0]
ypos = keypoints[:, 0, 1]
dist = np.sqrt(np.square(xpos[1:]-xpos[:-1])+np.square(ypos[1:]-ypos[:-1]))

vel = speed
corr_mat = np.correlate(vel, vel, mode='full')
corr_mat = corr_mat[corr_mat.size//2:]

plt.figure()
plt.plot(corr_mat[0:1000])
plt.xscale('log')
plt.xlabel('log scale')
plt.title('autocorrelation of speed')
plt.show()

#%%

#%%
muscles = np.array(filtered_EMG)[:10000]

auto_corrs = 0
muscles = muscles - np.mean(muscles, axis=0)

for i, muscle in enumerate(muscles):
    auto_corrs += (np.correlate(muscle, muscle, mode="full")) / len(muscle)
auto_corrs /= 16
plt.plot(auto_corrs)



plt.figure()
plt.plot(speed)
plt.show()