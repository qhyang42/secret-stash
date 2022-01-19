#%% prepare data for training tvae
import numpy as np
import os

datadir = 'Y:/Parkerlab/Behavior/Clozapine'
conditions = ['HighDose', 'LowDose', 'Vehicle']
conditions2 = ['Amph', 'Control']
condition_choose = []


