import numpy as np
a = np.load('dataset/data_24k/001_030_raw.npy',allow_pickle=True)
print(a[1].shape)
b = np.load('dataset/data_24k/001_030.npy',allow_pickle=True)
print(b[1].shape)