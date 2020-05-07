# import os
# import numpy as np
#
# data_folder = '/home/hazmat/GitHub/Visual-Anomaly-Detection/data/Hi_AD/'
# train_folder = data_folder + 'train/'
# test_folder = data_folder + 'test/'
# files = os.listdir(data_folder)
# tifs = list()
# for file in files:
#     if file.endswith('.tif') or file.endswith('.tiff'):
#         tifs.append(file)
# np.random.shuffle(tifs)
# cutoff = np.ceil(len(tifs)*0.1)
# os.makedirs(train_folder)
# os.makedirs(test_folder)
#
# for i in range(len(tifs)):
#     if i <= cutoff:
#         os.rename(data_folder + tifs[i], test_folder + tifs[i])
#     else:
#         os.rename(data_folder + tifs[i], train_folder + tifs[i])
#
