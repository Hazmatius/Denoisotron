import os
import numpy as np


def listdir(folder):
    contents = os.listdir(folder)
    contents = [i for i in contents if (not i.startswith('.') and os.path.isdir(os.path.join(folder, i)))]
    return contents

def find_folder(search_dir, folder_name):
    print('\r' + search_dir, end='')
    results = list()
    contents = listdir(search_dir)
    # contents = [i for i in contents if exclude not in i]
    for content in contents:
        if folder_name in content:
            result = os.path.join(search_dir, content)
            results.append(result)
        else:
            sub_result = find_folder(os.path.join(search_dir, content), folder_name)
            results.extend(sub_result)
    return results

def create_point_name(pointpath):
    pointpath.replace()




search_dir = '/Volumes/G-DRIVE USB/bangelo_data/'
output_dir = '/Volumes/G-DRIVE USB/datasets/all_extracted_data/'

results = find_folder(search_dir, 'TIFs')
print('\n\n\n\n\n\n\n')
for i in results:
    if 'extracted' in i:
        print(i)










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
