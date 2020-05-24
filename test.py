import src.mibi_dataloader as md
from scipy import io as scipy_io
import numpy as np



main_folder = '/Users/raymondbaranski/Desktop/data/'

point1 = md.loadPoint_tiffFolder(main_folder + 'tif_folders/Point1', 'TIFs')
print(type(point1['tags']['Au']['XResolution']))
print(point1['tags']['Au']['XResolution'])

md.savePoint_mat(point1, main_folder + 'tif_folders/Point1')

point2 = md.loadPoint_mat(main_folder + 'tif_folders/Point1.mat')
print(type(point2['tags']['Au']['XResolution']))
print(point2['tags']['Au']['XResolution'])

md.savePoint_mat(point2, main_folder + 'tif_folders/Point1_1')
point3 = md.loadPoint_mat(main_folder + 'tif_folders/Point1_1.mat')
print(type(point3['tags']['Au']['XResolution']))
print(point3['tags']['Au']['XResolution'])