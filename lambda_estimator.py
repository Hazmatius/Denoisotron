import os
import exifread
import torch
from scipy import io as scipy_io
from skimage import io as skimage_io
from src.modules import SelfSupervisedEstimator


def loadTIFF_folder(point_path, subfolder='TIFs'):
    data_path = point_path + '/' + subfolder + '/'
    files = os.listdir(data_path)
    tifs = list()
    for file in files:
        if file.endswith('.tif') or file.endswith('.tiff'):
            tifs.append(file)
    labels = list()
    counts = dict()
    tags = dict()
    for tif in tifs:
        label = tif.replace('.tif', '')
        labels.append(label)
        counts[label] = skimage_io.imread(data_path + '/' + tif).astype(int)
        with open(data_path + '/' + tif, 'rb') as f:
            tags[label] = exifread.process_file(f)
    point = dict()
    point['counts'] = counts
    point['labels'] = labels
    point['tags'] = tags
    return point


def savePoint_mat(point, point_path):
    scipy_io.savemat(point_path + '.mat', point)


def estimate_lambda(point, network):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    with torch.no_grad():
        network.eval()
        network.to(device)
        
        lambdas = dict()
        for label in point['labels']:
            raw_counts = torch.tensor(point['counts'][label]).unsqueeze(0).unsqueeze(0).to(device).float()
            lambda_estimate = network.process(raw_counts)
            lambda_estimate[lambda_estimate<0] = 0
            lambda_estimate = lambda_estimate[0,0,:,:].detach().cpu().numpy()
            lambdas[label] = lambda_estimate
        point['lambdas'] = lambdas
    return point




# estimator_folder = '/Users/raymondbaranski/GitHub/Denoisotron/models/'
# estimator_name = 'estimator-2'
# input_path = '/Volumes/ALEX_SSD/BANGELO_LAB/190615/extracted/Point1'
# output_path = '/Volumes/G-DRIVE USB/Point1'
#
# estimator = SelfSupervisedEstimator.load_model(estimator_folder, estimator_name)
#
# point = loadTIFF_folder(input_path)
# point = estimate_lambda(point, estimator)
# savePoint_mat(point, output_path)