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

def loadMat(dir, name):
    struct = scipy_io.loadmat(dir + name + '.mat')
    loadname = getname(struct.keys())
    data = struct[loadname]
    return torch.tensor(data)

def getname(some_names):
    for name in some_names:
        if name[1] != '_':
            return name
    return 0

def saveTensorAsMat(dir, name, data):
    struct = {}
    struct[name] = data.detach().cpu().numpy()
    scipy_io.savemat(dir + name + '.mat', struct)

def netEstimate(dir, name, network):
    data = loadMat(dir, name).float()
    if data.dim() == 2:
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 3:
        data = data.permute(2, 0, 1).unsqueeze(1)
    else:
        return 0
    ndata = network.process(data)
    saveTensorAsMat(dir, 'lest_' + name, ndata[0, 0, :, :])



estimator_folder = '/Users/raymondbaranski/GitHub/Denoisotron/models/'
estimator_name = 'estimator-2'

input_path = '/Volumes/G-DRIVE USB/bangelo_data/BANGELO_LAB/190615/extracted/Point1'
output_path = '/Volumes/G-DRIVE USB/Point1'

data_path = '/Users/raymondbaranski/GitHub/Noise_Estimation/data/'
estimator = SelfSupervisedEstimator.load_model(estimator_folder, estimator_name)

# for i in range(1,151):
#     print(i)
#     netEstimate(data_path, 'rnoise1_' + str(i), estimator)
#     netEstimate(data_path, 'rnoise2_' + str(i), estimator)



netEstimate(data_path, 'pnoise', estimator)
# netEstimate(data_path, 'counts2', estimator)
# netEstimate(data_path, 'counts3', estimator)

#
# point = loadTIFF_folder(input_path)
# point = estimate_lambda(point, estimator)
# savePoint_mat(point, output_path)