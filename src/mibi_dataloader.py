import torch
import os
import gc
from skimage import io
import numpy as np
import random
from random import randint
from random import choice
import math
import copy
import pickle
from scipy.ndimage import gaussian_filter
import exifread
from skimage import io as skimage_io
from scipy import io as scipy_io
from PIL import Image, ImageSequence
from PIL.TiffTags import TAGS
import csv


def get_labels_from_csv(csvpath):
    label_dict = dict()
    with open(csvpath) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        linenum = 0
        for row in csvreader:
            linenum += 1
            if linenum != 1:
                point = row[0]
                label = row[1]
                if label.isnumeric():
                    label = float(label)
                label_dict[point] = label
    return label_dict


def get_tensor_from_dict(point, field, **kwargs):
    """
    :param point: a dictionary expected to contain a 'markers' key, corresponding to 'labels' in MATLAB-verse. Note,
    should be either a list or an np-array object array of strings
    :param field: the field you wish to extract extract from the point as a matrix, such as 'counts'
    :param kwargs: you may optionally specify which markers you want to extract with the 'markers' key-word argument,
    specify as an iterable of strings.
    :return: a 32-bit floating point tensor that is M x W x H, where M is the number of markers, W and H
    are width and height of the variable specified by 'field'
    """
    markers = point['markers']
    test_img = torch.tensor(point['counts'][markers[0]])
    width = test_img.size(0)
    height = test_img.size(1)

    if 'markers' in kwargs:
        tensor = torch.zeros([len(kwargs['markers']), width, height]).float()
        for i in range(len(kwargs['markers'])):
            marker = kwargs['markers'][i]
            tensor[i, :, :] = torch.tensor(point[field][marker]).float()
    else:
        tensor = torch.zeros([len(markers), width, height]).float()
        for i in range(len(markers)):
            marker = markers[i]
            tensor[i, :, :] = torch.tensor(point[field][marker]).float()
    return tensor, markers


def loadmat(filename):
    '''
    RIPPED off of stackoverflow
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy_io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    RIPPED off of stackoverflow
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy_io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    RIPPED off of stackoverflow
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy_io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadpoint_mat_file(point_path):
    point = loadmat(point_path)
    markers = point['markers']
    for i in range(len(markers)):
        markers[i] = markers[i].strip()
    point['markers'] = markers.tolist()
    return point


def savepoint_mat(point, point_path):
    point['markers'] = np.array(point['markers'], dtype=np.object)
    scipy_io.savemat(point_path + '.mat', point)


# Used to load a Point organized as a folder of tif images
def loadpoint_tiff_folder(point_path, subfolder='TIFs'):
    contents = listdir(os.path.join(point_path, subfolder))
    tifs = [i for i in contents if i.endswith('.tif') or i.endswith('.tiff')]
    tifs.sort()

    markers = list()
    counts = dict()
    tags = dict()
    for tif in tifs:
        marker = tif.split('.')[0]
        markers.append(marker)
        counts[marker] = skimage_io.imread(os.path.join(point_path, subfolder, tif)).astype(int)

        tif = Image.open(os.path.join(point_path, subfolder, tif))
        subtags = dict()
        for key in tif.tag:
            tagname = TAGS.get(key, 'missing')
            subtags[tagname] = tif.tag[key][0]
        tags[marker] = subtags
    point = dict()
    point['counts'] = counts
    point['markers'] = markers
    point['tags'] = tags
    point['path'] = point_path
    return point


# Used to load a Point organized as a multi-page tif
def loadpoint_tiff_multi(point_path):
    multitiff = Image.open(point_path)

    markers = list()
    counts = dict()
    tags = dict()

    for i in range(multitiff.n_frames):
        multitiff.seek(i)
        subtags = dict()
        for key in multitiff.tag:
            tagname = TAGS.get(key, 'missing')
            subtags[tagname] = multitiff.tag[key]
            if tagname == 'PageName':
                marker = multitiff.tag[key][0]
        markers.append(marker)
        counts[marker] = np.array(multitiff).astype(int)
        tags[marker] = subtags

    point = dict()
    point['counts'] = counts
    point['markers'] = markers
    point['tags'] = tags
    point['path'] = point_path

    return point


def loadpoint_tiff_single(tif_path):
    tif = Image.open(tif_path)

    markers = list()
    counts = dict()
    tags = dict()

    marker = 'unknown'
    subtags = dict()
    for key in tif.tag:
        tagname = TAGS.get(key, 'missing')
        subtags[tagname] = tif.tag[key]
        if tagname == 'PageName':
            marker = tif.tag[key][0]
    if marker == 'unknown':
        marker = os.path.basename(tif_path).split('.')[0].split('_')[-1]
    markers.append(marker)
    counts[marker] = np.array(tif).astype(int)
    tags[marker] = subtags

    point = dict()
    point['counts'] = counts
    point['markers'] = markers
    point['tags'] = tags
    point['path'] = tif_path

    return point


def get_num_channels(point):
    return len(point['markers'])


def get_size(point, dim):
    test_marker = point['markers'][0]
    return point['counts'][test_marker].shape[dim]


def listdir(folder):
    contents = os.listdir(folder)
    contents = [i for i in contents if not i.startswith('.')]
    return contents


class PointReader:
    """
    Used to load the raw data of a MIBI dataset, the main function is "load_data", which takes in 4 main arguments:
    data_folder specifies where the data is
    label_type specifies whether and how the data is labeled
    input_format specifies the format of the raw data to be read
    output_format specifies the desired organization of the data for the MIBIDataLoader
    kwargs are option arguments, more details in the load_data docstring
    """
    # data_folder is the folder full of data
    # label_type is one of ['none', 'pixel', 'point']
    # input_format is one of ['tiff', 'multitiff', 'tiffolder', 'matfile']
    # output_format is one of ['point', 'marker', 'pixel']

    @staticmethod
    def load_data(data_folder, input_format, output_format, label_args, **kwargs):
        """
        :param data_folder: directory where all data for the dataset is stored
        :param label_type: label_type should be one of ['none', 'pixel', 'point']. If 'none', assumes that data is just
        strewn about about inside of the data_folder. If 'pixel', assumes that each point is nested inside of a folder
        that also contains a label image containing pixel-wise labels. If 'point' assumes that the data_folder contains
        label folders, named by their label, inside of each being all the points with that label
        :param input_format: input_format should be one of ['tiff', 'multitiff', 'tifffolder', 'matfile']. If 'tiff',
        points are assumed to be single-image tiff files (presumably channels have been seperated). If 'multitiff'
        points are expected to be multi-page tiffs, with each page corresponding to a channel. If 'tifffolder', points
        are expected to be organized as Point folders containing some substructure (usually a TIFs folder) containting
        single-page TIF files corresponding to individual channels. If 'matfile', points are expected to be organized as

        :param output_format:
        :param kwargs:
        :return:
        """
        results = PointReader.get_metadata(data_folder, label_args, **kwargs)
        results = PointReader.get_raw_data(results, input_format, **kwargs)
        results = PointReader.reorganize_data(results, output_format, **kwargs)
        results = PointReader.organize_labels(results, output_format, **kwargs)
        del results['points']

        return results

    @staticmethod
    def get_metadata(data_folder, label_args, **kwargs):

        results = dict()
        contents = listdir(data_folder)
        if label_args == 'none':
            data_paths = [os.path.join(data_folder, i) for i in contents]
            results['data_paths'] = data_paths

        else:
            label_format = label_args['label_format']
            if label_format == 'image':
                contents = [i for i in contents if os.path.isdir(os.path.join(data_folder, i))]
                data_paths = [os.path.join(data_folder, i, 'data') for i in contents]
                label_paths = [os.path.join(data_folder, i, 'label.tif') for i in contents]
                labels = [os.imread(i) for i in label_paths]
                results['data_paths'] = data_paths
                results['point_labels'] = labels
            elif label_format == 'folder':
                all_labels = [i for i in contents if os.path.isdir(os.path.join(data_folder, i))]
                data_paths = list()
                labels = list()
                for label in all_labels:
                    subcontents = listdir(os.path.join(data_folder, label))
                    data_paths.extend([os.path.join(data_folder, label, i) for i in subcontents])
                    labels.extend([label for i in subcontents])
                    results['data_paths'] = data_paths
                    results['point_labels'] = labels
            elif label_format == 'csv':
                csvpath = label_args['csv_path']
                contents = [i for i in contents if os.path.isdir(os.path.join(data_folder, i))]
                data_paths = [os.path.join(data_folder, i, 'data') for i in contents]
                label_dict = get_labels_from_csv(csvpath)
                labels = [label_dict[i.split('.')[0]] for i in contents]
                results['data_paths'] = data_paths
                results['point_labels'] = labels
            else:
                raise SystemExit('Error: label_type must be one of [\'none\', \'pixel\', \'point\'].')

        return results

    @staticmethod
    def get_raw_data(results, input_format, **kwargs):
        # Loading the data
        if input_format == 'tiff':
            points = [loadpoint_tiff_single(i) for i in results['data_paths']]
        elif input_format == 'multitiff':
            points = [loadpoint_tiff_multi(i) for i in results['data_paths']]
        elif input_format == 'tifffolder':
            points = [loadpoint_tiff_folder(i) for i in results['data_paths']]
        elif input_format == 'matfile':
            points = [loadpoint_mat_file(i) for i in results['data_paths']]
        else:
            raise SystemExit('Error: input_format must be one of [\'tiff\', \'multitiff\', \'tifffolder\', \'matfile\']')

        if 'point_labels' in results:
            for i in range(len(points)):
                points[i]['point_label'] = results['point_labels'][i]
        results['points'] = points

        return results

    @staticmethod
    def reorganize_data(results, output_format, label_args, **kwargs):
        # Organize the data
        n_points = len(results['points'])
        n_channels = get_num_channels(results['points'][0])
        width = get_size(results['points'][0], 0)
        height = get_size(results['points'][0], 0)

        results['samples'] = list()
        results['sources'] = list()
        if label_args != 'none':
            results['raw_labels'] = list()

        if output_format == 'point':
            for point in results['points']:
                array, markers = get_tensor_from_dict(point, 'counts')
                results['samples'].append(array.unsqueeze(0))
                results['sources'].append(point['path'])
                if label_args != 'none':
                    results['raw_labels'].append(point['point_label'])

        elif output_format == 'marker':
            for point in results['points']:
                for j in range(n_channels):
                    marker = point['markers'][j]
                    results['samples'].append(torch.tensor(point['counts'][marker]).unsqueeze(0).unsqueeze(0).float())
                    results['sources'].append(point['path'] + ':' + marker)
                    if label_args != 'none':
                        results['raw_labels'].append(point['point_label'])

        elif output_format == 'pixel':
            for point in results['points']:
                array, markers = get_tensor_from_dict(point, 'counts')
                for i in range(array.shape[0]):
                    for j in range(array.shape[1]):
                        results['samples'].append(array[:, i, j].unsqueeze(0))
                        results['sources'].append(point['path'] + ':[' + str(i) + ',' + str(j) + ']')
                        if label_args != 'none':
                            label_format = label_args['label_format']
                            if label_format == 'image':
                                results['raw_labels'].append(point['point_label'][i, j])
                            else:
                                results['raw_labels'].append(point['point_label'])

        else:
            raise SystemExit('Error: output_format must be one of [\'point\', \'marker\', \'pixel\']')

        return results

    @staticmethod
    def organize_labels(results, label_args, **kwargs):
        if label_args == 'none':
            pass
        else:
            if label_args['label_type'] == 'regression':
                results['labels'] = results['raw_labels']
            elif label_args['label_type'] == 'categorical':
                label_dict = label_args['label_dict']
                if label_args['label_format'] == 'image':
                    results['labels'] = [int(i) for i in results['raw_labels']]
                elif label_args['label_format'] == 'folder':
                    results['labels'] = [label_dict[i] for i in results['raw_labels']]
                elif label_args['label_format'] == 'csv':
                    if isinstance(results['raw_labels'][0], str):
                        results['labels'] = [label_dict[i] for i in results['raw_labels']]
                    else:
                        results['labels'] = [int(i) for i in results['raw_labels']]
                n_classes = len(label_dict.keys())
                results['labels_onehot'] = [torch.zeros(1,n_classes)._scatter(1,torch.tensor(i).unsqueeze(0).long(),1) for i in results['labels']]
        return results



class FlatTIFLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_folder_of_points(**kwargs):
        directory = kwargs['folder']

        files = os.listdir(directory)
        random.shuffle(files)

        points = list()
        for file in files:
            if not file[0] == '.':
                print(file)
                points.append(loadpoint_mat_file(directory + file))

        test_point = points[0]
        labels = test_point['labels']
        num_channels = len(labels)

        pixels = torch.zeros([0, num_channels]).float()

        for point in points:
            point_pixels, labels = get_tensor_from_dict(point)
            point_pixels = torch.flatten(point_pixels, start_dim=1).float().transpose(0,1)
            pixels = torch.cat([pixels, point_pixels], dim=0)

        return pixels, labels


class FlatMIBIData:
    def __init__(self, **kwargs):
        tiff_loader = FlatTIFLoader()

        self.pixels, self.labels = tiff_loader.load_folder_of_points(**kwargs)
        self.num_labels = len(self.labels)
        self.num_samples = self.pixels.size(0)

    # dummy function
    def set_crop(self, crop):
        pass

    def get_samples(self, sample_indices, flatten):
        pxs = torch.zeros([len(sample_indices), self.num_labels], dtype=torch.float32)
        for i in range(len(sample_indices)):
            pxs[i, :] = self.pixels[sample_indices[i], :]
        samples = {
            'x': pxs.cuda()
        }
        return samples

    def prepare_epoch(self):
        self.sample_queue = np.random.permutation(int(self.num_samples))

    def get_epoch_length(self):
        return int(self.num_samples)

    def get_next_minibatch_idxs(self, minibatch_size):
        if len(self.sample_queue) == 0:  # there is nothing left in the minibatch queue
            return None
        elif len(self.sample_queue)<minibatch_size:  # we just have to return the last of the dataset
            return None
            # minibatch_idxs = np.copy(self.sample_queue)
            # self.sample_queue = np.array([])
            # return minibatch_idxs
        else:  # we return a normal minibatch
            minibatch_idxs = np.copy(self.sample_queue[0:minibatch_size])
            self.sample_queue = self.sample_queue[minibatch_size:]
            return minibatch_idxs

    def get_next_minibatch(self, minibatch_size):
        sample_idxs = self.get_next_minibatch_idxs(minibatch_size)
        if sample_idxs is None:
            return None
        else:
            return self.get_samples(sample_idxs, False)


class MultiTIFLoader:
    def __init__(self):
        pass

    @staticmethod
    def flip(x):
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        return x

    @staticmethod
    def blur_augment(x):
        x_4 = torch.tensor(gaussian_filter(x, 4)).unsqueeze(0)
        x_16 = torch.tensor(gaussian_filter(x, 16)).unsqueeze(0)
        x_64 = torch.tensor(gaussian_filter(x, 64)).unsqueeze(0)
        x_128 = torch.tensor(gaussian_filter(x, 128)).unsqueeze(0)
        x = torch.tensor(x).unsqueeze(0)
        y = torch.cat([x, x_4, x_16, x_64, x_128])
        return y

    @staticmethod
    def load_and_augment_tifs(**kwargs):
        directory = kwargs['folder']

        gc.disable()
        files = os.listdir(directory)
        tifs = list()
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                tifs.append(file)
        # print(tifs)

        if 'num' in kwargs:
            num_tifs = kwargs['num']
        else:
            num_tifs = len(tifs)

        test_img = io.imread(os.path.join(directory, tifs[0])).astype(int)
        test_img = torch.tensor(test_img, dtype=torch.uint8)

        if 'flip' in kwargs:
            test_img = MultiTIFLoader.flip(test_img)

        data_width = test_img.shape[0]
        data_height = test_img.shape[1]

        # augment with blurs of 4, 16, 64, 128
        images = torch.zeros([num_tifs, 1+4, data_width, data_height])
        for i in range(num_tifs):
            tif = tifs[i]

            print('\rLoading.......' + str(100 * i / num_tifs) + '%', end='')
            path = os.path.join(directory, tif)
            img = io.imread(path).astype(float)
            img = MultiTIFLoader.blur_augment(img)

            if 'flip' in kwargs:
                img = MultiTIFLoader.flip(img)
            # expectation is that we'll have [channels, size, size]
            images[i, :, :, :] = img/255.0
            # print(torch.max(images[i, :, :, :]))
            # print()
        gc.enable()
        return images, tifs, data_width, data_height

    @staticmethod
    def load_multipage_tifs(**kwargs):
        directory = kwargs['folder']

        gc.disable()
        files = os.listdir(directory)
        random.shuffle(files)
        try:
            if 'number' in kwargs:
                files = files[0:(kwargs['number']-1)]
            else:
                files = files[0:1000]
        except:
            pass

        # print(files)
        tifs = list()
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                tifs.append(file)
                # print(file)
        # print(tifs)

        if 'num' in kwargs:
            num_tifs = kwargs['num']
        else:
            num_tifs = len(tifs)

        test_img = io.imread(os.path.join(directory, tifs[0])).astype(int)
        test_img = torch.tensor(test_img, dtype=torch.uint8)

        if 'flip' in kwargs:
            test_img = MultiTIFLoader.flip(test_img)

        data_width = test_img.shape[0]
        data_height = test_img.shape[1]

        images = torch.zeros([num_tifs, 1, data_width, data_height])
        for i in range(num_tifs):
            tif = tifs[i]

            print('\rLoading.......' + str(100 * i / num_tifs) + '%', end='')
            path = os.path.join(directory, tif)
            img = io.imread(path)
            img = torch.tensor(img)

            if 'flip' in kwargs:
                img = MultiTIFLoader.flip(img)
            # expectation is that we'll have [channels, size, size]
            images[i, :, :, :] = img
            # print(torch.max(images[i, :, :, :]))
            # print()
        gc.enable()
        return images, tifs, data_width, data_height

    @staticmethod
    def load_folder(**kwargs):
        imagelist = list()
        labellist = list()

        tiff_data = MultiTIFLoader.load_multipage_tifs(**kwargs)
        imagelist.append(tiff_data[0])
        labellist.append(torch.tensor([0]).repeat(imagelist[-1].shape[0], 1))
        nameslist = tiff_data[1]

        images = torch.cat(imagelist, 0)
        labels = torch.cat(labellist, 0)
        source = nameslist

        return images, labels, source

    @staticmethod
    def load_augmented(**kwargs):
        imagelist = list()
        labellist = list()

        tiff_data = MultiTIFLoader.load_and_augment_tifs(**kwargs)
        imagelist.append(tiff_data[0])
        labellist.append(torch.tensor([0]).repeat(imagelist[-1].shape[0], 1))
        nameslist = tiff_data[1]

        images = torch.cat(imagelist, 0)
        labels = torch.cat(labellist, 0)
        source = nameslist

        return images, labels, source

    @staticmethod
    def load_label_folder(**kwargs):
        imagelist = list()
        labellist = list()
        nameslist = list()

        folder = kwargs['folder']
        for label in kwargs['labels']:
            kwargs['folder'] = folder + label
            tiff_data = MultiTIFLoader.load_multipage_tifs(**kwargs)
            imagelist.append(tiff_data[0])
            labellist.append(torch.tensor(kwargs['labels'][label]).repeat(imagelist[-1].shape[0], 1))
            nameslist = nameslist + tiff_data[1]

        images = torch.cat(imagelist, 0)
        labels = torch.cat(labellist, 0)
        source = nameslist

        return images, labels, source

    @staticmethod
    def load_complex(folder, point, ext, in_args):
        # we assume that kwargs is going to hold
        args = copy.deepcopy(in_args)
        for arg in args:
            img = io.imread(os.path.join(folder, point, args[arg] + ext)).astype(np.int32)
            args[arg] = torch.tensor(img).float()
        return args

    @staticmethod
    def load_complex_folder(**kwargs):
        directory = kwargs['complex_folder']
        ext = '.tiff'
        args = {
            'x': 'tifdata',
            'c': 'labeldata'
        }
        folders = list()

        elements = os.listdir(directory)
        for element in elements:
            if element.endswith(''):
                folders.append(element)
        # let's assume that folders somehow only contains folders
        num_tifs = len(folders)
        # our test folder
        cmplx = MultiTIFLoader.load_complex(directory, folders[0], ext, args)
        num_channels = cmplx['x'].shape[0]
        data_width = cmplx['x'].shape[1]
        data_height = cmplx['x'].shape[2]

        images = torch.zeros([num_tifs, num_channels, data_width, data_height])
        labels = torch.zeros([num_tifs, 1, data_width, data_height])
        for i in range(num_tifs):
            folder = folders[i]
            print('\rLoading.......' + str(100 * i / num_tifs) + '%', end='')
            cmplx = MultiTIFLoader.load_complex(directory, folder, ext, args)
            images[i, :, :, :] = cmplx['x']
            labels[i, :, :, :] = cmplx['c']
        print()
        return images, labels, folders


class MIBIDataLoader:
    def __init__(self, **kwargs):
        tiff_loader = MultiTIFLoader()

        self.flag = 'normal'
        if 'flag' in kwargs:
            self.flag = kwargs['flag']

        if 'datasets' in kwargs:
            imageslist = list()
            labelslist = list()
            sourcelist = list()
            for dataset in kwargs['datasets']:
                imageslist.append(dataset.images)
                labelslist.append(dataset.labels)
                sourcelist.append(dataset.source)
            self.images = torch.cat(imageslist, 0)
            self.labels = torch.cat(labelslist, 0)
            self.source = sum(sourcelist, [])
        elif 'labels' in kwargs and 'images' not in kwargs:
            self.images, self.labels, self.source = tiff_loader.load_label_folder(**kwargs)
            self.labeldict = kwargs['labels']
            for label in self.labeldict.keys():
                self.labeldict[label] = torch.tensor(self.labeldict[label])
        elif 'folder' in kwargs:
            if self.flag == 'aug':
                self.images, self.labels, self.source = tiff_loader.load_augmented(**kwargs)
            else:
                self.images, self.labels, self.source = tiff_loader.load_folder(**kwargs)
        elif 'complex_folder' in kwargs:
            self.images, self.labels, self.source = MultiTIFLoader.load_complex_folder(**kwargs)
            self.flag = 'complex'
        else:
            self.images = kwargs['images']
            if 'labels' in kwargs:
                self.labels = kwargs['labels']
            self.source = kwargs['source']
        
        self.num_points = self.images.size()[0]
        self.image_shape = self.images[0].size()  # should be [num_channels, width, height]
        self.num_channels = self.image_shape[0]
        self.channels = self.num_channels

        self.batchsize = 100
        self.crop = 32
        self.stride = 16
        self.crop_limit = self.image_shape[1] - self.crop

        if 'crop' in kwargs:
            self.set_crop(kwargs['crop'])
        if 'stride' in kwargs:
            self.set_stride(kwargs['stride'])
        if 'scale' in kwargs:
            for i in range(len(self.images)):
                self.images[i] = self.images[i] * kwargs['scale']

        print('There are ', int(self.vidxmax * self.num_points), 'samples')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    # should be overwritten
    def get_image(self, sample_index):
        point_index = np.floor(sample_index / self.vidxmax).astype(int)
        vindex = sample_index % self.vidxmax
        x, y = self.vind2sub(vindex)
        rot = randint(0, 4)
        img_crop = self.crop_image(self.images[point_index], x, y)
        if self.flag == 'normal':
            x_img = self.rotate(img_crop, rot)
            return x_img
        else:
            lab_crop = self.crop_image(self.labels[point_index], x, y)
            lab_crop = lab_crop * (2**(torch.randn(1)/2))
            return self.rotate(img_crop, rot), self.rotate(lab_crop, rot)

    # should be overwwritten
    def get_samples(self, sample_indices, flatten):
        x_imgs = torch.zeros([len(sample_indices), 1, self.crop, self.crop], dtype=torch.float32)
        for i in np.arange(len(sample_indices)):
            j = sample_indices[i]
            x_imgs[i, :] = self.get_image(j)
        samples = {
            'x': x_imgs.float().cuda()
        }
        return samples

    # we know we have self.image_width and self.image_height
    # we also have self.crop
    # we need to define a stride somewhere

    # we need a way to prepare mini batches
    # each full epoch we need to go through all (self.num_points*self.vidxmax) samples

    # each epoch take a random permutation of all numbers in [0, self.num_points*self.vidxmax)
    # each minibatch take the next 'minibatchsize' set of numbers from this list
    # the image index we need is going to be float(sample_index / self.vidxmax)
    # the crop index is going to mod(sample_index, self.vidxmax)

    def set_crop(self, crop):
        self.crop = crop
        self.calculate_vdims()
        print(self.vidxmax)

    def set_stride(self, stride):
        self.stride = stride
        self.calculate_vdims()

    def calculate_vdims(self):
        # figure out how many crop windows there are across
        self.v_width = np.floor((self.image_shape[1]-self.crop)/self.stride).astype(int)
        self.v_height = np.floor((self.image_shape[2]-self.crop)/self.stride).astype(int)
        self.vidxmax = self.v_width * self.v_height
        self.crop_limit = self.image_shape[1] - self.crop

    # virtual index to subscript
    def vind2sub(self, index):
        # we should probably have already calculated the virtual_width and virtual_height
        y = np.floor(index / self.v_width).astype(int)
        x = (index - self.v_width * y).astype(int)
        y = y * self.stride
        x = x * self.stride
        return x, y

    def prepare_epoch(self):
        self.sample_queue = np.random.permutation(int(self.vidxmax * self.num_points))

    def get_epoch_length(self):
        return int(self.vidxmax * self.num_points)

    def crop_image(self, image, x, y):
        return image[:, x:(x + self.crop), y:(y + self.crop)]

    def random_rotate(self, image):
        a = randint(0, 4)
        if a == 0:
            return image
        elif a == 1:
            return image.transpose(1, 2)
        elif a == 2:
            return image.flip(1)
        else:
            return image.transpose(1, 2).flip(2)

    def rotate(self, image, a):
        if a == 0:
            return image
        elif a == 1:
            return image.transpose(1, 2)
        elif a == 2:
            return image.flip(1)
        else:
            return image.transpose(1, 2).flip(2)

    def get_point_indices(self, sample_indices):
        point_indices = np.floor(sample_indices / self.vidxmax).astype(int)
        return point_indices

    def get_next_minibatch_idxs(self, minibatch_size):
        if len(self.sample_queue) == 0:  # there is nothing left in the minibatch queue
            return None
        elif len(self.sample_queue)<minibatch_size:  # we just have to return the last of the dataset
            return None
            # minibatch_idxs = np.copy(self.sample_queue)
            # self.sample_queue = np.array([])
            # return minibatch_idxs
        else:  # we return a normal minibatch
            minibatch_idxs = np.copy(self.sample_queue[0:minibatch_size])
            self.sample_queue = self.sample_queue[minibatch_size:]
            return minibatch_idxs

    def get_next_minibatch(self, minibatch_size):
        sample_idxs = self.get_next_minibatch_idxs(minibatch_size)
        if sample_idxs is None:
            return None
        else:
            return self.get_samples(sample_idxs, False)

    # legacy function
    def random_crop(self, image):
        a = randint(0, self.crop_limit)
        b = randint(0, self.crop_limit)
        return image[:, a:(a + self.crop), b:(b + self.crop)]

    # legacy function
    def get_batch(self, batchsize, flatten):
        # we're going to return batchsize randomly cropped images pulled with replacement
        # we'll also return the labels for this batch
        sample_indices = [randint(0, self.__len__() - 1) for p in range(0, batchsize)]

        if flatten:
            sample = torch.zeros([batchsize, self.channels * self.crop * self.crop], dtype=torch.float32)
            for j in range(batchsize):
                # randcrop, a, b =
                img = self.random_rotate(self.random_crop(self.__getitem__(sample_indices[j])))
                sample[j, :] = img.reshape([torch.numel(img)])
        else:
            sample = torch.zeros([batchsize, self.channels, self.crop, self.crop], dtype=torch.float32)
            for j in range(batchsize):
                sample[j, :] = self.random_rotate(self.random_crop(self.__getitem__(sample_indices[j])))
        batch = {
            'x': torch.tensor(sample).float().cuda(),
            'c': self.labels[sample_indices].long().squeeze(1).cuda()
        }
        return batch

    def pickle(self, filepath):
        pickling_on = open(filepath, 'wb')
        pickle.dump(self, pickling_on, protocol=4)
        pickling_on.close()

    @staticmethod
    def depickle(filepath):
        pickle_off = open(filepath, 'rb')
        dataset = pickle.load(pickle_off)
        return dataset


class EstimatorLoader(MIBIDataLoader):
    def __init__(self, **kwargs):
        super(EstimatorLoader, self).__init__(**kwargs)

    def get_image(self, sample_index):
        point_index = np.floor(sample_index / self.vidxmax).astype(int)
        vindex = sample_index % self.vidxmax
        x, y = self.vind2sub(vindex)
        rot = randint(0, 4)
        img_crop = self.crop_image(self.images[point_index], x, y)
        x_img = self.rotate(img_crop, rot)

        # pick a random lambda value to corrupt the raw image
        lam = torch.rand(1).item() * 5
        l_img = torch.zeros(1, x_img.shape[1], x_img.shape[2]) + lam
        noise = torch.tensor(np.random.poisson(lam, [1, x_img.shape[1], x_img.shape[2]]))

        # corrupt the raw image
        n_img = x_img + noise.float()
        return x_img, n_img, l_img

    # should be overwwritten
    def get_samples(self, sample_indices, flatten):
        x_imgs = torch.zeros([len(sample_indices), self.num_channels, self.crop, self.crop], dtype=torch.float32)
        n_imgs = torch.zeros([len(sample_indices), self.num_channels, self.crop, self.crop], dtype=torch.float32)
        l_imgs = torch.zeros([len(sample_indices), 1, self.crop, self.crop], dtype=torch.float32)
        for i in np.arange(len(sample_indices)):
            j = sample_indices[i]
            x_imgs[i, :], n_imgs[i, :], l_imgs[i, :] = self.get_image(j)
        samples = {
            'x': x_imgs.float().cuda(),
            'nx': n_imgs.float().cuda(),
            'l': l_imgs.float().cuda()
        }
        return samples


class DenoiserLoader(MIBIDataLoader):
    def __init__(self, **kwargs):
        super(DenoiserLoader, self).__init__()

    def get_image(self, sample_index):
        point_index = np.floor(sample_index / self.vidxmax).astype(int)
        vindex = sample_index % self.vidxmax
        x, y = self.vind2sub(vindex)
        rot = randint(0, 4)
        img_crop = self.crop_image(self.images[point_index], x, y)
        if self.flag == 'normal':
            x_img = self.rotate(img_crop, rot)
            lam = torch.rand(1).item()*5
            l_img = torch.zeros(x_img.shape)+lam
            noise = torch.tensor(np.random.poisson(lam, x_img.shape))
            n_img = x_img + noise.float()
            return x_img, n_img, l_img
        else:
            lab_crop = self.crop_image(self.labels[point_index], x, y)
            return self.rotate(img_crop, rot), self.rotate(lab_crop, rot)

    # should be overwwritten
    def get_samples(self, sample_indices, flatten):
        x_imgs = torch.zeros([len(sample_indices), 1, self.crop, self.crop], dtype=torch.float32)
        n_imgs = torch.zeros([len(sample_indices), 1, self.crop, self.crop], dtype=torch.float32)
        l_imgs = torch.zeros([len(sample_indices), 1, self.crop, self.crop], dtype=torch.float32)
        for i in np.arange(len(sample_indices)):
            j = sample_indices[i]
            x_imgs[i, :], n_imgs[i, :], l_imgs[i, :] = self.get_image(j)
        samples = {
            'x': x_imgs.float().cuda(),
            'nx': n_imgs.float().cuda(),
            'l': l_imgs.float().cuda()
        }
        return samples