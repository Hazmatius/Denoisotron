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


class MIBIData:
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


class EstimatorLoader(MIBIData):
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


class DenoiserLoader(MIBIData):
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