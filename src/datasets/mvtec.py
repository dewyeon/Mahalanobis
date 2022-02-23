import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    def __init__(self, root_path='../data', class_name='bottle', is_train=True,
                 resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.root_path = root_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        self.download()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.mvtec_folder_path, self.class_name, phase)
        gt_dir = os.path.join(self.mvtec_folder_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

    def download(self):
        """Download dataset if not exist"""

        if not os.path.exists(self.mvtec_folder_path):
            tar_file_path = self.mvtec_folder_path + '.tar.xz'
            if not os.path.exists(tar_file_path):
                download_url(URL, tar_file_path)
            print('unzip downloaded dataset: %s' % tar_file_path)
            tar = tarfile.open(tar_file_path, 'r:xz')
            tar.extractall(self.mvtec_folder_path)
            tar.close()

        return


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class MultiModal_MVTecDataset(Dataset):
	def __init__(self, c, is_train=True):
		index_list = [int(i.strip()) for i in c.class_indexes.split(',')]
		for idx in index_list:
			assert idx in range(15), 'class_index: {}, should be between 0 to 14'.format(idx)
		
		self.dataset_path = c.data_path
		self.class_indexes = index_list
		self.is_train = is_train
		self.cropsize = c.crp_size
		# load dataset
		self.x, self.y, self.mask = self.load_dataset_folder()
		# set transforms
		if is_train:
			self.transform_x = T.Compose([
				T.Resize(c.img_size, Image.ANTIALIAS),
				T.RandomRotation(5),
				T.CenterCrop(c.crp_size),
				T.ToTensor()])
		# test:
		else:
			self.transform_x = T.Compose([
				T.Resize(c.img_size, Image.ANTIALIAS),
				T.CenterCrop(c.crp_size),
				T.ToTensor()])
		# mask
		self.transform_mask = T.Compose([
			T.Resize(c.img_size, Image.NEAREST),
			T.CenterCrop(c.crp_size),
			T.ToTensor()])

		self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

		
		""" transform setting at Padim Experiment """
		# # set transforms 
		# self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
		#							   T.CenterCrop(cropsize),
		#							   T.ToTensor(),
		#							   T.Normalize(mean=[0.485, 0.456, 0.406],
		#										   std=[0.229, 0.224, 0.225])])
		# self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
		#								  T.CenterCrop(cropsize),
		#								  T.ToTensor()])
		
	def __getitem__(self, idx):
		x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
		x = Image.open(x).convert('RGB')
		# x = Image.open(x)
		# """ ziper, screw, grid class -> gray to RGB trnasform """
		# if len(np.array(x).shape) == 2: # gray-scale image (only 1 channel)
		#	 x = np.expand_dims(np.array(x), axis=2)
		#	 x = np.concatenate([x, x, x], axis=2)
		#	 x = Image.fromarray(x.astype('uint8')).convert('RGB')
	   
		x = self.normalize(self.transform_x(x))

		if y == 0:
			mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
		else:
			if mask is None:
				mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]]) + 1.
			else:
				mask = Image.open(mask)
				mask = self.transform_mask(mask)

		return x, y, mask

	def __len__(self):
		return len(self.x)

	def load_dataset_folder(self):
		phase = 'train' if self.is_train else 'test'
		x, y, mask = [], [], []
		
		cls_list = [MVTEC_CLASS_NAMES[cls_nm] for cls_nm in self.class_indexes if cls_nm is not None]

		for cls_nm in cls_list:
			img_dir = os.path.join(self.dataset_path, cls_nm, phase)
			gt_dir = os.path.join(self.dataset_path, cls_nm, 'ground_truth')
			
			img_types = sorted(os.listdir(img_dir))
			for img_type in img_types:
				# load images
				img_type_dir = os.path.join(img_dir, img_type)
				if not os.path.isdir(img_type_dir):
					continue
				img_fpath_list = sorted([os.path.join(img_type_dir, f)
										 for f in os.listdir(img_type_dir)
										 if f.endswith('.png')])
				x.extend(img_fpath_list)

				# load gt labels
				if img_type == 'good':
					y.extend([0] * len(img_fpath_list))
					mask.extend([None] * len(img_fpath_list))
				else:
					y.extend([1] * len(img_fpath_list))
					gt_type_dir = os.path.join(gt_dir, img_type)
					img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
					gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
									for img_fname in img_fname_list]
					mask.extend(gt_fpath_list)
				 
		assert len(x) == len(y), 'number of x and y should be same'
	
		return list(x), list(y), list(mask)