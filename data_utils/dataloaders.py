import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import h5py
import subprocess
import shlex
import json
import glob
# from .. ops import transform_functions, se3
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from ops.transform_functions import PCTransform, convert2transformation
import random
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-10:] == 'data_utils':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))

def download_modelnet40():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))

def load_data(train, use_normals):
	if train: partition = 'train'
	else: partition = 'test'
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	all_data = []
	all_label = []
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
		f = h5py.File(h5_name)
		if use_normals: data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
		else: data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label

def deg_to_rad(deg):
	return np.pi / 180 * deg

def create_random_transform(dtype, max_rotation_deg, max_translation):
	max_rotation = deg_to_rad(max_rotation_deg)
	rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
	trans = np.random.uniform(-max_translation, max_translation, [1, 3])
	quat = transform_functions.euler_to_quaternion(rot, "xyz")

	vec = np.concatenate([quat, trans], axis=1)
	vec = torch.tensor(vec, dtype=dtype)
	return vec

def jitter_pointcloud(pointcloud, sigma=0.04, clip=0.05):
	# N, C = pointcloud.shape
	sigma = 0.04*np.random.random_sample()
	pointcloud += torch.empty(pointcloud.shape).normal_(mean=0, std=sigma).clamp(-clip, clip)
	return pointcloud

def farthest_subsample_points(pointcloud1, num_subsampled_points=768):
	pointcloud1 = pointcloud1
	num_points = pointcloud1.shape[0]
	nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
							 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
	random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
	idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
	gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
	return pointcloud1[idx1, :], gt_mask

def uniform_2_sphere(num: int = None):
	"""Uniform sampling on a 2-sphere

	Source: https://gist.github.com/andrewbolster/10274979

	Args:
		num: Number of vectors to sample (or None if single)

	Returns:
		Random Vector (np.ndarray) of size (num, 3) with norm 1.
		If num is None returned value will have size (3,)

	"""
	if num is not None:
		phi = np.random.uniform(0.0, 2 * np.pi, num)
		cos_theta = np.random.uniform(-1.0, 1.0, num)
	else:
		phi = np.random.uniform(0.0, 2 * np.pi)
		cos_theta = np.random.uniform(-1.0, 1.0)

	theta = np.arccos(cos_theta)
	x = np.sin(theta) * np.cos(phi)
	y = np.sin(theta) * np.sin(phi)
	z = np.cos(theta)

	return np.stack((x, y, z), axis=-1)

def planar_crop(points, p_keep= 0.7):
	p_keep = np.array(p_keep, dtype=np.float32)

	rand_xyz = uniform_2_sphere()
	pts = points.numpy()
	centroid = np.mean(pts[:, :3], axis=0)
	points_centered = pts[:, :3] - centroid

	dist_from_plane = np.dot(points_centered, rand_xyz)

	mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
	idx_x = torch.Tensor(np.nonzero(mask))

	return torch.Tensor(pts[mask, :3]), idx_x

def knn_idx(pts, k):
	kdt = cKDTree(pts) 
	_, idx = kdt.query(pts, k=k+1)
	return idx[:, 1:]

def get_rri(pts, k):
	# pts: N x 3, original points
	# q: N x K x 3, nearest neighbors
	q = pts[knn_idx(pts, k)]
	p = np.repeat(pts[:, None], k, axis=1)
	# rp, rq: N x K x 1, norms
	rp = np.linalg.norm(p, axis=-1, keepdims=True)
	rq = np.linalg.norm(q, axis=-1, keepdims=True)
	pn = p / rp
	qn = q / rq
	dot = np.sum(pn * qn, -1, keepdims=True)
	# theta: N x K x 1, angles
	theta = np.arccos(np.clip(dot, -1, 1))
	T_q = q - dot * p
	sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
	cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
	psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)
	idx = np.argpartition(psi, 1)[:, :, 1:2]
	# phi: N x K x 1, projection angles
	phi = np.take_along_axis(psi, idx, axis=-1)
	feat = np.concatenate([rp, rq, theta, phi], axis=-1)
	return feat.reshape(-1, k * 4)

def get_rri_cuda(pts, k, npts_per_block=1):
	try:
		import pycuda.autoinit
		from pycuda import gpuarray
		from pycuda.compiler import SourceModule
	except Exception as e:
		print("Error raised in pycuda modules! pycuda only works with GPU, ", e)
		raise

	mod_rri = SourceModule(open('rri.cu').read() % (k, npts_per_block))
	rri_cuda = mod_rri.get_function('get_rri_feature')

	N = len(pts)
	pts_gpu = gpuarray.to_gpu(pts.astype(np.float32).ravel())
	k_idx = knn_idx(pts, k)
	k_idx_gpu = gpuarray.to_gpu(k_idx.astype(np.int32).ravel())
	feat_gpu = gpuarray.GPUArray((N * k * 4,), np.float32)

	rri_cuda(pts_gpu, np.int32(N), k_idx_gpu, feat_gpu,
				grid=(((N-1) // npts_per_block)+1, 1),
				block=(npts_per_block, k, 1))

	feat = feat_gpu.get().reshape(N, k * 4).astype(np.float32)
	return feat


class UnknownDataTypeError(Exception):
	def __init__(self, *args):
		if args: self.message = args[0]
		else: self.message = 'Datatype not understood for dataset.'

	def __str__(self):
		return self.message


class ModelNet40Data(Dataset):
	def __init__(
		self,
		train=True,
		num_points=1024,
		download=True,
		randomize_data=False,
		use_normals=False
	):
		super(ModelNet40Data, self).__init__()
		if download: download_modelnet40()
		self.data, self.labels = load_data(train, use_normals)
		if not train: self.shapes = self.read_classes_ModelNet40()
		self.num_points = num_points
		self.randomize_data = randomize_data

	def __getitem__(self, idx):
		if self.randomize_data: current_points = self.randomize(idx)
		else: current_points = self.data[idx].copy()

		current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
		label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.data.shape[0]

	def randomize(self, idx):
		pt_idxs = np.arange(0, self.num_points)
		np.random.shuffle(pt_idxs)
		return self.data[idx, pt_idxs].copy()

	def get_shape(self, label):
		return self.shapes[label]

	def read_classes_ModelNet40(self):
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
		file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
		shape_names = file.read()
		shape_names = np.array(shape_names.split('\n')[:-1])
		return shape_names
class LoadedDataset(Dataset):
    def __init__(self, filenames):
        super().__init__()
        self.filenames =filenames

    def __getitem__(self, idx):
        return self.load_file(self.filenames[idx])

    def load_file(self, filename):
        if type(filename) == str:
            data = torch.load(filename)
        return data

    def __get__(self):
        pass
    def __len__(self):
        return len(self.filenames)
    def __remove__(self, idx):
        del self.filenames[idx]

class ReadDataset(Dataset):
    def __init__(self,  source):
     
        self.data = torch.from_numpy(source).float()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def RandomSplit(datasets, train_set_percentage, val_set_percentage):
    # Mezcla aleatoria de los datos
    # np.random.shuffle(datasets)

    # Calcula las longitudes de los conjuntos de datos
    total_length = len(datasets)
    train_length = int(total_length * train_set_percentage)
    val_length = int(total_length * val_set_percentage)

    # Divide los datos en conjuntos de entrenamiento, validación y prueba
    train_data = datasets[:train_length]
    val_data = datasets[train_length:train_length + val_length]
    test_data = datasets[train_length + val_length:]

    return train_data, val_data, test_data
def CranioCS(nameFile, batch_size, train_set_percentage=0.8, val_set_percentage=0.1, shuffle=False, num_workers=0, pin_memory=True):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR[:-10], 'data')
    npArray = torch.load(os.path.join(DATA_DIR, nameFile))
    # rttnM = ReadDataset(npArray['rttnM'])
    # trnsV = ReadDataset(npArray['trnsV'])
#    sclrF = ReadDataset(npArray['sclrF'])
    DBReg = ReadDataset(npArray['DBReg'])
    DBTran = ReadDataset(npArray['DBTra'])
    igt=convert2transformation(torch.from_numpy(npArray['rttnM']).float(),torch.from_numpy(npArray['trnsV']).float().unsqueeze(2))
    # pcR_copy = DBReg.copy()
    # pcR_copy = random.sample(pcR_copy, len(pcR_copy))
    combined_dataset = list(zip(DBReg, DBTran, igt))
    del DBReg, DBTran, npArray
    train_data, val_data, test_data = RandomSplit(combined_dataset, train_set_percentage, val_set_percentage)

    train_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def DataLoaders(dir, fileName, batch_size, testbatch=1, train_set_percentage=0.8, val_set_percentage=0.1, shuffle=False, num_workers=0, pin_memory=True):
	files = [str(x) for x in Path(dir).glob(fileName)]
	dataset = LoadedDataset(files)

	rttnM = []
	trnsV = []
	DBReg = []
	DBTran = []
	meshName = []
	for i in range(len(dataset)):
		if i==0:
			rttnM = np.expand_dims(np.array(dataset[i]['rttnM']).reshape(3,3), axis=0)
			trnsV = np.expand_dims(dataset[i]['trnsV'], axis=0)
			DBReg = dataset[i]['DBReg'].unsqueeze(0)
			DBTran = dataset[i]['DBTra'].unsqueeze(0)
			meshName.append(files[i][41:-12])
		else:
			rttnM = np.concatenate([rttnM, np.expand_dims(np.array(dataset[i]['rttnM']).reshape(3,3), axis=0)], axis=0)
			trnsV = np.concatenate([trnsV, np.expand_dims(dataset[i]['trnsV'], axis=0)], axis=0)
			DBReg = torch.cat([DBReg,dataset[i]['DBReg'].unsqueeze(0)], dim=0)
			DBTran = torch.cat([DBTran, dataset[i]['DBTra'].unsqueeze(0)], dim=0)
			meshName.append(files[i][41:-12])
	
	igt=convert2transformation(torch.from_numpy(rttnM).float(),torch.from_numpy(trnsV).float().unsqueeze(2))
    # # pcR_copy = DBReg.copy()
    # # pcR_copy = random.sample(pcR_copy, len(pcR_copy))
	combined_dataset = list(zip(DBReg, DBTran, igt, meshName))
    # del DBReg, DBTran, npArray
    # train_data, val_data, test_data = RandomSplit(combined_dataset, train_set_percentage, val_set_percentage)

	train_data, val_data, test_data = torch.utils.data.random_split(combined_dataset,[train_set_percentage,val_set_percentage,val_set_percentage], torch.Generator().manual_seed(42))

	train_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
	val_loader = DataLoader(val_data, shuffle=shuffle, num_workers=num_workers, batch_size=testbatch, pin_memory=pin_memory)
	test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=testbatch, pin_memory=pin_memory)

	return train_loader, val_loader, test_loader


class RegistrationData(Dataset):
    def __init__(self, data_class=ModelNet40Data(), partial_source=False, partial_template=False, noise=False, normals=False, additional_params={}):
        super(RegistrationData, self).__init__()
# 		available_algorithms = ['PCRNet', 'PointNetLK', 'DCP', 'PRNet', 'iPCRNet', 'RPMNet', 'DeepGMR']
# 		if algorithm in available_algorithms: self.algorithm = algorithm
# 		else: raise Exception("Algorithm not available for registration.")
		
        self.set_class(data_class)
        self.partial_template = partial_template
        self.partial_source = partial_source
        self.noise = noise
        self.additional_params = additional_params
        self.use_rri = False
        self.normals = normals

        # self.transforms = PCTransform(angle_range=45,translation_range=0.5,normals)
        self.transforms = PCTransform(45,0.5,normals)

    def __len__(self):
        return len(self.data_class)

    def set_class(self, data_class):
        self.data_class = data_class

    def __getitem__(self, index):
        template, label = self.data_class[index]
        self.transforms.index = index				# for fixed transformations in PCRNet.
        source = self.transforms(template)

		# Check for Partial Data.
        if self.additional_params.get('partial_point_cloud_method', None) == 'planar_crop':
            source, gt_idx_source = planar_crop(source)
            template, gt_idx_template = planar_crop(template)
            intersect_mask, intersect_x, intersect_y  = np.intersect1d(gt_idx_source, gt_idx_template, return_indices=True)
            
            self.template_mask = torch.zeros(template.shape[0])
            self.source_mask = torch.zeros(source.shape[0])
            self.template_mask[intersect_y]  = 1
            self.source_mask[intersect_x]  = 1
        else:
            if self.partial_source: source, self.source_mask = farthest_subsample_points(source)
            if self.partial_template: template, self.template_mask = farthest_subsample_points(template)



		# Check for Noise in Source Data.
        if self.noise: source = jitter_pointcloud(source)

        if self.use_rri:
            template, source = template.numpy(), source.numpy()
            template = np.concatenate([template, self.get_rri(template - template.mean(axis=0), self.nearest_neighbors)], axis=1)
            source = np.concatenate([source, self.get_rri(source - source.mean(axis=0), self.nearest_neighbors)], axis=1)
            template, source = torch.tensor(template).float(), torch.tensor(source).float()

        igt = self.transforms.igt
		
        if self.additional_params.get('use_masknet', False):
            if self.partial_source and self.partial_template:
                return template, source, igt, self.template_mask, self.source_mask
            elif self.partial_source:
                return template, source, igt, self.source_mask
            elif self.partial_template:
                return template, source, igt, self.template_mask
        else:
            return template, source, igt


if __name__ == '__main__':
	class Data():
		def __init__(self):
			super(Data, self).__init__()
			self.data, self.label = self.read_data()

		def read_data(self):
			return [4,5,6], [4,5,6]

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			return self.data[idx], self.label[idx]

	cd = RegistrationData(ModelNet40Data(train=False), normals=True)
# 	import ipdb; ipdb.set_trace()
