#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from DataSetGraph_SphericalMaps import MyData
import vtk
import SimpleITK as sitk
from os import path

# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def euler2quat(e, order=3):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack(
        (np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1
    )
    ry = np.stack(
        (np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1
    )
    rz = np.stack(
        (np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1
    )

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)

def ApplyTransformPC(pc, mask, rttn, trns, scl=None):
    
    outPC = torch.zeros(pc.shape).to(pc)
    for x in range(pc.shape[0]):
        mask_src=pc[x,mask[x]].to(pc)
        outPC[x,mask[x]]=torch.matmul((mask_src-torch.mean(mask_src,axis=0)),rttn[x].transpose(1,0))+trns[x]+torch.mean(mask_src,axis=0)
    return outPC

def npmat2euler(mats, seq='xyz'):
    eulers = []
    # for i in range(mats.shape[0]):
    for i in range(len(mats)):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')
def error_rotmat_angles(mat_pred, mat_gt, seq='xyz'):
    # mat_diff = np.matmul(mat_pred, mat_gt.T)
    return npmat2euler(np.matmul(mat_pred, mat_gt.transpose((0,2,1))))

def error_euler_angles(mat_pred,eulers_gt, seq='xyz'):
    mat_diff = []
    for i in range(mat_pred.shape[0]):
        r_pred =  mat_pred[i]
        r_gt = Rotation.from_euler(seq,eulers_gt[i],degrees=True).as_matrix() 
        mat_diff.append(r_pred.dot(r_gt.T))

    return npmat2euler(mat_diff)

def fit_in_m1_to_1(points):
    '''
    Input: Nx3 
    Output: Nx3 
    fits the point cloud in [(-1,-1,-1) to (1,1,1)]
    '''
    points = points - np.mean(points,axis=0)
    dist_from_orig = np.linalg.norm(points,axis=1)
    points = points/np.max(dist_from_orig)
    return points

def get_transformations(igt):
	R_ba = igt[:, 0:3, 0:3]								# Ps = R_ba * Pt
	translation_ba = igt[:, 0:3, 3].unsqueeze(2)		# Ps = Pt + t_ba
	R_ab = R_ba.permute(0, 2, 1)						# Pt = R_ab * Ps
	translation_ab = -torch.bmm(R_ab, translation_ba)	# Pt = Ps + t_ab
	return R_ab, translation_ab, R_ba, translation_ba

def batched_pairwise_dist(a, b):
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P

def distChamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    P = batched_pairwise_dist(a, b)
    return torch.mean(torch.min(P, 2)[0].float(), dim=-1), torch.mean(torch.min(P, 1)[0].float(), dim=-1), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def euler2rot(x):

    B = x.size(0)
    
    sinX = torch.sin(x[:,0])
    sinY = torch.sin(x[:,1])
    sinZ = torch.sin(x[:,2])

    cosX = torch.cos(x[:,0])
    cosY = torch.cos(x[:,1])
    cosZ = torch.cos(x[:,2])

    Rx = torch.zeros((B, 3, 3))
    Rx[:,0, 0] = 1.0
    Rx[:,1, 1] = cosX
    Rx[:,1, 2] = -sinX
    Rx[:,2, 1] = sinX
    Rx[:,2, 2] = cosX

    Ry = torch.zeros((B,3, 3))
    Ry[:,0, 0] = cosY
    Ry[:,0, 2] = sinY
    Ry[:,1, 1] = 1.0
    Ry[:,2, 0] = -sinY
    Ry[:,2, 2] = cosY

    Rz = torch.zeros((B,3, 3))
    Rz[:,0, 0] = cosZ
    Rz[:,0, 1] = -sinZ
    Rz[:,1, 0] = sinZ
    Rz[:,1, 1] = cosZ
    Rz[:,2, 2] = 1.0

    R = torch.matmul(torch.matmul(Rz, Ry), Rx)
    return R

def plotPCbatch(pcArray1, pcArray2, pcArray3, show = True, save = False, name=None, fig_count=4 , sizex = 5, sizey=10):
    
    pc1 = pcArray1[0:fig_count]
    pc2 = pcArray2[0:fig_count]
    pc3 = pcArray3[0:fig_count]
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count*3):

        ax = fig.add_subplot(3,fig_count,i+1, projection='3d')
        
        if(i<fig_count):
            ax.scatter(pc1[i,:,0], pc1[i,:,2], pc1[i,:,1], c='b', marker='.', alpha=0.8, s=8)
        elif i>=fig_count and i<fig_count*2:
            ax.scatter(pc2[i-fig_count,:,0], pc2[i-fig_count,:,2], pc2[i-fig_count,:,1], c='r', marker='.', alpha=0.8, s=8)
        else:
            ax.scatter(pc1[i-2*fig_count,:,0], pc1[i-2*fig_count,:,2], pc1[i-2*fig_count,:,1], c='b', marker='.', alpha=0.8, s=8)
            ax.scatter(pc3[i-2*fig_count,:,0], pc3[i-2*fig_count,:,2], pc3[i-2*fig_count,:,1], c='r', marker='.', alpha=0.8, s=8)


        ax.set_xlim3d(-0.6, 0.6)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)
            
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig
def ApplyTransform(data, transform):
    # Creating a copy of the input meshes
    a = vtk.vtkPolyData()
    a.DeepCopy(data)
    data = a

    for p in range(data.GetNumberOfPoints()):
        coords = np.array(data.GetPoint(p))
        newCoords = transform.TransformPoint(coords.astype(np.float64))
        data.GetPoints().SetPoint(p, newCoords[0], newCoords[1], newCoords[2])
    # Recalculating the normals and saving
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(data)
    filter.ComputeCellNormalsOff()
    filter.ComputePointNormalsOff()
    filter.NonManifoldTraversalOff()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    data = filter.GetOutput()
    return data
def ReadPolyData(filename):
    if filename.endswith('.vtk'):
        reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def WritePolyData(data, filename):
    if filename.endswith('.vtk'):
        writer = vtk.vtkPolyDataWriter()
    else:
        writer = vtk.vtkXMLPolyDataWriter()
    # Saving landmarks
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Update()
    return

def ApplyTransform2Mesh(meshName,R, t, s=[]):
    basedirs = 'D:\OneDrive - The University of Colorado Denver\FinalModelingData'
    mesh = ReadPolyData(path.join(basedirs,str(meshName),'ExternalHeadSurface-updated.vtp'))
    rigid_euler = sitk.AffineTransform(3)
    rigid_euler.SetTranslation(t[:,0].T.astype(float))
    rigid_euler.SetMatrix(R.ravel().astype(float))
    if s:
        rigid_euler.SetScale(s)
    mesh = ApplyTransform(mesh, rigid_euler)

    WritePolyData(mesh, path.join('meshResults',meshName)+'.vtp')
    
def AddTransformGraph(dataset, angle = 0, translation = 1):
    rad = np.pi / 180 * angle
    R = euler2rot(torch.Tensor(np.random.uniform(-rad,rad,3*len(dataset))*np.pi).reshape(len(dataset),3))
    trns = torch.Tensor(np.random.uniform(-translation,translation,3*len(dataset)).reshape(len(dataset),3))
    Data = []
    for i in range(len(dataset)):
        pos2 = torch.matmul((dataset[i].pos-torch.mean(dataset[i].pos,axis=0)), R[i].transpose(1,0))+trns[i]+torch.mean(dataset[i].pos,axis=0)
        data = MyData(pos = torch.tensor(dataset[i].pos), edge_index = torch.tensor(dataset[i].edge_index), num_nodes = len(dataset[i].pos), R=R[i], trns = trns[i], GT=torch.tensor(pos2), batch = torch.zeros(len(dataset[i].pos), dtype = torch.int64))
        # dataR = MyData(x = x, pos = torch.tensor(pos2), edge_index = torch.tensor(edge_indices, dtype = torch.long), node_weight = node_weights, num_nodes = len(pos), R=R, trns = trns, scl= scl)
        Data.append(data)
    return Data