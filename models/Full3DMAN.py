# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# import chamfer3D.dist_chamfer_3D
from ops.util import farthest_point_sample, index_points
import torch_geometric as tog
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import ChebConv, GraphNorm
from scipy.spatial.transform import Rotation
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def npmat2euler(mats, seq='xyz'):
    eulers = []
    # for i in range(mats.shape[0]):
    for i in range(len(mats)):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')
def error_rotmat_angles(mat_pred, mat_gt, seq='xyz'):
    # mat_diff = np.matmul(mat_pred, mat_gt.T)
    return npmat2euler(np.matmul(mat_pred, mat_gt))

def make_rotate_matrix(x):

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

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels , 1, bias=True)
        self.k_conv = nn.Conv1d(channels, channels , 1, bias=True)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=True)
        self.trans_conv = nn.Conv1d(channels, channels, 1, bias=True)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y = None):
        
        if y is not None:
            
            x_q = self.q_conv(y).permute(0, 2, 1) # b, n, c 
            x_k = self.k_conv(x)# b, c, n        
            x_v = self.v_conv(x)
            energy = torch.matmul(x_q,x_k) # b, n, n 
            attention = self.softmax(energy)
            attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
            x_r = x_v @ attention # b, c, n 
            # x_r = self.act(self.after_norm(self.trans_conv(y - x_r)))
            x_r = self.act(self.trans_conv(y - x_r))
            x = y + x_r
        else:
            x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
            x_k = self.k_conv(x)# b, c, n        
            x_v = self.v_conv(x)
            energy = x_q @ x_k # b, n, n 
            attention = self.softmax(energy)
            attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
            x_r = x_v @ attention # b, c, n 
            x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
            x = x + x_r
        return x
    

class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv11 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.conv12 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.bn11 = nn.BatchNorm1d(channels)
        self.bn12 = nn.BatchNorm1d(channels)

        self.conv21 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.conv22 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.bn21 = nn.BatchNorm1d(channels)
        self.bn22 = nn.BatchNorm1d(channels)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)
        self.sa5 = SA_Layer(channels)
        self.sa6 = SA_Layer(channels)
        

        self.relu = nn.LeakyReLU(0.5)
        
    def forward(self, x, y = None):     
        if y is not None:
           
            # y = self.relu(self.bn21(self.conv21(y))) # B, D, N
            # y = self.relu(self.bn22(self.conv22(y)))           
            x1 = self.sa1(x, y)
            x2 = self.sa2(y, x)
            x3 = self.sa3(x1, x)
            x4 = self.sa4(x2, y)
            
            x = torch.cat((x1, x2, x3, x4), dim=1)
            # x = torch.cat((x3, x4, x5, x6), dim=1)
            
        else:
            x = self.relu(self.bn11(self.conv11(x))) # B, D, N
            x = self.relu(self.bn12(self.conv12(x)))

            x1 = self.sa1(x)
            x2 = self.sa2(x1)
            x3 = self.sa3(x2)
            x4 = self.sa4(x3)
        
            x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class Decoder(nn.Module):
    def __init__(self, latent_size, ffn_hidden = 512, n_head = 4, drop_prob = 0.5):
        super(Decoder, self).__init__()
        
        self.latent_size = latent_size

        self.pt_last = StackedAttention(self.latent_size)
        
        self.dec1 = nn.Conv1d(self.latent_size*4, self.latent_size, kernel_size=1, bias=True)
        self.dec2 = nn.Linear(self.latent_size*2 ,self.latent_size // 2, bias=True)

        self.ActF = nn.LeakyReLU(0.5)

    def forward(self, src, trg): 
        x = self.pt_last(src, trg)
        x = self.ActF(self.dec1(x))
        x = self.dec2(torch.cat((torch.max(x, 2)[0], torch.min(x, 2)[0]), dim=1))
        return x


class N3DMAN(nn.Module):
    def __init__(self, point_size, latent_size, scale, use_normals):
        super(N3DMAN, self).__init__()

        self.latent_size = latent_size
        self.point_size = point_size
        self.scl = scale
        self.relu = nn.LeakyReLU(0.5)   

        if use_normals:
            self.conv1 = nn.Conv1d(6, self.latent_size, kernel_size=1, bias=True)
            self.conv1 = nn.Conv1d(6, self.latent_size, kernel_size=1, bias=True)
        else:
            self.conv1 = nn.Conv1d(3, self.latent_size, kernel_size=1, bias=True)
            self.conv2 = nn.Conv1d(3, self.latent_size, kernel_size=1, bias=True)

        self.decoder = Decoder(self.latent_size, ffn_hidden=512)


        self.proj_rot = nn.Linear(self.latent_size // 2, 3, bias=True)
        self.proj_trans = nn.Linear(self.latent_size // 2, 3, bias=True)
        if self.scl:
            self.proj_sca = nn.Linear(self.latent_size // 2, 1, bias=True)


    def forward(self, source, target):

        src = self.relu(self.conv1(source))
        trg = self.relu(self.conv2(target))

        x = self.decoder(src, trg)
        

        rttn = F.sigmoid(self.proj_rot(x))
        # rttn = rttn / torch.norm(rttn, p = 2, dim = 1, keepdim=True)
        rttn = (rttn-0.5)*np.pi

        trns = self.proj_trans(x)
        if self.scl:
            scl = F.sigmoid(self.proj_sca(x))+0.5
        else:
            scl=[]

        return rttn, trns, scl
    

class Encoder(nn.Module):
    def __init__(self, latent_size,device, drop_prob=0.1):
        super(Encoder, self).__init__()
        
        self.latent_size = latent_size
        self.device = device
        in_channels = 9
        out_channels = self.latent_size
        hidden_features = int(self.latent_size/3)
        Ks = [4,8]
        use_weights = False
        use_bias = False
        self.out_channels = out_channels
        self.use_weights = use_weights
        self.output_heat_map = False


        self.cheby1 = ChebConv(in_channels, hidden_features, K = Ks[0], bias=use_bias)#.to(self.device)
        self.batchnorm1 = GraphNorm(hidden_features)#.to(self.device)
        self.cheby2 = ChebConv(hidden_features, hidden_features, K = Ks[0], bias=use_bias)#.to(self.device)
        self.batchnorm2 = GraphNorm(hidden_features)#.to(self.device)
        self.cheby3 = ChebConv(hidden_features, hidden_features, K = Ks[0], bias=use_bias)#.to(self.device)
        self.batchnorm3 = GraphNorm(hidden_features)#.to(self.device)
        self.cheby4 = ChebConv(in_channels, hidden_features, K = Ks[1], bias=use_bias)#.to(self.device)
        self.batchnorm4 = GraphNorm(hidden_features)#.to(self.device)
        self.cheby5 = ChebConv(hidden_features, hidden_features, K = Ks[1], bias=use_bias)#.to(self.device)
        self.batchnorm5 = GraphNorm(hidden_features)#.to(self.device)
        self.cheby6 = ChebConv(hidden_features, hidden_features, K = Ks[1], bias=use_bias)#.to(self.device)
        self.batchnorm6 = GraphNorm(hidden_features)#.to(self.device)
        
        #feature selector MLP
        if self.use_weights:
            #feature selector for Low, medium, and high res features
            self.feature_selector = ChebConv(in_channels, 1, K = 1)

        #fc layers -- concatenated with lower level embeddings
        self.fc0 = nn.Linear(hidden_features*3, out_channels, bias=use_bias)
        self.final_fc = nn.Linear(hidden_features*6, self.out_channels)#.to(self.devices[0])
        
        
        self.conv1 = nn.Conv1d(6, self.latent_size, kernel_size=1, bias=use_bias)
        self.conv1 = nn.Conv1d(self.latent_size, self.latent_size, kernel_size=1, bias=use_bias)
        self.relu = nn.LeakyReLU(0.5)
        self.sig = nn.Sigmoid()
        self.dropout1 = nn.Dropout(drop_prob)
        self.bn1 = nn.BatchNorm1d(self.latent_size)
        

    def forward(self, data):         
        #grab our data from the loader
        pos, features, batch, edge_indices = data.pos, data.x, data.batch, data.edge_index
        x0 = torch.cat([pos,features], dim = 1).to(torch.float32)
        #one side
        out1 = (self.relu(self.cheby1(x0, edge_indices, batch = batch)))
        out2 = (self.relu(self.cheby2(out1, edge_indices, batch = batch)))
        out3 = (self.relu(self.cheby3(out2, edge_indices, batch = batch)))

        # other side
        out4 = (self.relu(self.cheby4(x0, edge_indices, batch = batch)))
        out5 = (self.relu(self.cheby5(out4, edge_indices, batch = batch)))
        out6 = (self.relu(self.cheby6(out5, edge_indices, batch = batch)))
        
        if not self.use_weights:
            hr = out1 + out4
            mr = out2 + out5
            lr = out3 + out6
        if self.use_weights:
            self.node_weights = self.feature_selector(x0, edge_indices, batch = batch).sigmoid()
            lw = self.node_weights[:,0,None]
            mw = self.node_weights[:,0,None]
            hw = self.node_weights[:,0,None]
            hr = out1 * hw + out4 * (1-hw)
            mr = out2 * mw + out5 * (1-mw)
            lr = out3 * lw + out6 * (1-lw)

        features = torch.cat([hr, mr, lr], dim = 1)
        
        
        fea2dense =to_dense_batch(features, batch)[0]
        pos2 =to_dense_batch(x0, batch)[0]
        
        out = (fea2dense.transpose(2,1)@pos2.to(torch.float32))
        out = out/torch.norm(out,dim=2).unsqueeze(2)
        return out

class Full3DMAN(nn.Module):
    def __init__(self, latent_size, scale=False, use_normals=False, device='cpu'):
        super(Full3DMAN, self).__init__()

        self.latent_size = latent_size
        self.scl = scale
        self.device = device
        drop_prob= 0.1
        
        self.encoder1 = Encoder(self.latent_size, self.device, drop_prob=drop_prob)
        self.encoder2 = Encoder(self.latent_size, self.device, drop_prob=drop_prob)
        
        self.ActF = nn.LeakyReLU(0.5)
        
        self.decoder = Decoder( self.latent_size, ffn_hidden=512, drop_prob=drop_prob)


        self.proj_rot = nn.Linear(self.latent_size // 2, 3)
        self.proj_trans = nn.Linear(self.latent_size // 2, 3)
        self.proj_sca = nn.Linear(self.latent_size // 2, 1)


    def forward(self, source, target):
        src = self.encoder1(source).to(self.device)
        trg = self.encoder2(target).to(self.device)
        
        x = self.decoder(src, trg)
        

        rttn = F.sigmoid(self.proj_rot(x))
        # rttn = rttn / torch.norm(rttn, p = 2, dim = 1, keepdim=True)
        rttn = (rttn-0.5)*np.pi

        trns = self.proj_trans(x)
        if self.scl:
            scl = F.sigmoid(self.proj_sca(x))+0.5
        else:
            scl=[]

        return rttn, trns, scl