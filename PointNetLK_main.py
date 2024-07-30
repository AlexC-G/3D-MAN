# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:10:17 2024

@author: cruzguea
"""

import open3d as o3d
from scipy.spatial.transform import Rotation
import os
import gc
import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from models.pointnet import PointNet
from models.pointnetlk import PointNetLK
from ops.util import transform_point_cloud, error_rotmat_angles
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
# from ops.display import display_three_clouds 
from data_utils import RegistrationData, ModelNet40Data, CranioCS, DataLoaders
import ops.util as utl
import datetime
from ops.transform_functions import convert2transformation
from ops.chamfer_distance import ChamferDistanceLoss
# from ops.frobenius_norm import FrobeniusNormLoss


epoch_COUNT = 0

class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()


def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

def test_one_epoch(args, net, test_loader):
    net.eval()
    with torch.no_grad():
    	# initialization
        mse_ab = 0
        mae_ab = 0
        mse_ba = 0
        mae_ba = 0
        mse = []
        mae = []
        chamfer=[]
    
        total_loss = 0
        total_cycle_loss = 0
        num_examples = 0
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
    
    
        rotations_ba = []
        translations_ba = []
        rotations_ba_pred = []
        translations_ba_pred = []
    
        eulers_ab = []
        eulers_ba = []
    
        batch_idx=0
        # total_correct_pred = 0
        itr = 0
        
        for data in tqdm(test_loader):
            if args.dataset == 'cranio':
                target, src, igt, _ = data
                transformations = utl.get_transformations(igt)
                transformations = [t.to(args.device) for t in transformations]
                rotation_ab, translation_ab, rotation_ba, translation_ba = transformations
            else:
                target, src, igt = data
                transformations = utl.get_transformations(igt)
                transformations = [t.to(args.device) for t in transformations]
                rotation_ab, translation_ab, rotation_ba, translation_ba = transformations
    
            target = target.to(args.device)
            src = src.to(args.device)
            igt = igt.to(args.device)
            batch_size = src.size(0)

            rotation_ab = rotation_ab.to(args.device)
            translation_ab = translation_ab.to(args.device)
            rotation_ba = rotation_ba.to(args.device)
            translation_ba = translation_ba.to(args.device)
            num_examples += batch_size
    
    		# model output
            output= net(target,src) 
            rotation_ab_pred = output['est_R']
            translation_ab_pred = output['est_t']
            # rotation_ab_pred = utl.euler2rot(rotation_ab_pred_euler).to(args.device)
            rotation_ba_pred = rotation_ab_pred.transpose(2, 1).contiguous().to(args.device)
            translation_ba_pred = -torch.matmul(rotation_ba_pred, translation_ab_pred.unsqueeze(2)).squeeze(2).to(args.device)
            
            
    		## save rotation and translation
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.unsqueeze(2).detach().cpu().numpy())
            eulers_ab.append(utl.npmat2euler(rotation_ab.cpu()))
    		
            rotations_ba.append(rotation_ba.detach().cpu().numpy())
            translations_ba.append(translation_ba.detach().cpu().numpy())
            rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
            translations_ba_pred.append(translation_ba_pred.unsqueeze(2).detach().cpu().numpy())
            eulers_ba.append(utl.npmat2euler(rotation_ba.cpu()))
    
    		# transforming the point cloud according to given rotation and translation
            transformed_src = transform_point_cloud(src.transpose(2,1), rotation_ab_pred, translation_ab_pred).transpose(2,1)

            transformed_target = transform_point_cloud(target.transpose(2,1), rotation_ba_pred, translation_ba_pred).transpose(2,1)
    
            identity = torch.eye(3).to(args.device).unsqueeze(0).repeat(batch_size, 1, 1)
    		
     
     		#  correspondence loss, as proposed in our paper
            if args.loss == 'mse_transf':
                loss = torch.mean(torch.norm(torch.matmul(rotation_ab_pred.transpose(2, 1),rotation_ab)-identity, p='fro', dim=[1,2])) + torch.mean(torch.norm(output['r'], dim=1)) #+ torch.mean(torch.norm(sclr - sclrO, dim=1))
                # dist1, dist2, _, _ = utl.distChamfer(transformed_src.transpose(2, 1), target.transpose(2, 1))
                loss_chamfer = ChamferDistanceLoss()(target, transformed_src)
                # translation loss, as proposed in DCP
            elif args.loss == 'mse':
                loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
             			   + F.mse_loss(translation_ab_pred.unsqueeze(2), translation_ab))
            elif args.loss == 'chamfer_dist':
                # dist1, dist2, _, _ = utl.distChamfer(transformed_src, target)
                # loss = torch.mean(torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1))
                loss_chamfer = ChamferDistanceLoss()(target, transformed_src)
                loss = loss_chamfer
            elif args.loss == 'mse_doubT':
                T_gt = output['est_T']
                eye = torch.eye(4).expand_as(T_gt).to(args.device)
                T_pred = convert2transformation(rotation_ab,translation_ab)
                loss = torch.mean(torch.norm(T_pred @ torch.inverse(T_gt)- eye, p='fro', dim=[1,2]))
                # mse2 = F.mse_loss(output['est_T'] @ T_gt, eye)
                # loss = mse1 + mse2
                
                loss_chamfer = ChamferDistanceLoss()(target, transformed_src)
                # dist1, dist2, _, _ = utl.distChamfer(transformed_src.transpose(2, 1), target.transpose(2, 1))
                # loss_chamfer = torch.mean(torch.mean(dist1, dim=-1) + torch.mean(dist2, dim=-1))
            else:
                raise Exception ("please verify the input loss function")
    
            total_loss += loss.item() * batch_size
            total_cycle_loss += loss_chamfer.item() * batch_size
            chamfer.append(loss_chamfer.item())
            # if itr%10==0:
            #     utl.plotPCbatch(target.detach().cpu().numpy(), src.detach().cpu().numpy(), transformed_src.detach().cpu().numpy() , show=True, save=True, name = ("output/PNLK_"+str(itr)), fig_count=1)
    
            try:
                mse_ab +=  torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
                mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size
                mse.append(torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item())
                mae.append(torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item())
    
                mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
                mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size
            except: # in partial point cloud case
                mse_ab += 0
                mae_ab += 0
                mse_ba += 0
                mae_ba += 0
    
            itr+=1  
            batch_idx +=1
        
        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        
        rotations_ba = np.concatenate(rotations_ba, axis=0)
        translations_ba = np.concatenate(translations_ba, axis=0)
        rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
        translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)
        
        eulers_ab = np.concatenate(eulers_ab, axis=0)
        eulers_ba = np.concatenate(eulers_ba, axis=0)
    
        return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
    		   mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
    		   mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
    		   translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
    		   translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba, mse, mae, chamfer#,\
    # 		   incorrect_correspondences


def train_one_epoch(args, net, train_loader, opt):
    net.train()
    global epoch_COUNT
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0
    
    total_loss = 0
    total_loss_dcp_rot = 0
    total_loss_dcp_t = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    
    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []
    
    eulers_ab = []
    eulers_ba = []

# 	total_correct_pred =0
    itr = 0
    for i, data in enumerate(tqdm(train_loader)):
        if args.dataset == 'cranio':
            target, src, igt, _ = data
            transformations = utl.get_transformations(igt)
            transformations = [t.to(args.device) for t in transformations]
            rotation_ab, translation_ab, rotation_ba, translation_ba = transformations
        else:
            target, src, igt = data
            transformations = utl.get_transformations(igt)
            transformations = [t.to(args.device) for t in transformations]
            rotation_ab, translation_ab, rotation_ba, translation_ba = transformations
        
        target = target.to(args.device)
        src = src.to(args.device)
        igt = igt.to(args.device)
        
        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        
        # model output
        output= net(target, src) 
        rotation_ab_pred = output['est_R']
        translation_ab_pred = output['est_t']
        # rotation_ab_pred = utl.euler2rot(rotation_ab_pred_euler).to(args.device)
        rotation_ba_pred = rotation_ab_pred.transpose(2, 1).contiguous().to(args.device)
        translation_ba_pred = -torch.matmul(rotation_ba_pred, translation_ab_pred.unsqueeze(2)).squeeze(2).to(args.device)
        
        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.unsqueeze(2).detach().cpu().numpy())
        eulers_ab.append(utl.npmat2euler(rotation_ab.cpu()))
        		
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.unsqueeze(2).detach().cpu().numpy())
        eulers_ba.append(utl.npmat2euler(rotation_ba.cpu()))
        
        # transforming the point cloud according to given rotation and translation
        transformed_src = transform_point_cloud(src.transpose(2,1), rotation_ab_pred, translation_ab_pred).transpose(2,1)

        transformed_target = transform_point_cloud(target.transpose(2,1), rotation_ba_pred, translation_ba_pred).transpose(2,1)
        
        identity = torch.eye(3).to(args.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        
        loss_dcp_rot = F.mse_loss(torch.matmul(rotation_ab_pred, rotation_ab), identity) 
        loss_dcp_t = F.mse_loss(translation_ab_pred.unsqueeze(2), translation_ab) 
        				
        
        #  correspondence loss, as proposed in our paper
        if args.loss == 'mse_transf':
            loss = torch.mean(torch.norm(torch.matmul(rotation_ab_pred.transpose(2, 1),rotation_ab)-identity, p='fro', dim=[1,2])) + torch.mean(torch.norm(output['r'], dim=1)) #+ torch.mean(torch.norm(sclr - sclrO, dim=1))
            # dist1, dist2, _, _ = utl.distChamfer(transformed_src.transpose(2, 1), target.transpose(2, 1))
            loss_chamfer = ChamferDistanceLoss()(target, transformed_src)
            # translation loss, as proposed in DCP
        elif args.loss == 'mse':
            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
			   + F.mse_loss(translation_ab_pred.unsqueeze(2), translation_ab))
        elif args.loss == 'chamfer_dist':

            loss_chamfer = ChamferDistanceLoss()(target, transformed_src)
            loss = loss_chamfer
        elif args.loss == 'mse_doubT':
            T_gt = output['est_T']
            eye = torch.eye(4).expand_as(T_gt).to(args.device)
            T_pred = convert2transformation(rotation_ab,translation_ab)
            loss = torch.mean(torch.norm(torch.matmul(T_pred, torch.inverse(T_gt))- eye, p='fro', dim=[1,2]))
            loss_chamfer = ChamferDistanceLoss()(target, transformed_src)
        else:
            raise Exception ("please verify the input loss function")
		 
        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size
        total_loss_dcp_rot += loss_dcp_rot.item() * batch_size
        total_loss_dcp_t += loss_dcp_t.item() * batch_size
        total_cycle_loss += loss_chamfer.item() * batch_size

        try:
            mse_ab +=  torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
            mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size
            mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
            mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size
        except: # in partial point cloud case
            mse_ab += 0
            mae_ab += 0
            mse_ba += 0
            mae_ba += 0

        itr+=1  
       
    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
    
    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)
    
    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)
    
    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
		   mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
		   mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
		   translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
		   translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba#,\
# 		   incorrect_correspondences


def test(args, net, test_loader, boardio, textio):

    test_loss, test_cycle_loss, \
    test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred, test_eulers_ab, test_eulers_ba, mse, mae, chamfer = test_one_epoch(args, net, test_loader)
    test_rmse_ab = np.sqrt(test_mse_ab)
# 	test_rmse_ba = np.sqrt(test_mse_ba)
    
    test_error_euler_angles_ab = error_rotmat_angles(test_rotations_ab_pred,test_rotations_ab)
# 	test_error_euler_angles_ab = error_euler_angles(test_rotations_ab_pred,np.degrees(test_eulers_ab))  # computing euler angle error
    test_r_mse_ab = np.mean(test_error_euler_angles_ab ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_error_euler_angles_ab))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))
    error_trans =(test_translations_ab - test_translations_ab_pred)
    np.savez_compressed('PointNetLK_main_cranio', mse=np.asarray(mse), mae=np.asarray(mae), chamfer=np.asarray(chamfer), ErrorRot=test_error_euler_angles_ab, ErrorTran=error_trans)
    
    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, Chamfer Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
	  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f' 
	  % (-1, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab, test_r_mse_ab,
		 test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))


def train(args, net, train_loader, test_loader, boardio, textio):
    if args.optimizator == 'sgd':
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizator == 'adam':
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)
    else:
        raise Exception('Not implemented')
    scheduler = MultiStepLR(opt, milestones=[10, 25, 50 ,150], gamma=0.25)
    
    print(len(train_loader),"len(train_loader)")
    best_test_loss = np.inf
    best_test_cycle_loss = np.inf
    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf
    
    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf


    for epoch in range(args.epochs):
		
        train_loss, train_cycle_loss, \
        train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, \
        train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
        train_translations_ba_pred, train_eulers_ab, train_eulers_ba = train_one_epoch(args, net, train_loader, opt)
        
        test_loss, test_cycle_loss, \
        test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
        test_translations_ba_pred, test_eulers_ab, test_eulers_ba, mse, mae, chamfer = test_one_epoch(args, net, test_loader)
        train_rmse_ab = np.sqrt(train_mse_ab)
        test_rmse_ab = np.sqrt(test_mse_ab)
        
        train_error_euler_angles_ab = error_rotmat_angles(train_rotations_ab_pred, train_rotations_ab)
        # train_error_euler_angles_ab = train_rotations_ab_pred-train_rotations_ab#error_euler_angles(train_rotations_ab_pred.transpose(2, 1),np.degrees(train_eulers_ab))  # computing euler angle error
        train_r_mse_ab = np.mean(train_error_euler_angles_ab ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_error_euler_angles_ab))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        test_error_euler_angles_ab = error_rotmat_angles(test_rotations_ab_pred,test_rotations_ab)
        # test_error_euler_angles_ab = test_rotations_ab_pred-test_rotations_ab#error_euler_angles(test_rotations_ab_pred.transpose(2, 1),np.degrees(test_eulers_ab))  # computing euler angle error
        test_r_mse_ab = np.mean(test_error_euler_angles_ab ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(test_error_euler_angles_ab))
        test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_cycle_loss = test_cycle_loss
            
            best_test_mse_ab = test_mse_ab
            best_test_rmse_ab = test_rmse_ab
            best_test_mae_ab = test_mae_ab
            
            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab
            
            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            if torch.cuda.device_count() > 1:
            #     torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            # else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Chamfer Loss:, %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
					  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
					  % (epoch, train_loss, train_cycle_loss, train_mse_ab, train_rmse_ab, train_mae_ab, train_r_mse_ab,
						 train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Chamfer Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
					  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f' 
					  % (epoch, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab, test_r_mse_ab,
						 test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Chamfer Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
					  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
					  % (epoch, best_test_loss, best_test_cycle_loss, best_test_mse_ab, best_test_rmse_ab,
						 best_test_mae_ab, best_test_r_mse_ab, best_test_r_rmse_ab,
						 best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab))

		# Train
        boardio.add_scalar('Train Loss', train_loss, epoch)
        boardio.add_scalar('A->B/train/MSE', train_mse_ab, epoch)
        boardio.add_scalar('A->B/train/RMSE', train_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/MAE', train_mae_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/MSE', train_r_mse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/RMSE', train_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/MAE', train_r_mae_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MSE', train_t_mse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/RMSE', train_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MAE', train_t_mae_ab, epoch)
# 		boardio.add_scalar('A->B/train/incorrect_correspondences', train_incorrect_correspondeces, epoch)


		# Test
        boardio.add_scalar('Test Loss', test_loss, epoch)
        boardio.add_scalar('A->B/test/MSE', test_mse_ab, epoch)
        boardio.add_scalar('A->B/test/RMSE', test_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/MAE', test_mae_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/MSE', test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/MAE', test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MSE', test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/RMSE', test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)
# 		boardio.add_scalar('A->B/test/incorrect_correspondences', test_incorrect_correspondeces, epoch)

		# Best Test
        boardio.add_scalar('Best Test Loss', best_test_loss, epoch)
        boardio.add_scalar('A->B/best_test/MSE', best_test_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/RMSE', best_test_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/MAE', best_test_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MSE', best_test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/RMSE', best_test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MSE', best_test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/RMSE', best_test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)
        # scheduler.step()
        if torch.cuda.device_count() > 1:
        #     torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%s.t7' % (args.exp_name, "last"))
        # else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%s.t7' % (args.exp_name, "last"))
        gc.collect()




# functions for testing on a single point cloud: test_bunny
def extract_vertices(mesh,num_vert=512):
	pcd_downsampled = mesh.sample_points_uniformly( number_of_points=num_vert)  
	points = np.asanyarray(pcd_downsampled.points)  
	return points


def rotate_cloud (points, add_noise = False):
	'''
	input:
		points = Nx3
	output:
		points = Nx3
	'''
	axis = np.random.rand(3,) - np.array([0.5,0.5,0.5])  
	axis = axis/np.linalg.norm(axis)
	# angle from [-pi/factor to +pi/factor]
	angle = 2*(np.random.uniform()-0.5) * np.pi / 1
	 
	Rot_instance = Rotation.from_rotvec(axis*angle) 
	R_ab = Rot_instance.as_matrix()

	points_rot = R_ab.dot(points.T).T
	
	
	if add_noise :
		points_rot = points_rot + np.random.rand(len(points_rot),3)*0.05
	shuffle_idx = np.random.permutation(np.arange(len(points_rot)))
	points_rot = points_rot[shuffle_idx,:]
	return points_rot,R_ab



def Network_input_format(points):
	points_cuda = points.reshape(512,3,1).astype('float32')
	points_cuda = torch.from_numpy(points_cuda)
	points_cuda = points_cuda.permute(2,1,0).cuda() # input shape shpuld be [1,3,512]
	return points_cuda

def create_pcd_obj(np_array,col=[1,0,0]):
	'''
	input: nx3 array
	output: pcd object 
			can be displayed using o3d.visualization.draw_geometries([pcd1,pcd2])
	'''

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(np_array[:,0:3])
	pcd.paint_uniform_color(col)
	return pcd

def options():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='PointNetLK_exp_', metavar='N',
						help='Name of the experiment')
    parser.add_argument('--model', type=str, default='PointNetLK', metavar='N',
						choices=['PointNetLK'],
						help='Model to use, [PointNetLK]')
    parser.add_argument('--fine_tune_pointnet', default='tune', type=str, choices=['fixed', 'tune'],
						help='train pointnet (default: tune)')
    parser.add_argument('--transfer_ptnet_weights', default='./checkpoints/exp_classifier/models/best_ptnet_model.t7', type=str,
						metavar='PATH', help='path to pointnet features file')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
						help='Dimension of embeddings')
    
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
						help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
						help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
						help='number of episode to train ')
    parser.add_argument('--optimizator', type=str, default='adam', metavar='N',
						choices=['adam, sgd'],
						help='Use optimizator (default: adam)')
    parser.add_argument('--lr', type=float, default=0.00002, metavar='LR',
						help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='LR',
						help='learning rate (default: 0.1 if using adam)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
						help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
						help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
						help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
						help='evaluate the model')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
						help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
						help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1000, metavar='N',
						help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='cranio', choices=['modelnet40','cranio'], metavar='N',
						help='dataset to use (default: modelnet40)')
    parser.add_argument('--use_normals', type=bool, default=False,metavar='N',
						help='use normals in the estimation')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
						help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
						help='Pretrained model path')
    parser.add_argument('--betas', type=float, default=(0.9,0.999), metavar='N', nargs='+',
						help='Betas for adam')
    parser.add_argument('--same_pointclouds', type=bool, default=True, metavar='N',
						help='R*src + t should be exactly same as target')
    parser.add_argument('--loss', type=str, default='mse_doubT', metavar='N',
						choices=['mse_transf', 'mse', 'chamfer_dist', 'mse_doubT'],
						help='loss function: choose one of [mse_transf or chanfer_dist]')
    parser.add_argument('--pretrained', type=bool, default = False, metavar='N',
						help='load pretrained model')	
    parser.add_argument('--scale', type=bool, default=False,metavar='N',
						help='scale factor of the model')
    
    # settings for PointNet
    parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
						help='train pointnet (default: tune)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
						help='symmetric function (default: max)')
    args = parser.parse_args()
    if not args.eval:
        args.exp_name =args.exp_name+ '_'+ timestamp
    return args

# main
def main():
    args = options()
    
    # for deterministic training
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)
    
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

	# dataloading
    num_workers = 0
    if args.dataset == 'modelnet40':

        train_dataset = RegistrationData(ModelNet40Data(train=True, num_points=args.num_points, use_normals=args.use_normals), normals=args.use_normals)
        test_dataset = RegistrationData(ModelNet40Data(train=False, num_points=args.num_points, use_normals=args.use_normals), normals=args.use_normals)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    elif args.dataset=='cranio':
        datadir = r'C:\Users\cruzguei\Documents\DataMeshCHCO'
        # train_loader, val_loader, test_loader = CranioCS('transformation3DPGraphs2PC5_P=1000.pt', shuffle=True, batch_size=args.batch_size,train_set_percentage=0.80)
        train_loader, val_loader, test_loader = DataLoaders(datadir, '*_DA1kPC.pt', batch_size=args.batch_size, shuffle=True, train_set_percentage=0.80)
    else:
        raise Exception("not implemented")

	# model loading
    if args.model == 'PointNetLK':
        ptnet = PointNet(emb_dims=args.emb_dims, use_bn=True)
        if args.transfer_ptnet_weights and os.path.isfile(args.transfer_ptnet_weights):
            ptnet.load_state_dict(torch.load(args.transfer_ptnet_weights, map_location='cpu'))
        if args.fine_tune_pointnet == 'tune':
            pass
        elif args.fine_tune_pointnet == 'fixed':
            for param in ptnet.parameters():
                param.requires_grad_(False)
               
        net = PointNetLK(feature_model=ptnet)
        # net = PCTR(args.num_points,args.emb_dims, args.scale, args.use_normals)#.to(args.device)
        if args.eval:
            if args.model_path:
                model_path = args.model_path
            else:
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.last.t7'
                print("Model loaded from ", model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
        if args.pretrained:
            if args.model_path =='':
                print ('Please specify path to pretrained weights \n For Ex: checkpoints/partial_global_512_identical/models/model.best.t7')
            else: 
                model_path = args.model_path
            print ("Using pretrained weights stored at:\n{}".format(model_path))
            net.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            args.device = 'cuda:1'
            ptnet = ptnet.to(args.device)
            net = net.to(args.device)
            # net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # args.device = 'cuda'
            
    else:
        raise Exception('Not implemented')
 
	# training and evaluation
    if args.eval:
        test(args, net, test_loader, boardio, textio)

    else:
        train(args, net, train_loader, val_loader, boardio, textio)
        test(args, net, test_loader, boardio, textio)


    print('FINISH')
    boardio.close()


if __name__ == '__main__':
	main()
