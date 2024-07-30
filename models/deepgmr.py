'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from visu_util import visualize
from ops.transform_functions import transform_point_cloud


def gmm_params(gamma, pts):
    '''
    Inputs:
        gamma: B x N x J
        pts: B x N x 3
    '''
    # pi: B x J
    pi = gamma.mean(dim=1)
    Npi = pi * gamma.shape[1]
    # mu: B x J x 3
    mu = gamma.transpose(1, 2) @ pts / Npi.unsqueeze(2)
    # diff: B x N x J x 3
    diff = pts.unsqueeze(2) - mu.unsqueeze(1)
    # sigma: B x J x 3 x 3
    eye = torch.eye(3).unsqueeze(0).unsqueeze(1).to(gamma.device)
    sigma = (
        ((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() * gamma).sum(dim=1) / Npi
    ).unsqueeze(2).unsqueeze(3) * eye
    return pi, mu, sigma


def gmm_register(pi_s, mu_s, mu_t, sigma_t):
    '''
    Inputs:
        pi: B x J
        mu: B x J x 3
        sigma: B x J x 3 x 3
    '''
    c_s = pi_s.unsqueeze(1) @ mu_s
    c_t = pi_s.unsqueeze(1) @ mu_t
    Ms = torch.sum((pi_s.unsqueeze(2) * (mu_s - c_s)).unsqueeze(3) @
                   (mu_t - c_t).unsqueeze(2) @ sigma_t.inverse(), dim=1)
    try :
        U, _, V = torch.svd(Ms.cpu())
    except Exception as error:
        print("error in SVD")
        U, _, V = torch.svd(torch.eye(3))
        # u, s, v = torch.svd(torch.eye(3))
    # U, _, V = torch.svd(Ms.cpu())
    U = U.to(pi_s)
    V = V.to(pi_s)
    S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
    try:
        S[:, 2, 2] = torch.det(V @ U.transpose(1, 2))
    except Exception as error:
        print("error in SVD")
    R = V @ S @ U.transpose(1, 2)
    t = c_t.transpose(1, 2) - R @ c_s.transpose(1, 2)
    bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
    T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
    return T


def rotation_error(R, R_gt):
    cos_theta = (torch.einsum('bij,bij->b', R, R_gt) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    return torch.acos(cos_theta) * 180 / math.pi


def translation_error(t, t_gt):
    return torch.norm(t - t_gt, dim=1)


def rmse(pts, T, T_gt):
    pts_pred = pts @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
    pts_gt = pts @ T_gt[:, :3, :3].transpose(1, 2) + T_gt[:, :3, 3].unsqueeze(1)
    return torch.norm(pts_pred - pts_gt, dim=2).mean(dim=1)


class Conv1dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(Conv1dBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True))


class FCBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(FCBNReLU, self).__init__(
            nn.Linear(in_planes, out_planes, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True))


class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.encoder = nn.Sequential(
            Conv1dBNReLU(3, 64),
            Conv1dBNReLU(64, 128),
            Conv1dBNReLU(128, 256))
        self.decoder = nn.Sequential(
            FCBNReLU(256, 128),
            FCBNReLU(128, 64),
            nn.Linear(64, 6))

    @staticmethod
    def f2R(f):
        r1 = F.normalize(f[:, :3])
        proj = (r1.unsqueeze(1) @ f[:, 3:].unsqueeze(2)).squeeze(2)
        r2 = F.normalize(f[:, 3:] - proj * r1)
        r3 = r1.cross(r2)
        return torch.stack([r1, r2, r3], dim=2)

    def forward(self, pts):
        f = self.encoder(pts)
        f, _ = f.max(dim=2)
        f = self.decoder(f)
        R = self.f2R(f)
        return R @ pts


class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()
        self.use_tnet = args.use_tnet
        self.tnet = TNet() if self.use_tnet else None
        d_input = args.k * 4 if args.use_rri else 3
        self.encoder = nn.Sequential(
            Conv1dBNReLU(d_input, 64),
            Conv1dBNReLU(64, 128),
            Conv1dBNReLU(128, 256),
            Conv1dBNReLU(256, args.d_model))
        self.decoder = nn.Sequential(
            Conv1dBNReLU(args.d_model * 2, 512),
            Conv1dBNReLU(512, 256),
            Conv1dBNReLU(256, 128),
            nn.Conv1d(128, args.n_clusters, kernel_size=1))

    def forward(self, pts):
        pts = self.tnet(pts) if self.use_tnet else pts
        f_loc = self.encoder(pts)
        f_glob, _ = f_loc.max(dim=2)
        f_glob = f_glob.unsqueeze(2).expand_as(f_loc)
        y = self.decoder(torch.cat([f_loc, f_glob], dim=1))
        return y.transpose(1, 2)


class DeepGMR(nn.Module):
    def __init__(self, args):
        super(DeepGMR, self).__init__()
        self.backbone = PointNet(args)
        self.use_rri = args.use_rri

    def regis_err(self, T_gt, reverse=False):
        if reverse:
            self.r_err_21 = rotation_error(self.T_21[:, :3, :3], T_gt[:, :3, :3])
            self.t_err_21 = translation_error(self.T_21[:, :3, 3], T_gt[:, :3, 3])
            return self.r_err_21.mean().item(), self.t_err_21.mean().item()
        else:
            self.r_err_12 = rotation_error(self.T_12[:, :3, :3], T_gt[:, :3, :3])
            self.t_err_12 = translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3])
            return self.r_err_12.mean().item(), self.t_err_12.mean().item()

    def forward(self, pts1, pts2):
        if self.use_rri:
            self.pts1 = pts1[..., :3]
            self.pts2 = pts2[..., :3]
            feats1 = pts1[..., 3:].transpose(1, 2)
            feats2 = pts2[..., 3:].transpose(1, 2)
        else:
            self.pts1 = pts1
            self.pts2 = pts2
            feats1 = (pts1 - pts1.mean(dim=1, keepdim=True)).transpose(1, 2)
            feats2 = (pts2 - pts2.mean(dim=1, keepdim=True)).transpose(1, 2)

        self.gamma1 = F.softmax(self.backbone(feats1), dim=2)
        self.pi1, self.mu1, self.sigma1 = gmm_params(self.gamma1, self.pts1)
        self.gamma2 = F.softmax(self.backbone(feats2), dim=2)
        self.pi2, self.mu2, self.sigma2 = gmm_params(self.gamma2, self.pts2)

        self.T_12 = gmm_register(self.pi1, self.mu1, self.mu2, self.sigma2)
        self.T_21 = gmm_register(self.pi2, self.mu2, self.mu1, self.sigma1)
        # self.T_gt = T_gt

        transformed_source = transform_point_cloud(pts2, self.T_21[:, :3, :3], self.T_21[:, :3, 3])

        result = {'est_R': self.T_21[:, :3, :3],
				  'est_t': self.T_21[:, :3, 3],
				  'est_R_inverse': self.T_12[:, :3, :3],
				  'est_t_inverse': self.T_12[:, :3, 3],
				  'est_T': self.T_21,
				  'est_T_inverse': self.T_12,
				  'r': feats1 - feats2,
				  'transformed_source': transformed_source}

        return result
