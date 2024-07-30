import numpy as np
from sklearn.neighbors import KDTree
import torch


def IterativeClosestPoint(source_pts, target_pts, iter=100, tau=1e-10):
    '''
    This function implements iterative closest point algorithm based 
    on Besl, P.J. & McKay, N.D. 1992, 'A Method for Registration 
    of 3-D Shapes', IEEE Transactions on Pattern Analysis and Machine 
    Intelligence, vol. 14, no. 2,  IEEE Computer Society. 

    inputs:
    source_pts : 3 x N
    target_pts : 3 x M
    tau : threshold for convergence
    Its the threshold when RMSE does not change comapred to the previous 
    RMSE the iterations terminate. 

    outputs:
    R : Rotation Matrtix (3 x 3)
    t : translation vector (3 x 1)
    k : num_iterations
    '''
    rot=[]
    trans=[]
    for i in range(len(source_pts)):
        k = 0
        current_pts = source_pts[i].copy()
        last_rmse = 0
        t = np.zeros((3, 1))
        R = np.eye(3, 3)

        # iteration loop
        while True and iter>k:
            neigh_pts = FindNeighborPoints(current_pts, target_pts[i])
            (R, t) = RegisterPoints(source_pts[i], neigh_pts)
            current_pts = ApplyTransformation(source_pts[i], R, t)
            rmse = ComputeRMSE(current_pts, neigh_pts)
            # print("iteration : {}, rmse : {}".format(k,rmse))

            if np.abs(rmse - last_rmse) < tau:
                break
            last_rmse = rmse
            k = k + 1
        rot.append(R)
        trans.append(t)
    rotations_ab = np.concatenate(rot, axis=0)
    translations_ab = np.concatenate(trans, axis=0)
    
    return rotations_ab, translations_ab


# Computes the root mean square error between two data sets.
# here we dont take mean, instead sum.
def ComputeRMSE(p1, p2):
    return np.sum(np.sqrt(np.sum((p1-p2)**2, axis=0)))


# applies the transformation R,t on pts
def ApplyTransformation(pts, R, t):
    return np.dot(R, pts) + t

# applies the inverse transformation of R,t on pts
def ApplyInvTransformation(pts, R, t):
    return np.dot(R.T,  pts - t)

# calculate naive transformation errors
def CalcTransErrors(R1, t1, R2, t2):
    Re = np.sum(np.abs(R1-R2))
    te = np.sum(np.abs(t1-t2))
    return (Re, te)


# point cloud registration between points p1 and p2
# with 1-1 correspondance
def RegisterPoints(p1, p2):
    u1 = np.mean(p1, axis=1).reshape((3, 1))
    u2 = np.mean(p2, axis=1).reshape((3, 1))
    pp1 = p1 - u1
    pp2 = p2 - u2
    W = np.dot(pp1, pp2.T)
    U, _, Vh = np.linalg.svd(W)
    R = np.dot(U, Vh).T
    if np.linalg.det(R) < 0:
        Vh[2, :] *= -1
        R = np.dot(U, Vh).T
    t = u2 - np.dot(R, u1)
    return (R, t)


# function to find source points neighbors in
# target based on KDTree
def FindNeighborPoints(source, target):
    n = source.shape[1]
    kdt = KDTree(target.T, leaf_size=30, metric='euclidean')
    index = kdt.query(source.T, k=1, return_distance=False).reshape((n,))
    return target[:, index]
