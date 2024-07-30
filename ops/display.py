import csv
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
# import transforms3d.euler as t3d
# import transforms3d.axangles as t3d_axang
# import tensorflow as tf
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

###################### Display Operations #########################

# Display data inside ModelNet files.
def display_clouds(filename,model_no):
    # Arguments:
        # filename:         Name of file to read the data from. (string)
        # model_no:         Number to choose the model inside that file. (int)

    data = []
    # Read the entire data from that file.
    with open(os.path.join('data','templates',filename),'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row = [float(x) for x in row]
            data.append(row)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    data = np.asarray(data)

    start_idx = model_no*2048
    end_idx = (model_no+1)*2048
    data = data[start_idx:end_idx,:]        # Choose specific data related to the given model number.

    X,Y,Z = [],[],[]
    for row in data:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    ax.scatter(X,Y,Z)
    plt.show()

# Display given Point Cloud Data in blue color (default).
def display_clouds_data(data):
    # Arguments:
        # data:         array of point clouds (num_points x 3)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    try:
        data = data.tolist()
    except:
        pass
    X,Y,Z = [],[],[]
    for row in data:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    ax.scatter(X,Y,Z)
    ax.set_aspect('equal')
    plt.show()

 

def display_two_clouds(data1,data2,title="title",disp_corr=False):
    # Arguments:
        # data1         Template Data (num_points x 3) (Red)
        # data2         Source Data (num_points x 3) (Green)
         

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
        data3 = data3.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=[1,0,0,1])
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=[0,1,0,0.5])
    
    if disp_corr:
        for i in range(len(data1)):
            ax.plot3D([data1[i][0],data2[i][0] ], \
                  [data1[i][1],data2[i][1] ], \
                  [data1[i][2],data2[i][2] ], \
                   c=[0,0,1,0.1] )
 

    # Add details to Plot.
    plt.legend((l1,l2),('training data','Source Data' ),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=15)
    ax.set_ylabel('Y-axis',fontsize=15)
    ax.set_zlabel('Z-axis',fontsize=15)
    # ax.set_xlim(-1,1.25)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-0.5,1.25)
    plt.title(title,fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    ax.set_aspect('equal')
    
    plt.show()

def display_two_clouds_corr_mat(data1,data2,corr_mat,text=["title","pointcloud1","pointcloud2"],disp_time=3):
    '''
    Arguments:
        data1         source (num_points x 3) (Red)
        data2         target (num_points x 3) (Green)

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    data3 = (data2.T.dot(corr_mat)).T 
    # print (data3,"data3")
 
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=np.array([1,0,0,1]))
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=np.array([0,1,0,1]))

    # X,Y,Z = [],[],[]
    # for row in data3:
    #   X.append(row[0])
    #   Y.append(row[1])
    #   Z.append(row[2])
    # l3 = ax.scatter(X,Y,Z,c=np.array([1,1,0,1]))
 
 
    for i in range(len(data1)):
        if np.random.rand() > 0.6:
            ax.plot3D([data1[i][0],data3[i][0]] , \
                      [data1[i][1],data3[i][1]] , \
                      [data1[i][2],data3[i][2]] , \
                       '--',c=np.array([0,0,1,1])) 
            # ax.plot3D([data1[i][0],data2[corr_idx_gt[i]][0] ], \
            #         [data1[i][1],data2[corr_idx_gt[i]][1] ], \
            #         [data1[i][2],data2[corr_idx_gt[i]][2] ], \
            #          c=np.array([0,0,0,0.5]))
 
    # Add details to Plot.
    plt.legend((l1,l2),(text[1],text[2]),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=5)
    ax.set_ylabel('Y-axis',fontsize=5)
    ax.set_zlabel('Z-axis',fontsize=5)
    # ax.set_xlim(-1,1.25)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-0.5,1.25)
    plt.title(text[0],fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    # ax.set_aspect('equal')
    
    plt.show(block=False)
    try:
        plt.pause(disp_time)
        plt.close()
    except:
        pass


def display_two_clouds_correspondence(data1,data2,corr_idx,corr_idx_gt,title="title",disp_time=3):
    '''
    Issue with this function:
    shuffles source data. 
    We generally want to shuffle target data and transform source data.
    '''
    # Arguments:
        # data1         Template Data (num_points x 3) (Red)
        # data2         Source Data (num_points x 3) (Green)
         

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
        data3 = data3.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=np.array([1,0,0,1]))
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=np.array([0,1,0,1]))

    # X,Y,Z = [],[],[]
    # for row in data3:
    #   X.append(row[0])
    #   Y.append(row[1])
    #   Z.append(row[2])
    # l3 = ax.scatter(X,Y,Z,c=np.array([1,1,0,1]))
 
 
    for i in range(len(corr_idx)):
        ax.plot3D([data1[i][0],data2[corr_idx[i]][0] ], \
                  [data1[i][1],data2[corr_idx[i]][1] ], \
                  [data1[i][2],data2[corr_idx[i]][2] ], \
                   '--',c=np.array([0,0,1,1]) )
        # ax.plot3D([data1[i][0],data2[corr_idx_gt[i]][0] ], \
        #         [data1[i][1],data2[corr_idx_gt[i]][1] ], \
        #         [data1[i][2],data2[corr_idx_gt[i]][2] ], \
        #          c=np.array([0,0,0,0.5]))
 
    # Add details to Plot.
    plt.legend((l1,l2),('transformed source','source'),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=15)
    ax.set_ylabel('Y-axis',fontsize=15)
    ax.set_zlabel('Z-axis',fontsize=15)
    # ax.set_xlim(-1,1.25)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-0.5,1.25)
    plt.title(title,fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    ax.set_aspect('equal')
    
    plt.show(block=False)
    try:
        plt.pause(disp_time)
        plt.close()
    except:
        pass     
# Display given template, source and predicted point cloud data.
def display_three_clouds(data1,data2,data3,title,legend_list):
    # Arguments:
        # data1         Template Data (num_points x 3) (Red)
        # data2         Source Data (num_points x 3) (Green)
        # data3         Predicted Data (num_points x 3) (Blue)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
        data3 = data3.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=[0,0,1,1],s=10)
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=[1,0,0,1],s=10)
    # Add Predicted Data in Plot
    X,Y,Z = [],[],[]
    for row in data3:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l3 = ax.scatter(X,Y,Z,c=[0,1,0,0.5],s=30)

    # Add details to Plot.
    plt.legend((l1,l2,l3),(legend_list[0],legend_list[1],legend_list[2]),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=15)
    ax.set_ylabel('Y-axis',fontsize=15)
    ax.set_zlabel('Z-axis',fontsize=15)
    # ax.set_xlim(-1,1.25)
    # ax.set_ylim(-1,1)
    # ax.set_zlim(-0.5,1.25)
    plt.title(title,fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    # ax.set_aspect('equal')

    plt.show()


def display_n_clouds(data_list,
                     legend_list,
                     color_list=['y','r','g','k','b','m','c'],
                     title="n_clouds"):
    """
    max 7 colors
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(legend_list)):
        data = data_list[i]
        ax.scatter(data[:,0],data[:,1],data[:,2], 
                   c=color_list[i],
                   s = 30-6*i,
                   alpha = 0.3+i/7.,
                   label=legend_list[i]) 
        
    
    plt.title(title,fontdict={'fontsize':25})
    ax.legend(fontsize=20,loc=1)
    plt.show(block=False)   




# Display template, source, predicted point cloud data with results after each iteration.
def display_itr_clouds(data1,data2,data3,ITR,title):
    # Arguments:
        # data1         Template Data (num_points x 3) (Red)
        # data2         Source Data (num_points x 3) (Green)
        # data3         Predicted Data (num_points x 3) (Blue)
        # ITR           Point Clouds obtained after each iteration (iterations x batch_size x num of points x 3) (Yellow)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    print(ITR.shape)        # Display Number of Point Clouds in ITR.
    try:
        data1 = data1.tolist()
        data2 = data2.tolist()
        data3 = data3.tolist()
    except:
        pass
    # Add Template Data in Plot
    X,Y,Z = [],[],[]
    for row in data1:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l1 = ax.scatter(X,Y,Z,c=[1,0,0,1])
    # Add Source Data in Plot
    X,Y,Z = [],[],[]
    for row in data2:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l2 = ax.scatter(X,Y,Z,c=[0,1,0,1])
    # Add Predicted Data in Plot
    X,Y,Z = [],[],[]
    for row in data3:
        X.append(row[0])
        Y.append(row[1])
        Z.append(row[2])
    l3 = ax.scatter(X,Y,Z,c=[0,0,1,1])
    # Add point clouds after each iteration in Plot.
    for itr_data in ITR:
        X,Y,Z = [],[],[]
        for row in itr_data[0]:
            X.append(row[0])
            Y.append(row[1])
            Z.append(row[2])
        ax.scatter(X,Y,Z,c=[1,1,0,0.5])

    # Add details to Plot.
    plt.legend((l1,l2,l3),('Template Data','Source Data','Predicted Data'),prop={'size':15},markerscale=4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('X-axis',fontsize=15)
    ax.set_ylabel('Y-axis',fontsize=15)
    ax.set_zlabel('Z-axis',fontsize=15)
    plt.title(title,fontdict={'fontsize':25})
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.zaxis.set_tick_params(labelsize=15)
    plt.show()

def display_rotation_vects_SO3(rot_vects):
    # print (rot_vects)
    # print (rot_vects.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = np.multiply(rot_vects[:,0],rot_vects[:,3])
    ys = np.multiply(rot_vects[:,1],rot_vects[:,3])
    zs = np.multiply(rot_vects[:,2],rot_vects[:,3])

    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
