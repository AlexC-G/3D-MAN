# 3D Mesh Alignment Network

Python implementation of 3D Mesh Alignment Network. <br><br>

Three-dimensional (3D) photogrammetry is an emerging imaging modality to study patient morphology, detect pathologic anomalies and track disease progression. It has recently gained special popularity to assess pediatric development because of its non-ionizing, non-invasive and cost-effective nature. However, while spatial registration is an important step to enable longitudinal and/or population-based analyses, most deep learning registration methods are designed for voxel-based image representations and their application to 3D photograms is limited, since they are represented as meshes with unordered points with variable resolutions connected by triangles. Although recent deep learning models to register meshes have been proposed, most require either spatial resampling to compensate for different number of points between input meshes or the prior identification of sparse landmarks, which compromise accuracy and increases inference cost. We present a novel geometric learning architecture that incorporates a new feature homogenization mechanism to transform spatial information from meshes with diverse numbers of points to a uniform dimensionality, using geometric convolutions via Chebyshev polynomials to exploit local structural information. Moreover, our model combines offset- and cross-attention between input meshes for improved registration. We first demonstrated the improvements of our offset cross-attention module in the registration task using the publicly available ModelNet40 dataset with point clouds representing diverse objects. Then, we showed state-of-the-art performance of our model incorporating geometric feature homogenization using a database with 1,744 manually annotated craniofacial 3D photograms of children with pathology. Unlike existing methods, this model does not require spatial sampling or prior landmark identification. <br><br>

<p align="center">
<img src='\Figures\Architecture.png'> <br>
</p>
Proposed network architecture with two modules: feature homogenization and offset cross-attention. The number next to each layer indicates its number of output features.<br><br>

## Dependencies:
- [Python](python.org)
- [Pytorch](https://pytorch.org/get-started/locally)
- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [NumPy](https://numpy.org/install/)
- [SimpleITK](https://simpleitk.org/)
- [VTK](https://pypi.org/project/vtk/)
- [scipy](https://scipy.org/)

    *Once Python is installed, each of these packages can be downloaded using [Python pip](https://pip.pypa.io/en/stable/installation/)*


## Using the code
Due to data privacy agreements, we are not able to share the 3D photograms dataset and the results are sensured for privacy. All the 3D photograms were transformed to pytorch geometric graphs and saved in ".pt" format

### Quick summary
**Input**: Two meshes in format of graphs: Target and Source.

**Output**: The mesh transformation: Rotation, Translation and Scaler.

## Results:
The results are compared to five state-of-the-art point cloud registration methods: (i) Deep Closest Point ([DCP](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)), [PointnetLK](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aoki_PointNetLK_Robust__Efficient_Point_Cloud_Registration_Using_PointNet_CVPR_2019_paper.pdf) , Deep Gaussian Mixture Registration ([DeepGMR](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_43)), and Point Cloud Registration Network ([PCRnet](https://arxiv.org/pdf/1908.07906)).<br><br>

<p align="center">
<img src='\Figures\ModelNetResults.png'> <br>
</p>
Qualitative registration results in Experiment 1 using the ModelNet40 dataset. Source and target point clouds are represented in red and blue, respectively.<br><br>

<p align="center">
<img src='\Figures\PointCloudResults.png'> <br>
</p>
Qualitative results of 3D-MAN registration in Experiment 3. Note that the method uses the origi-nal mesh triangles but point clouds were used only in this figure for an improved visualization. Source and target meshes are represented in red and blue, respectively. a) Registration of two meshes from the same subject. b) Registration of two meshes from the same subject but with different scale. c) Registration of two meshes from different subjects at the same scale. d) Regis-tration of two meshes from the different subjects at different scales.<br><br>


