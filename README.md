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
**Input**: Two meshes in format of graphs: Target and Source. These meshes can be differet from each other, with distinct number of nodes and edges.

**Output**: The mesh transformation: Rotation, Translation and Scaler.
<br><br>
## Results:
The results are compared to five state-of-the-art point cloud registration methods: (i) Deep Closest Point ([DCP](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)), [PointnetLK](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aoki_PointNetLK_Robust__Efficient_Point_Cloud_Registration_Using_PointNet_CVPR_2019_paper.pdf) , Deep Gaussian Mixture Registration ([DeepGMR](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_43)), and Point Cloud Registration Network ([PCRnet](https://arxiv.org/pdf/1908.07906)).<br><br>

<p align="center">
<img src='\Figures\ModelNetResults.png'> <br>
</p>
Qualitative registration results in Experiment 1 using the ModelNet40 dataset. Source and target point clouds are represented in red and blue, respectively.<br><br>

<p align="center">
<img src='\Figures\PointCloudResults.png'> <br>
</p>
Qualitative results of 3D-MAN registration in Experiment 3. Note that the method uses the original mesh triangles but point clouds were used only in this figure for an improved visualization. Source and target meshes are represented in red and blue, respectively. a) Registration of two meshes from the same subject. b) Registration of two meshes from the same subject but with different scale. c) Registration of two meshes from different subjects at the same scale. d) Registration of two meshes from the different subjects at different scales.<br><br>

## Acknowledgments 
This work is supported by the National Institute of Dental & Craniofacial Research (NIDCR) of the National Institutes of Health (NIH) under Award Number R01DE032681. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. <br><br>

## Citation
This work will present in the Workshop on GRaphs in biomedicAl Image anaLysis ([GRAIL2024](https://grail-miccai.github.io/#program)) from MICCAI2024 and it will publish in the workshop proceedings volume. If you use this code or ideas from our paper, please cite our paper:<br> <br>
* Mesh registration via geometric feature homogenization and offset cross-attention: application to 3D photogrammetry<br>
 [Inés A. Cruz-Guerrero](https://orcid.org/0000-0001-8034-8530)<sup>1</sup>,
 [Connor Elkhill](https://orcid.org/0000-0001-8753-9575) <sup>1</sup>,
 Jiawei Liu <sup>1</sup>,
 Phuong Nguyen <sup>2</sup>,
 Brooke French <sup>2</sup>, and
 [Antonio R. Porras](https://orcid.org/0000-0001-5989-2953)<sup>1,2,3,4</sup> <br>
<sup>1</sup> Department of Biostatistics and Informatics, Colorado School of Public Health, University of Colorado Anschutz Medical Campus, Aurora, CO <br>
<sup>2</sup> Department of Pediatric Plastic and Reconstructive Surgery, Children's Hospital Colorado, Aurora, CO <br>
<sup>3</sup> Department of Pediatric Neurosurgery, Children’s Hospital Colorado, Aurora, CO <br>
<sup>4</sup> Departments of Pediatrics, Surgery and Biomedical Informatics, School of Medicine, University of Colorado Anschutz Medical Campus, Aurora, CO <br><br>

```
@INPROCEEDINGS{Cruz-Guerrero2024,
  author={Inés A. Cruz-Guerrero, Connor Elkhill, Jiawei Liu, Phuong Nguyen, Brooke French, and Antonio R. Porras},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention}, 
  title={Mesh registration via geometric feature homogenization and offset cross-attention: application to 3D photogrammetry}, 
  year={2024},
  note={in press},
  organization={Springer},
  doi={}}
```
<br><br>

## Contact 
If you have any questions, please email Alejandro Cruz-Guerrero at alejandro.cruzguerrero@cuanschutz.edu
