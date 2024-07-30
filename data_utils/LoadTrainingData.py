import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from DataSetGraph_SphericalMaps import LoadedDataset
from pathlib import Path
import pdb
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt

datadir = r'C:\Users\cruzguei\OneDrive - The University of Colorado Denver\SphericalMapAutoEncoder'
files = [str(x) for x in Path(datadir).glob('*.pt')]

dataset = LoadedDataset(files)
loader =  DataLoader(dataset, batch_size=1, pin_memory=True)
sexH=0
sexM=0
CD_SagCS=0
CD_MetCS=0
age=[]
CfDia=[]
PostOp=0
for i in loader:
    # print(i.x)
    # print(i.y)
    # print(i.pinfo)
    if i.pinfo['PtSex']==['M']:
        sexH+=1
    else:
        sexM+=1
    if i.pinfo['Time from surgery (days)'].item() >0:
        PostOp+=1
    age.append(i.pinfo['Age at image (days)'].item())
    CfDia.extend(i.pinfo['Craniofacial Diagnosis'])
    # if i.pinfo['Craniofacial Diagnosis']==['Sagittal Craniosynostosis']:
    #     CD_SagCS+=1
    # if i.pinfo['Craniofacial Diagnosis']==['Metopic Craniosynostosis']:
    #     CD_MetCS+=1
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (Metopic and Sagittal Craniosynostosis)']:
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (Sagittal Craniosynostosis and Coronal)']:
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (Sagittal and medial lambdoid)']:
    # if i.pinfo['Craniofacial Diagnosis']==['Bilateral Coronal Craniosynostosis']:
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (Right lambdoid, Bilateral Coronal and Metopic)']:
    # if i.pinfo['Craniofacial Diagnosis']==['Right Unilateral Coronal Craniosynostosis']:
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (involved sutures not identified), Hydrocephalus']:
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (Metopic, Anterior Sagittal and Bilateral Coronal)']:
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (Sagittal, Metopic, Bilateral Coronal, Bilateral Squamosal and partial superior lambdoid), Microcephaly']:
    # if i.pinfo['Craniofacial Diagnosis']==['Right Unilateral Coronal Craniosynostosis, Right anterior plagiocephaly, Left Unilateral Coronal Craniosynostosis']:
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (Sagittal and Lambdoid), Hypertelorism, Shunted Hydrocephalus, frontonasal dysplasia, Chiari 1.5 malformation']:
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (Right Unilateral Coronal and Right Unilateral Squamosal)']:
    # if i.pinfo['Craniofacial Diagnosis']==['Multisutural Craniosynostosis (Sagittal and Bilateral Coronal)']:
    # cora = i
    # coragraph = to_networkx(cora,to_undirected=True)
    # # node_labels = cora.y[list(coragraph.nodes)].numpy()
    # plt.figure(1,figsize=(14,12)) 
    # nx.draw(coragraph)
    # plt.show()