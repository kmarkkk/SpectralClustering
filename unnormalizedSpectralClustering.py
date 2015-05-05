#!/usr/bin/python  

import sys
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.io import loadmat
from scipy.linalg import eig  
from scipy.cluster.vq import kmeans2  
from scipy.sparse.linalg import eigen 
from scipy.spatial.kdtree import KDTree  
  

def constructKnnGraph(points, numNeighbor):
  def euclidean_kernel(a, b):
    d = np.linalg.norm(a-b)
    return d
  knn = {}
  kt = KDTree(points)
  for i, point in enumerate(points):
    for neighbour in kt.query(point, numNeighbor + 1)[1]:
      if i != neighbour: 
        knn.setdefault(i, []).append((euclidean_kernel(point, points[neighbour]), neighbour))
  return knn


def constructSimilarityGraph(points):
  numNeighbor = 10
  graphPoints = constructKnnGraph(points, numNeighbor)
  return graphPoints

def getWeightMatrix(graphPoints):  
  n = len(graphPoints)  
  W = np.zeros((n, n))  
  for point, nearest_neighbours in graphPoints.iteritems():  
      for distance, neighbour in nearest_neighbours:  
          W[point][neighbour] = distance  
  return W  

def getDegreeMatrix(W):  
  D = np.diag([sum(Wi) for Wi in W])
  return D

def computeLaplacian(D, W):
  L = D - W
  return L

def constructEigenvectorMatrix(L, numClusters):
  evals, evcts = eig(L) 
  evals, evcts = evals.real, evcts.real 
  edict = dict(zip(evals, evcts.transpose())) 
  evals = sorted(edict.keys()) 
  Y = np.array([edict[k] for k in evals[0:numClusters]]).transpose()
  return Y

def computeClusterIndex(Y, numClusters, points):
  res, idx = kmeans2(Y, numClusters, minit='random')
  print idx
  clusterIndex = [[] for x in range(numClusters)]
  for i in range(0, len(points)):
    clusterIndex[idx[i]].append(i) 
  return clusterIndex

def getClusters(points, clusterIndex):
  numClusters = len(clusterIndex)
  clusterPoints = [[] for x in range(numClusters)]
  for i in range(0, len(clusterIndex)):
    for j in range(0, len(clusterIndex[i])):
      
      clusterPoints[i].append(points[clusterIndex[i][j]])
  return clusterPoints

def getSamplePoints():
  pt1 = np.random.normal(1, 0.2, (100,2))
  pt2 = np.random.normal(2, 0.5, (300,2))
  pt3 = np.random.normal(3, 0.3, (100,2))
  pt2[:,0] += 1
  pt3[:,0] -= 0.5
  xy = np.concatenate((pt1, pt2, pt3))
  return xy

def getData(dir):
  data = loadmat(dir)
  b = np.array(data['images'])
  return b

def main(args):  
  numClusters = 3
  points = getData('images.mat') # getSamplePoints()
  sizes = points.shape
  new_data = []
  for i in range(50): #(sizes[2]):
    one_point = points[:, :, i]
    one_point = one_point.reshape((1, 784))
    new_data.append(one_point)
  graphPoints = constructSimilarityGraph(new_data)
  W = getWeightMatrix(graphPoints)
  D = getDegreeMatrix(W)
  L = computeLaplacian(D, W)
  Y = constructEigenvectorMatrix(L, numClusters)
  clusterIndex = computeClusterIndex(Y, numClusters, points)
  clustersPoints = getClusters(points, clusterIndex)
  

if __name__ == "__main__": 
  main(sys.argv)