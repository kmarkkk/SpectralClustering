#!/usr/bin/python  

import sys
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.linalg import eig  
from scipy.cluster.vq import kmeans2  
from scipy.sparse.linalg import eigen 
from scipy.spatial.kdtree import KDTree  





  

def constructKnnGraph(points, numNeighbor, distance):
  def euclidean_kernel(a, b):
    d = np.linalg.norm(a-b)
    return d
  knn = {}
  kt = KDTree(points)
  for i, point in enumerate(points):
    for neighbour in kt.query(point, n + 1)[1]:
      if i != neighbour: 
        knn.setdefault(i, []).append((euclidean_kernel(point, points[neighbour]), neighbour))
  return knn


def constructSimilarityGraph(points):
  graphPoints = constructKnnGraph(points, numNeighbor, distance)
  return graphPoints


def main(args):  
  points = getSamplePoints()
  graphPoints = constructSimilarityGraph(points)
  W = getWeightMatrix(graphPoints)
  D = getDegreeMatrix(W)
  L = computeLaplacian(D, W)
  U = constructEigenvectorMatrix(L)
  clusterIndex = computeClusterIndex(U)
  clusters = getClusters(points, clusterIndex)

if __name__ == "__main__": 
  main(sys.argv)