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
    for neighbour in kt.query(point, numNeighbor + 1)[1]:
      if i != neighbour: 
        knn.setdefault(i, []).append((euclidean_kernel(point, points[neighbour]), neighbour))
  return knn


def constructSimilarityGraph(points):
  graphPoints = constructKnnGraph(points, numNeighbor, distance)
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
  clusterIndex = []
  for i in range(0, len(points) + 1):
    clusterIndex[idx[i]].append(i) 
  return clusterIndex

def getClusters(points, clusterIndex):
  clusterPoints = []
  for i in range(0, k + 1):
    for j in range(0, len(clusterIndex[k]))
      clusterPoints[i].append(points[clusterIndex[j]])
  return clusterPoints

  

def main(args):  
  numClusters = 10
  points = getSamplePoints()
  graphPoints = constructSimilarityGraph(points)
  W = getWeightMatrix(graphPoints)
  D = getDegreeMatrix(W)
  L = computeLaplacian(D, W)
  Y = constructEigenvectorMatrix(L, numClusters)
  clusterIndex = computeClusterIndex(Y, numClusters, points)
  clustersPoints = getClusters(points, clusterIndex)

if __name__ == "__main__": 
  main(sys.argv)