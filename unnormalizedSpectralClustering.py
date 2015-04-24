#!/usr/bin/python  

import sys
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.linalg import eig  
from scipy.cluster.vq import kmeans2  
from scipy.sparse.linalg import eigen 
from scipy.spatial.kdtree import KDTree  








def main(args):  
  points = getSamplePoints()
  graphPoints = constructSimilarityGraph()
  W = getWeightMatrix(graphPoints)
  D = getDegreeMatrix(W)
  L = computeLaplacian(D, W)
  U = constructEigenvectorMatrix(L)
  clusterIndex = computeClusterIndex(U)
  clusters = getClusters(points, clusterIndex)

if __name__ == "__main__": 
  main(sys.argv)