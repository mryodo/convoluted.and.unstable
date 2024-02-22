import numpy as np
import scipy as sc
import torch
from utils import *

def delaunay2edges(tri):
      list_of_edges = []
      for triangle in tri.simplices:
            for e1, e2 in [[0,1],[1,2],[2,0]]:
                  list_of_edges.append(less_first(triangle[e1],triangle[e2]))
      array_of_edges = np.unique(list_of_edges, axis=0)
      return array_of_edges

def B1fromEdges(n, edges):
      B1=np.zeros((n, edges.shape[0]))

      for i in range(edges.shape[0]):
            B1[int(edges[i, 0]),i]=-1
            B1[int(edges[i, 1]),i]=1
      return sc.sparse.csr_matrix(B1)

def B2fromTrig(edges, trigs):
      B2=np.zeros((edges.shape[0], trigs.shape[0]))

      for i in range(trigs.shape[0]):
            B2[np.where((edges==np.array([trigs[i, 0], trigs[i, 1]])).all(axis=1))[0][0] ,i]=1
            B2[np.where((edges==np.array([trigs[i, 0], trigs[i, 2]])).all(axis=1))[0][0] ,i]=-1
            B2[np.where((edges==np.array([trigs[i, 1], trigs[i, 2]])).all(axis=1))[0][0] ,i]=1
      return sc.sparse.csr_matrix(B2)

def getEdge2Trig( edges2, trigs2 ):
      edg2Trig = []
      for i in range(edges2.shape[0]):
            tmp = []
            for j in range(trigs2.shape[0]):
                  if ((trigs2[j, 0] == edges2[i, 0]) and (trigs2[j, 1] == edges2[i, 1])) or ((trigs2[j, 0] == edges2[i, 0]) and (trigs2[j, 2] == edges2[i, 1])) or ((trigs2[j, 1] == edges2[i, 0]) and (trigs2[j, 2] == edges2[i, 1])):
                        tmp.append( j )
            edg2Trig.append( set(tmp) )
      return edg2Trig

def getTrig2Edge( edges2, trigs2, edg2Trig ):
      trig2Edg = [ set() for i in range( trigs2.shape[0] ) ]
      for i in range( edges2.shape[0] ):
            for j in edg2Trig[i]:
                  trig2Edg[j].add(i)
      return trig2Edg


def generateTriangulation( N = 10, instable = False, device = "cpu" ):
      points = np.random.rand( N, 2) * 0.8 + 0.1
      points = np.vstack([ points, np.array([ [ 0, 0], [1, 0], [0, 1], [1, 1]]) ])

      n = N + 4
      tri = sc.spatial.Delaunay(points)
      trians = tri.simplices

      trians = np.sort( trians, axis = 1 )
      trians = trians[np.lexsort((trians[:, 2], trians[:, 1], trians[:, 0]))]

      row = torch.randint(trians.shape[0], size = (1, 1))[0][0].item()
      trians = trians[torch.arange(1, trians.shape[0]+1) != row, ...]
      row = torch.randint(trians.shape[0], size = (1, 1))[0][0].item()
      trians = trians[torch.arange(1, trians.shape[0]+1) != row, ...]

      edges = delaunay2edges(tri)

      B1 = B1fromEdges( n, edges )
      B2 = B2fromTrig( edges, trians )

      edg2Trig = getEdge2Trig( edges, trians )
      trig2Edge = getTrig2Edge( edges, trians, edg2Trig )


      w_e = 0.99 * np.ones( edges.shape[0] ) + 0.02 * np.random.rand( edges.shape[0] )
      if (instable):
            eps = 1e-3
            w_e[9] = eps
            w_e[15] = eps
            w_e[45] = eps
      w = np.array( [ np.min( w_e[ list( trig2Edge[i] ) ] )
            for i in range( trians.shape[0] )
      ] )

      W2 = sc.sparse.diags( np.sqrt(w) )
      W1 = sc.sparse.diags( np.sqrt(w_e) )

      tmp = B1 @ W1 @ W1 @ B1.T
      tmp.setdiag(0)
      W0inv = sc.sparse.diags( np.asarray( 1 / np.sqrt( - tmp.sum( axis = 1 ) ) ).reshape(-1) )

      B1w = W0inv @ B1 @ W1
      B2w = sc.sparse.diags( 1/np.sqrt(w_e) ) @ B2 @ W2

      Ld = B1w.T @ B1w
      Lu = B2w @ B2w.T

      return coo_to_torch( Ld, device = device ), coo_to_torch( Lu , device = device), coo_to_torch( B1w, device = device ), coo_to_torch( B2w , device = device), coo_to_torch( W0inv , device = device), coo_to_torch( W1 , device = device), coo_to_torch( W2, device = device ), edges, trians, n, points, edg2Trig, trig2Edge