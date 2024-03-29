# need to check if we need this imports
import torch
import numpy as np

def less_first(a, b):
      return [a,b] if a < b else [b,a]

def coo_to_torch( coo, device = "cpu" ):
      coo = coo.tocoo()
      values = coo.data
      indices = np.vstack((coo.row, coo.col))

      i = torch.LongTensor(indices)
      v = torch.FloatTensor(values)
      shape = coo.shape


      if device == torch.device("mps"):
            return torch.tensor( coo.todense(), dtype = torch.float32, device = device )
      else:
            return torch.sparse_coo_tensor(i, v, torch.Size(shape), device = device)

def condPlus( Lu, thr = 1e-6, device = "cpu" ):
      if device == torch.device("mps"): 
            lam = np.linalg.eigvalsh(Lu.numpy(force=True)) 
            return lam[-1] / lam[np.abs(lam) >  thr][0]
      else:
            lam = torch.linalg.eigvalsh(Lu.to_dense())
            return lam[-1] / lam[torch.abs(lam) >  thr][0]

def get_top_eig( Lu, Ld, device = "cpu" ):
      if device == torch.device("mps"):
            lamU = np.linalg.eigvalsh(Lu.numpy(force=True))
            lamD = np.linalg.eigvalsh(Ld.numpy(force=True))
      else:
            lamU = torch.linalg.eigvalsh(Lu.to_dense())
            lamD = torch.linalg.eigvalsh(Ld.to_dense())
      return torch.max( torch.tensor( [ lamU[-1], lamD[-1] ], device = device, dtype = torch.float32 ) )

def build_stacks( m1, Ld, Lu, K = 5, device = "cpu" ):
      LdStack = torch.zeros( m1, m1, K, device = device )
      LdStack[:, :, 0] = torch.eye( m1 )
      for i in range(1, K):
            LdStack[ :, :, i ] = LdStack[ :, :, i - 1 ] @ Ld

      LuStack = torch.zeros( m1, m1, K, device = device )
      LuStack[:, :, 0] = torch.eye( m1 )
      for i in range(1, K):
            LuStack[ :, :, i ] = LuStack[ :, :, i - 1 ] @ Lu

      return LdStack, LuStack


def initial_weights(in_size, out_size, K, gamma = 0.2, variance = 1.0, device = "cpu"):
      tmp = variance*torch.rand((in_size, out_size, K), device = device)
      for i in range(1, K):
            tmp[:, :, i] = (gamma**i) * tmp[:, :, i]  
      return tmp