import torch

# generate output and missing values

def output_generator( m1, W1, edg2Trig, var = 0., device = "cpu" ):
      return torch.tensor( [ W1.to_dense().diag()[list(entry)].sum() for entry in edg2Trig ], device = device ) + var * torch.randn(m1, device = device)

def get_missing( m1, dropRate = 0.2, valRate = 0.1, device = "cpu" ):
      res = torch.randperm( m1, device = device )
      return torch.sort(res[ : int( m1*dropRate ) ]).values, torch.sort(res[ int( m1*dropRate ) : int( m1*dropRate ) + int( m1*valRate )  ]).values, torch.sort(res[ int( m1*dropRate ) + int( m1*valRate ) : ]).values