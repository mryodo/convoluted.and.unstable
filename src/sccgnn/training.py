from torcheval.metrics import R2Score  #maybe there is a better way to get R2?
import torch
from math import log
import torch.nn as nn

def P1( x, B1w, b1b1t_inv ):
      return B1w.T @ ( b1b1t_inv @ ( B1w @ x ) )
      #return B1w.T @ torch.linalg.pinv( (B1w @ B1w.T).to_dense() ) @ B1w @ x
      #return B1w.T @ torch.linalg.lstsq( (B1w @ B1w.T).to_dense(), B1w @ x  ).solution

def P2( x, B2w, b2tb2_inv ):
      return B2w @ ( b2tb2_inv @ ( B2w.T @ x ) )
      #return B2w @ torch.linalg.pinv( (B2w.T @ B2w).to_dense() ) @ B2w.T @ x
      #return B2w @ torch.linalg.lstsq( (B2w.T @ B2w).to_dense(), B2w.T @ x ).solution


# Losses and accuracies (GLOBAL VARIABLES EVEREYWHERE)

class classicalLoss( nn.Module ):
      def __init__(self, saved):
            super().__init__()
            self.saved = saved

      def forward(self, output, target):
            return torch.mean((output[self.saved] - target[self.saved])**2 ) 

def my_accuracy( output, target, ind  ):
      
      accuracy = R2Score( )
      accuracy.update( output[ind], target[ind] )
      return accuracy.compute()

def MAPE( output, target, ind ):
    abs_error = (torch.abs(target[ind] - output[ind])) / torch.abs(target[ind])
    sum_abs_error = torch.sum(abs_error)
    mape_loss = (sum_abs_error / target[ind].shape[0]) * 100
    return mape_loss


def my_val_accuracy( output, target, val_ind  ):
      
      accuracy = R2Score( )
      accuracy.update( output[val_ind], target[val_ind] )
      return accuracy.compute()

class componentLoss( nn.Module ):
      def __init__(self, saved, B1w, B2w, b1b1t_inv, b2tb2_inv, α1, α2):
            super().__init__()
            self.B1w, self.B2w, self.b1b1t_inv, self.b2tb2_inv, self.α1, self.α2, self.saved  = B1w, B2w, b1b1t_inv, b2tb2_inv, α1, α2, saved

      def forward(self, inputs, targets):
            diff_vec = inputs - targets
            return torch.mean(( P1(diff_vec, self.B1w, self.b1b1t_inv))[self.saved]**2 ) +self.α1 * torch.mean(  P2(diff_vec, self.B2w, self.b2tb2_inv)[self.saved]**2 ) + self.α2 * torch.mean( ( diff_vec[self.saved] - P1(diff_vec, self.B1w, self.b1b1t_inv)[self.saved] - P2(diff_vec, self.B2w, self.b2tb2_inv)[self.saved] ) ** 2 )


# masking and dataset generation per epoch
def generate_mask( p, y, saved, device = "cpu" ):
      e = torch.zeros_like( y , device = device)
      for i, _ in enumerate( e ):
            if i in saved:
                  if torch.rand(1) < p:
                        e[i] = y[i]
      return e

def form_epoch_data( y, y_real, saved, fillerValue, p = 0.75, multiplier = 4, device = "cpu" ):
      tries = int( multiplier * round( log( y.shape[0] ) / p ) )
      mask = torch.zeros( y.shape[0], tries , device = device)
      for i in range( tries ):
            mask[:, i] = generate_mask( p, y, saved, device = device )
      mask[ mask == 0 ] = fillerValue

      return [ ( mask[ :, i ], y_real )  for i in range(tries) ]