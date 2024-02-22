import torch.nn as nn
import torch

def initial_weights(in_size, out_size, K, gamma = 0.2, variance = 1.0, device = "cpu"):
      tmp = variance*torch.rand((in_size, out_size, K), device = device)
      for i in range(1, K):
            tmp[:, :, i] = (gamma**i) * tmp[:, :, i]  
      return tmp
#LAYER
class SC( nn.Module ):
      def __init__(self, in_size, out_size, K, variance = 1.0, device = "cpu", gamma = 0.1):
            super().__init__()
            self.device = device
            self.in_size = in_size
            self.out_size = out_size
            self.K = K
            self.gamma = gamma
            self.variance = variance
            self.coeff_P = nn.parameter.Parameter(initial_weights(self.in_size, self.out_size, self.K, gamma = self.gamma, variance = self.variance , device = self.device ) )
            self.coeff_Q = nn.parameter.Parameter(initial_weights(self.in_size, self.out_size, self.K, gamma = self.gamma, variance = self.variance , device = self.device ) )

            

      def forward(self, LdStack, LuStack, x):
            # here we need to write the assemble of the polynomial MAY BE NOT OPTIMAL
            return sum([ LdStack[:, :, i] @ x @ self.coeff_P[:, :, i] for i in range( self.K ) ]) + sum([ LuStack[:, :, i] @ x @ self.coeff_Q[:, :, i] for i in range( self.K ) ])

#NETWORK
class SCCGNN( nn.Module ):
      def __init__(self, K = 5, L = 3, variance = 0.01, device = "cpu" ):
            super().__init__()
            self.L = L
            self.variance = variance
            self.K = K
            self.device = device
            self.SC_1 = SC(in_size = 1, out_size = self.L, K = self.K, variance=self.variance, device = self.device )
            self.SC_2 = SC(in_size = self.L, out_size = self.L, K = self.K, variance=self.variance, device = self.device)
            self.SC_3 = SC(in_size = self.L, out_size = 1, K = self.K, variance=self.variance, device = self.device)

      def forward( self, LdStack, LuStack, x ):
             out1_1 = self.SC_1( LdStack, LuStack, x )
             out1_2 = self.SC_2( LdStack, LuStack, nn.LeakyReLU()(out1_1) )
             out1_3 = self.SC_3( LdStack, LuStack, nn.LeakyReLU()(out1_2) )
             return out1_3