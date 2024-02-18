import torch.nn as nn
import torch

#LAYER
class SC( nn.Module ):
      def __init__(self, in_size, out_size, K, variance = 1.0, device = "cpu"):
            super().__init__()
            self.device = device
            self.in_size = in_size
            self.out_size = out_size
            self.K = K
            self.coeff_P = nn.parameter.Parameter(variance*torch.randn((self.in_size, self.out_size, self.K), device = device))
            self.coeff_Q = nn.parameter.Parameter(variance*torch.randn((self.in_size, self.out_size, self.K), device = device))


      def forward(self, LdStack, LuStack, x):
            # here we need to write the assemble of the polynomial MAY BE NOT OPTIMAL
            return sum([ LdStack[:, :, i] @ x @ self.coeff_P[:, :, i] for i in range( self.K ) ]) + sum([ LuStack[:, :, i] @ x @ self.coeff_Q[:, :, i] for i in range( self.K ) ])

#NETWORK
class SCCGNN( nn.Module ):
      def __init__(self, K = 5, L = 3, variance = 0.01 ):
            super().__init__()
            self.L = L
            self.variance = variance
            self.K = K

            self.SC_1 = SC(in_size = 1, out_size = self.L, K = self.K, variance=self.variance)
            self.SC_2 = SC(in_size = self.L, out_size = self.L, K = self.K, variance=self.variance)
            self.SC_3 = SC(in_size = self.L, out_size = 1, K = self.K, variance=self.variance)

      def forward( self, LdStack, LuStack, x ):
             out1_1 = self.SC_1( LdStack, LuStack, x )
             out1_2 = self.SC_2( LdStack, LuStack, nn.LeakyReLU()(out1_1) )
             out1_3 = self.SC_3( LdStack, LuStack, nn.LeakyReLU()(out1_2) )
             return out1_3