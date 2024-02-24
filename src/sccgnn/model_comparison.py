# %%
# package testing framework 
from importlib import reload
from datetime import datetime
from datetime import timedelta
from time import time

from tqdm import trange


import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.")

import utils
utils = reload(utils)
from utils import *

import triangulation
triangulation = reload(triangulation)
from triangulation import *

import network
network = reload(network)
from network import *

import output
output = reload(output)
from output import *

import training
training = reload(training)
from training import *


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
#device = torch.device("mps")
print("Running on: ", device)




N = 100  # size of simplicial complex

# IMPORTANT: stable / non-stable flag inside `generateTriangulation`
Ld, Lu, B1w, B2w, W0inv, W1, W2, edges, trians, n, points, edg2Trig, trig2Edge = generateTriangulation( N, instable = False, device = device )


if device == torch.device("mps"):
      b1b1t_inv = torch.tensor( np.linalg.pinv( (B1w @ B1w.T).numpy(force = True) ), device = device, dtype = torch.float32) 
      b2tb2_inv = torch.tensor( np.linalg.pinv( (B2w.T @ B2w).numpy(force = True) ), device = device, dtype = torch.float32) 
else:
      b1b1t_inv = torch.linalg.pinv( (B1w @ B1w.T).to_dense() ).to_sparse()
      b2tb2_inv = torch.linalg.pinv( (B2w.T @ B2w).to_dense() ).to_sparse()

topeig = get_top_eig(Lu, Ld, device = device)
Ld = 1. / ( 1. * topeig ) * Ld #normalistion !!! MAY AFFECT CONVERGENCE / RATE OF CONVERGENCE
Lu = 1. / ( 1. * topeig ) * Lu
m1 = Ld.shape[0]

#%%
# define the size of each filter
K = 5
LdStack, LuStack = build_stacks( m1, Ld, Lu, K, device = device )

print( condPlus( Ld, device = device ), condPlus( Lu, device = device  ), condPlus( Ld + Lu, device = device ) ) # check: if unstable, big numbers, if stable -- "decent"
print("--------------------------------------")
print()

# generate target output
var = 0.01
y = output_generator( m1, W1, edg2Trig, var = var, device = device )
gr, curl, harm =  P1( y, B1w, b1b1t_inv) , P2( y, B2w, b2tb2_inv) , y - P1( y, B1w, b1b1t_inv) -  P2( y, B2w, b2tb2_inv) 
y = gr / torch.norm(gr) + curl / torch.norm(curl) + harm / torch.norm(harm)
y_real = y.clone()

# define missing entries
dropRate = 0.2  
valRate = 0.2

ind, val_ind, saved = get_missing( m1, dropRate, valRate, device = device )
fillerValue = torch.median( y[ saved ] )
y[ ind ] = fillerValue
print( "naive guess level: ", MAPE(y, y_real, ind) )
print()



#%%
rep_best_loss = []
rep_best_val = []
L = 10
print("K: ", K, "  L: ", L, " classical")
for i in range(3):
      α1 = 15.0 # parameters for smart loss. Will not affect classical
      α2 = 30.0

      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

      verbose = False
      variance = 3.0
      p = 0.9
      MAX_EPOCH = 2500
      is_classical = True

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device ) #initiate model
      learning_rate = 0.02 # I don't fucking know how much is better
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      if is_classical:
            criterion = classicalLoss(saved)
      else:
            criterion = componentLoss( saved, B1w, B2w, b1b1t_inv, b2tb2_inv, α1, α2 )      

      my_loss = [] # we store losses per epoch here
      my_acc = [] # we store accuracies per epoch here
      my_val_acc = []

      y_norm = [ torch.linalg.norm( P1( y - y_real, B1w, b1b1t_inv ) ).item() ]
      z_norm = [ torch.linalg.norm( P2( y - y_real, B2w, b2tb2_inv ) ).item() ]
      h_norm = [ torch.linalg.norm( y - y_real - P1( y - y_real, B1w, b1b1t_inv ) -  P2( y - y_real, B2w, b2tb2_inv ) ).item() ]

      best_loss = 1_000_000.
      #best_acc = -1000.0
      best_acc = 1_000_000.0

      begin_time = time()

      for epoch in trange(MAX_EPOCH, desc="EPOCHS"):
            model.train(True)

            data = form_epoch_data( y, y_real, saved, fillerValue, p, device = device)
            running_loss = 0.
            last_loss = 0.
            for i, data_point in enumerate( data ):
                  inputs, labels = data_point
                  inputs = inputs.reshape(-1, 1)
                  labels = labels.reshape(-1, 1)
                  optimizer.zero_grad()

                  outputs = model(LdStack, LuStack, inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.item()
                  if i % len(data) == len(data) - 1 :
                              last_loss = running_loss / len(data)
                              running_loss = 0.

            my_loss.append( last_loss )
            model.eval()
            out = model(LdStack, LuStack, y.reshape(-1, 1) )
            #my_acc.append( my_accuracy( out, y_real.reshape(-1, 1), ind ) )
            my_acc.append( MAPE( out, y_real.reshape(-1, 1), ind ) )
            y_norm.append( torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item() )
            z_norm.append( torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item() )
            h_norm.append( torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item() )

            #val_acc = my_val_accuracy( out, y_real.reshape(-1, 1), val_ind )
            val_acc = MAPE( out, y_real.reshape(-1, 1), val_ind )
            my_val_acc.append( val_acc )

            if val_acc < best_acc and epoch > 0.9 * MAX_EPOCH:
            #best_loss = my_loss[-1]
                  best_acc = val_acc
                  if is_classical:
                        model_path_val = '../../model_dump/BVA_model_{}_N{}_K{}_L{}_classical'.format(timestamp, N, K, L)  # we may want to change the name...
                  else:
                        model_path_val = '../../model_dump/BVA_model_{}_N{}_K{}_L{}_smart_{}|{}'.format(timestamp, N, K, L, α1, α2)  # we may want to change the name...
                  torch.save(model.state_dict(), model_path_val)
      
            if best_loss > my_loss[-1]: #and epoch > 0.5 * MAX_EPOCH:
                  best_loss = my_loss[-1]
                  #best_acc = val_acc
                  if is_classical:
                        model_path_loss = '../../model_dump/BL_model_{}_N{}_K{}_L{}_classical'.format(timestamp, N, K, L)  # we may want to change the name...
                  else:
                        model_path_loss = '../../model_dump/BL_model_{}_N{}_K{}_L{}_smart_{}|{}'.format(timestamp, N, K, L, α1, α2)  # we may want to change the name...
                  torch.save(model.state_dict(), model_path_loss)


            if verbose:
                  if epoch % 20 == 0:
                        print(' epoch{}  ||   loss: {}, accuracy: {}'.format(epoch, my_loss[-1], my_acc[-1]))

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device )
      model.load_state_dict( torch.load(model_path_val) )
      model.eval()
      out = model(LdStack, LuStack, y.reshape(-1, 1) )
      fin_loss = criterion( out, y_real)
      #fin_acc =  my_accuracy( out, y_real.reshape(-1, 1), ind )
      fin_acc =  MAPE( out, y_real.reshape(-1, 1), ind )
      fin_y_norm = torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item()
      fin_z_norm = torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item()
      fin_h_norm = torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item()

      #print()
      #print("--------------------------------------")
      #print("BEST VALIDATION MODEL:")
      #print("acc: ", round(fin_acc.item(), 4), #round(fin_loss.item(),4), round(fin_y_norm,4), round(fin_z_norm,4), round(fin_h_norm,4), 
           # "  acc/best test: ", #round(fin_acc.item()/max(my_acc).item(),4), "  best val acc/best test: ", round(best_acc.item()/max(my_acc).item(),4) )
      rep_best_val.append(fin_acc.item()) 

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device )
      model.load_state_dict( torch.load(model_path_loss) )
      model.eval()
      out = model(LdStack, LuStack, y.reshape(-1, 1) )
      fin_loss = criterion( out, y_real)
      #fin_acc =  my_accuracy( out, y_real.reshape(-1, 1), ind )
      fin_acc =  MAPE( out, y_real.reshape(-1, 1), ind )
      fin_y_norm = torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item()
      fin_z_norm = torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item()
      fin_h_norm = torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item()
      rep_best_loss.append(fin_acc.item()) 

      #print()
      #print("BEST LOSS MODEL:")
      #print("acc: ", round(fin_acc.item(), 4), #round(fin_loss.item(),4), round(fin_y_norm,4), round(fin_z_norm,4), round(fin_h_norm,4), 
       #     "  acc/best test: ", round(fin_acc.item()/max(my_acc).item(),4), "  best val acc/best test: ", round(best_acc.item()/max(my_acc).item(),4) )
      #print( "time: ", str(timedelta(seconds = time()-begin_time)) )
      #print("--------------------------------------")
      #print()
print(" loss: ", rep_best_loss)
print(" val: ", rep_best_val)
print("--------------------------------------")
print()

#%%
rep_best_loss = []
rep_best_val = []
L = 10
print("K: ", K, "  L: ", L, " smart 10/30")
for i in range(3):
      α1 = 10.0 # parameters for smart loss. Will not affect classical
      α2 = 30.0

      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

      verbose = False
      variance = 3.0
      p = 0.9
      MAX_EPOCH = 2500
      is_classical = False

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device ) #initiate model
      learning_rate = 0.02 # I don't fucking know how much is better
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      if is_classical:
            criterion = classicalLoss(saved)
      else:
            criterion = componentLoss( saved, B1w, B2w, b1b1t_inv, b2tb2_inv, α1, α2 )      

      my_loss = [] # we store losses per epoch here
      my_acc = [] # we store accuracies per epoch here
      my_val_acc = []

      y_norm = [ torch.linalg.norm( P1( y - y_real, B1w, b1b1t_inv ) ).item() ]
      z_norm = [ torch.linalg.norm( P2( y - y_real, B2w, b2tb2_inv ) ).item() ]
      h_norm = [ torch.linalg.norm( y - y_real - P1( y - y_real, B1w, b1b1t_inv ) -  P2( y - y_real, B2w, b2tb2_inv ) ).item() ]

      best_loss = 1_000_000.
      #best_acc = -1000.0
      best_acc = 1_000_000.0

      begin_time = time()

      for epoch in trange(MAX_EPOCH, desc="EPOCHS"):
            model.train(True)

            data = form_epoch_data( y, y_real, saved, fillerValue, p, device = device)
            running_loss = 0.
            last_loss = 0.
            for i, data_point in enumerate( data ):
                  inputs, labels = data_point
                  inputs = inputs.reshape(-1, 1)
                  labels = labels.reshape(-1, 1)
                  optimizer.zero_grad()

                  outputs = model(LdStack, LuStack, inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.item()
                  if i % len(data) == len(data) - 1 :
                              last_loss = running_loss / len(data)
                              running_loss = 0.

            my_loss.append( last_loss )
            model.eval()
            out = model(LdStack, LuStack, y.reshape(-1, 1) )
            #my_acc.append( my_accuracy( out, y_real.reshape(-1, 1), ind ) )
            my_acc.append( MAPE( out, y_real.reshape(-1, 1), ind ) )
            y_norm.append( torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item() )
            z_norm.append( torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item() )
            h_norm.append( torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item() )

            #val_acc = my_val_accuracy( out, y_real.reshape(-1, 1), val_ind )
            val_acc = MAPE( out, y_real.reshape(-1, 1), val_ind )
            my_val_acc.append( val_acc )

            if val_acc < best_acc and epoch > 0.9 * MAX_EPOCH:
            #best_loss = my_loss[-1]
                  best_acc = val_acc
                  if is_classical:
                        model_path_val = '../../model_dump/BVA_model_{}_N{}_K{}_L{}_classical'.format(timestamp, N, K, L)  # we may want to change the name...
                  else:
                        model_path_val = '../../model_dump/BVA_model_{}_N{}_K{}_L{}_smart_{}|{}'.format(timestamp, N, K, L, α1, α2)  # we may want to change the name...
                  torch.save(model.state_dict(), model_path_val)
      
            if best_loss > my_loss[-1]: #and epoch > 0.5 * MAX_EPOCH:
                  best_loss = my_loss[-1]
                  #best_acc = val_acc
                  if is_classical:
                        model_path_loss = '../../model_dump/BL_model_{}_N{}_K{}_L{}_classical'.format(timestamp, N, K, L)  # we may want to change the name...
                  else:
                        model_path_loss = '../../model_dump/BL_model_{}_N{}_K{}_L{}_smart_{}|{}'.format(timestamp, N, K, L, α1, α2)  # we may want to change the name...
                  torch.save(model.state_dict(), model_path_loss)


            if verbose:
                  if epoch % 20 == 0:
                        print(' epoch{}  ||   loss: {}, accuracy: {}'.format(epoch, my_loss[-1], my_acc[-1]))

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device )
      model.load_state_dict( torch.load(model_path_val) )
      model.eval()
      out = model(LdStack, LuStack, y.reshape(-1, 1) )
      fin_loss = criterion( out, y_real)
      #fin_acc =  my_accuracy( out, y_real.reshape(-1, 1), ind )
      fin_acc =  MAPE( out, y_real.reshape(-1, 1), ind )
      fin_y_norm = torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item()
      fin_z_norm = torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item()
      fin_h_norm = torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item()

      #print()
      #print("--------------------------------------")
      #print("BEST VALIDATION MODEL:")
      #print("acc: ", round(fin_acc.item(), 4), #round(fin_loss.item(),4), round(fin_y_norm,4), round(fin_z_norm,4), round(fin_h_norm,4), 
           # "  acc/best test: ", #round(fin_acc.item()/max(my_acc).item(),4), "  best val acc/best test: ", round(best_acc.item()/max(my_acc).item(),4) )
      rep_best_val.append(fin_acc.item()) 

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device )
      model.load_state_dict( torch.load(model_path_loss) )
      model.eval()
      out = model(LdStack, LuStack, y.reshape(-1, 1) )
      fin_loss = criterion( out, y_real)
      #fin_acc =  my_accuracy( out, y_real.reshape(-1, 1), ind )
      fin_acc =  MAPE( out, y_real.reshape(-1, 1), ind )
      fin_y_norm = torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item()
      fin_z_norm = torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item()
      fin_h_norm = torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item()
      rep_best_loss.append(fin_acc.item()) 

      #print()
      #print("BEST LOSS MODEL:")
      #print("acc: ", round(fin_acc.item(), 4), #round(fin_loss.item(),4), round(fin_y_norm,4), round(fin_z_norm,4), round(fin_h_norm,4), 
       #     "  acc/best test: ", round(fin_acc.item()/max(my_acc).item(),4), "  best val acc/best test: ", round(best_acc.item()/max(my_acc).item(),4) )
      #print( "time: ", str(timedelta(seconds = time()-begin_time)) )
      #print("--------------------------------------")
      #print()
print(" loss: ", rep_best_loss)
print(" val: ", rep_best_val)
print("--------------------------------------")
print()

#%%
rep_best_loss = []
rep_best_val = []
L = 10
print("K: ", K, "  L: ", L, " smart 15/30")
for i in range(3):
      α1 = 15.0 # parameters for smart loss. Will not affect classical
      α2 = 30.0

      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

      verbose = False
      variance = 3.0
      p = 0.9
      MAX_EPOCH = 2500
      is_classical = False

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device ) #initiate model
      learning_rate = 0.02 # I don't fucking know how much is better
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      if is_classical:
            criterion = classicalLoss(saved)
      else:
            criterion = componentLoss( saved, B1w, B2w, b1b1t_inv, b2tb2_inv, α1, α2 )      

      my_loss = [] # we store losses per epoch here
      my_acc = [] # we store accuracies per epoch here
      my_val_acc = []

      y_norm = [ torch.linalg.norm( P1( y - y_real, B1w, b1b1t_inv ) ).item() ]
      z_norm = [ torch.linalg.norm( P2( y - y_real, B2w, b2tb2_inv ) ).item() ]
      h_norm = [ torch.linalg.norm( y - y_real - P1( y - y_real, B1w, b1b1t_inv ) -  P2( y - y_real, B2w, b2tb2_inv ) ).item() ]

      best_loss = 1_000_000.
      #best_acc = -1000.0
      best_acc = 1_000_000.0

      begin_time = time()

      for epoch in trange(MAX_EPOCH, desc="EPOCHS"):
            model.train(True)

            data = form_epoch_data( y, y_real, saved, fillerValue, p, device = device)
            running_loss = 0.
            last_loss = 0.
            for i, data_point in enumerate( data ):
                  inputs, labels = data_point
                  inputs = inputs.reshape(-1, 1)
                  labels = labels.reshape(-1, 1)
                  optimizer.zero_grad()

                  outputs = model(LdStack, LuStack, inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.item()
                  if i % len(data) == len(data) - 1 :
                              last_loss = running_loss / len(data)
                              running_loss = 0.

            my_loss.append( last_loss )
            model.eval()
            out = model(LdStack, LuStack, y.reshape(-1, 1) )
            #my_acc.append( my_accuracy( out, y_real.reshape(-1, 1), ind ) )
            my_acc.append( MAPE( out, y_real.reshape(-1, 1), ind ) )
            y_norm.append( torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item() )
            z_norm.append( torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item() )
            h_norm.append( torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item() )

            #val_acc = my_val_accuracy( out, y_real.reshape(-1, 1), val_ind )
            val_acc = MAPE( out, y_real.reshape(-1, 1), val_ind )
            my_val_acc.append( val_acc )

            if val_acc < best_acc and epoch > 0.9 * MAX_EPOCH:
            #best_loss = my_loss[-1]
                  best_acc = val_acc
                  if is_classical:
                        model_path_val = '../../model_dump/BVA_model_{}_N{}_K{}_L{}_classical'.format(timestamp, N, K, L)  # we may want to change the name...
                  else:
                        model_path_val = '../../model_dump/BVA_model_{}_N{}_K{}_L{}_smart_{}|{}'.format(timestamp, N, K, L, α1, α2)  # we may want to change the name...
                  torch.save(model.state_dict(), model_path_val)
      
            if best_loss > my_loss[-1]: #and epoch > 0.5 * MAX_EPOCH:
                  best_loss = my_loss[-1]
                  #best_acc = val_acc
                  if is_classical:
                        model_path_loss = '../../model_dump/BL_model_{}_N{}_K{}_L{}_classical'.format(timestamp, N, K, L)  # we may want to change the name...
                  else:
                        model_path_loss = '../../model_dump/BL_model_{}_N{}_K{}_L{}_smart_{}|{}'.format(timestamp, N, K, L, α1, α2)  # we may want to change the name...
                  torch.save(model.state_dict(), model_path_loss)


            if verbose:
                  if epoch % 20 == 0:
                        print(' epoch{}  ||   loss: {}, accuracy: {}'.format(epoch, my_loss[-1], my_acc[-1]))

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device )
      model.load_state_dict( torch.load(model_path_val) )
      model.eval()
      out = model(LdStack, LuStack, y.reshape(-1, 1) )
      fin_loss = criterion( out, y_real)
      #fin_acc =  my_accuracy( out, y_real.reshape(-1, 1), ind )
      fin_acc =  MAPE( out, y_real.reshape(-1, 1), ind )
      fin_y_norm = torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item()
      fin_z_norm = torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item()
      fin_h_norm = torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item()

      #print()
      #print("--------------------------------------")
      #print("BEST VALIDATION MODEL:")
      #print("acc: ", round(fin_acc.item(), 4), #round(fin_loss.item(),4), round(fin_y_norm,4), round(fin_z_norm,4), round(fin_h_norm,4), 
           # "  acc/best test: ", #round(fin_acc.item()/max(my_acc).item(),4), "  best val acc/best test: ", round(best_acc.item()/max(my_acc).item(),4) )
      rep_best_val.append(fin_acc.item()) 

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device )
      model.load_state_dict( torch.load(model_path_loss) )
      model.eval()
      out = model(LdStack, LuStack, y.reshape(-1, 1) )
      fin_loss = criterion( out, y_real)
      #fin_acc =  my_accuracy( out, y_real.reshape(-1, 1), ind )
      fin_acc =  MAPE( out, y_real.reshape(-1, 1), ind )
      fin_y_norm = torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item()
      fin_z_norm = torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item()
      fin_h_norm = torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item()
      rep_best_loss.append(fin_acc.item()) 

      #print()
      #print("BEST LOSS MODEL:")
      #print("acc: ", round(fin_acc.item(), 4), #round(fin_loss.item(),4), round(fin_y_norm,4), round(fin_z_norm,4), round(fin_h_norm,4), 
       #     "  acc/best test: ", round(fin_acc.item()/max(my_acc).item(),4), "  best val acc/best test: ", round(best_acc.item()/max(my_acc).item(),4) )
      #print( "time: ", str(timedelta(seconds = time()-begin_time)) )
      #print("--------------------------------------")
      #print()
print(" loss: ", rep_best_loss)
print(" val: ", rep_best_val)
print("--------------------------------------")
print()


#%%
rep_best_loss = []
rep_best_val = []
L = 10
print("K: ", K, "  L: ", L, " smart 10/50")
for i in range(3):
      α1 = 10.0 # parameters for smart loss. Will not affect classical
      α2 = 50.0

      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

      verbose = False
      variance = 3.0
      p = 0.9
      MAX_EPOCH = 2500
      is_classical = False

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device ) #initiate model
      learning_rate = 0.02 # I don't fucking know how much is better
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      if is_classical:
            criterion = classicalLoss(saved)
      else:
            criterion = componentLoss( saved, B1w, B2w, b1b1t_inv, b2tb2_inv, α1, α2 )      

      my_loss = [] # we store losses per epoch here
      my_acc = [] # we store accuracies per epoch here
      my_val_acc = []

      y_norm = [ torch.linalg.norm( P1( y - y_real, B1w, b1b1t_inv ) ).item() ]
      z_norm = [ torch.linalg.norm( P2( y - y_real, B2w, b2tb2_inv ) ).item() ]
      h_norm = [ torch.linalg.norm( y - y_real - P1( y - y_real, B1w, b1b1t_inv ) -  P2( y - y_real, B2w, b2tb2_inv ) ).item() ]

      best_loss = 1_000_000.
      #best_acc = -1000.0
      best_acc = 1_000_000.0

      begin_time = time()

      for epoch in trange(MAX_EPOCH, desc="EPOCHS"):
            model.train(True)

            data = form_epoch_data( y, y_real, saved, fillerValue, p, device = device)
            running_loss = 0.
            last_loss = 0.
            for i, data_point in enumerate( data ):
                  inputs, labels = data_point
                  inputs = inputs.reshape(-1, 1)
                  labels = labels.reshape(-1, 1)
                  optimizer.zero_grad()

                  outputs = model(LdStack, LuStack, inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.item()
                  if i % len(data) == len(data) - 1 :
                              last_loss = running_loss / len(data)
                              running_loss = 0.

            my_loss.append( last_loss )
            model.eval()
            out = model(LdStack, LuStack, y.reshape(-1, 1) )
            #my_acc.append( my_accuracy( out, y_real.reshape(-1, 1), ind ) )
            my_acc.append( MAPE( out, y_real.reshape(-1, 1), ind ) )
            y_norm.append( torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item() )
            z_norm.append( torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item() )
            h_norm.append( torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item() )

            #val_acc = my_val_accuracy( out, y_real.reshape(-1, 1), val_ind )
            val_acc = MAPE( out, y_real.reshape(-1, 1), val_ind )
            my_val_acc.append( val_acc )

            if val_acc < best_acc and epoch > 0.9 * MAX_EPOCH:
            #best_loss = my_loss[-1]
                  best_acc = val_acc
                  if is_classical:
                        model_path_val = '../../model_dump/BVA_model_{}_N{}_K{}_L{}_classical'.format(timestamp, N, K, L)  # we may want to change the name...
                  else:
                        model_path_val = '../../model_dump/BVA_model_{}_N{}_K{}_L{}_smart_{}|{}'.format(timestamp, N, K, L, α1, α2)  # we may want to change the name...
                  torch.save(model.state_dict(), model_path_val)
      
            if best_loss > my_loss[-1]: #and epoch > 0.5 * MAX_EPOCH:
                  best_loss = my_loss[-1]
                  #best_acc = val_acc
                  if is_classical:
                        model_path_loss = '../../model_dump/BL_model_{}_N{}_K{}_L{}_classical'.format(timestamp, N, K, L)  # we may want to change the name...
                  else:
                        model_path_loss = '../../model_dump/BL_model_{}_N{}_K{}_L{}_smart_{}|{}'.format(timestamp, N, K, L, α1, α2)  # we may want to change the name...
                  torch.save(model.state_dict(), model_path_loss)


            if verbose:
                  if epoch % 20 == 0:
                        print(' epoch{}  ||   loss: {}, accuracy: {}'.format(epoch, my_loss[-1], my_acc[-1]))

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device )
      model.load_state_dict( torch.load(model_path_val) )
      model.eval()
      out = model(LdStack, LuStack, y.reshape(-1, 1) )
      fin_loss = criterion( out, y_real)
      #fin_acc =  my_accuracy( out, y_real.reshape(-1, 1), ind )
      fin_acc =  MAPE( out, y_real.reshape(-1, 1), ind )
      fin_y_norm = torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item()
      fin_z_norm = torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item()
      fin_h_norm = torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item()

      #print()
      #print("--------------------------------------")
      #print("BEST VALIDATION MODEL:")
      #print("acc: ", round(fin_acc.item(), 4), #round(fin_loss.item(),4), round(fin_y_norm,4), round(fin_z_norm,4), round(fin_h_norm,4), 
           # "  acc/best test: ", #round(fin_acc.item()/max(my_acc).item(),4), "  best val acc/best test: ", round(best_acc.item()/max(my_acc).item(),4) )
      rep_best_val.append(fin_acc.item()) 

      model = SCCGNN( K = K, L = L, variance = 1.0, device = device )
      model.load_state_dict( torch.load(model_path_loss) )
      model.eval()
      out = model(LdStack, LuStack, y.reshape(-1, 1) )
      fin_loss = criterion( out, y_real)
      #fin_acc =  my_accuracy( out, y_real.reshape(-1, 1), ind )
      fin_acc =  MAPE( out, y_real.reshape(-1, 1), ind )
      fin_y_norm = torch.linalg.norm( P1( out - y_real, B1w, b1b1t_inv ) ).item()
      fin_z_norm = torch.linalg.norm( P2( out - y_real, B2w, b2tb2_inv ) ).item()
      fin_h_norm = torch.linalg.norm( out - y_real - P1( out - y_real, B1w, b1b1t_inv ) - P2( out - y_real, B2w, b2tb2_inv ) ).item()
      rep_best_loss.append(fin_acc.item()) 

      #print()
      #print("BEST LOSS MODEL:")
      #print("acc: ", round(fin_acc.item(), 4), #round(fin_loss.item(),4), round(fin_y_norm,4), round(fin_z_norm,4), round(fin_h_norm,4), 
       #     "  acc/best test: ", round(fin_acc.item()/max(my_acc).item(),4), "  best val acc/best test: ", round(best_acc.item()/max(my_acc).item(),4) )
      #print( "time: ", str(timedelta(seconds = time()-begin_time)) )
      #print("--------------------------------------")
      #print()
print(" loss: ", rep_best_loss)
print(" val: ", rep_best_val)
print("--------------------------------------")
print()



# %%
