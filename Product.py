#%% load packages
import torch 
import scipy.io
from torch.utils.tensorboard import SummaryWriter
import os
import time
from utils_Product import *
import sympy as sym
import numpy as np
import scipy.stats

torch.set_default_dtype(torch.float64)

## fix random seeds
torch.manual_seed(0)

## log training statistics
# Writer will output to ./runs/ directory by default
writer = SummaryWriter("Concrete_Prod")

## select device
if torch.cuda.is_available():
    cuda_tag = "cuda:0"
    device = torch.device(cuda_tag)  # ".to(device)" should be added to all models and inputs
    print("Running on " + cuda_tag)
else:
    device = torch.device("cpu")
    print("Running on the CPU")

## data preprocessing
# load data
data = scipy.io.loadmat(os.path.join(os.path.dirname(os.getcwd()), 'precracked.mat')) # a dictionary
data_torch = {k:torch.from_numpy(v).to(device).double() for _, (k, v) in enumerate(data.items()) if k!='__header__' and k!='__version__' and k!='__globals__'} 
data_normalized_coeff = {k: torch.linalg.norm(torch.ones_like(v))/torch.linalg.norm(v) for _, (k, v) in enumerate(data_torch.items())}

# split data based on sources
dataset_ind = [15, 43, 48, 50, 58, 77, 89, 109, 127, 142, 148, 156, 161, 209, 228] # precracked

data_all = [{k:v[:dataset_ind[0]].reshape((-1, 1)) for _, (k, v) in enumerate(data_torch.items())}]
for i in range(len(dataset_ind)-1):
    data_all.append({k:v[dataset_ind[i]:dataset_ind[i+1]].reshape((-1, 1)) for _, (k, v) in enumerate(data_torch.items())})
data_all.append({k:v[dataset_ind[i+1]:].reshape((-1, 1)) for _, (k, v) in enumerate(data_torch.items())})

#%%
bragging = 100
data_normalized_tr_not_bragging_all = []
data_normalized_val_not_bragging_all = []
data_normalized_tr_all = []
data_normalized_all = []
for data in data_all:
    p_c_flag = True 
    data_normalized_tr_not_bragging_g_2200, data_normalized_val_not_bragging_g_2200, data_normalized_tr_g_2200, data_normalized_g_2200 = preprocess(data, p_c_flag,
     device, data_normalized_coeff,
     bragging)

    p_c_flag = False 
    data_normalized_tr_not_bragging_leq_2200, data_normalized_val_not_bragging_leq_2200, data_normalized_tr_leq_2200, data_normalized_leq_2200 = preprocess(data,
     p_c_flag,
     device, data_normalized_coeff,
     bragging)

    if data_normalized_tr_not_bragging_g_2200 == None and data_normalized_tr_not_bragging_leq_2200 == None:
        continue
    elif data_normalized_tr_not_bragging_g_2200 == None:
        data_normalized_tr_not_bragging = data_normalized_tr_not_bragging_leq_2200
        data_normalized_val_not_bragging = data_normalized_val_not_bragging_leq_2200
        data_normalized_tr = data_normalized_tr_leq_2200
        data_normalized = data_normalized_leq_2200
    elif data_normalized_tr_not_bragging_leq_2200 == None:
        data_normalized_tr_not_bragging = data_normalized_tr_not_bragging_g_2200
        data_normalized_val_not_bragging = data_normalized_val_not_bragging_g_2200
        data_normalized_tr = data_normalized_tr_g_2200
        data_normalized = data_normalized_g_2200
    else:
        data_normalized_tr_not_bragging = {k: torch.cat([v, data_normalized_tr_not_bragging_g_2200[k]], 0) for _, (k, v) in enumerate(data_normalized_tr_not_bragging_leq_2200.items())} 
        data_normalized_val_not_bragging = {k: torch.cat([v, data_normalized_val_not_bragging_g_2200[k]], 0) for _, (k, v) in enumerate(data_normalized_val_not_bragging_leq_2200.items())} 
        data_normalized_tr = {k: torch.cat([v, data_normalized_tr_g_2200[k]], 1) for _, (k, v) in enumerate(data_normalized_tr_leq_2200.items())} 
        data_normalized = {k: torch.cat([v, data_normalized_g_2200[k]], 0) for _, (k, v) in enumerate(data_normalized_leq_2200.items())} 

    data_normalized_tr_not_bragging_all.append(data_normalized_tr_not_bragging)
    data_normalized_val_not_bragging_all.append(data_normalized_val_not_bragging)
    data_normalized_tr_all.append(data_normalized_tr)
    data_normalized_all.append(data_normalized)

#%%

## dictionary paras
free_coeff = torch.zeros((bragging, 6, 1), dtype=torch.float64,device=device, requires_grad=True)
sparse_coeff = torch.zeros((bragging, 102, 1), dtype=torch.float64,device=device, requires_grad=True)

data_loss_coeff_core = torch.zeros((len(data_normalized_val_not_bragging_all), 1), dtype=torch.float64,device=device, requires_grad=True) 
 
#%%
""" training """
start_time = time.time()

epoch = 10000 # 10000
lhs_pred_tr, lhs_ref_tr, lhs_pred_val, lhs_ref_val = train(free_coeff, sparse_coeff, data_loss_coeff_core, epoch, data_normalized_tr_all, 
data_normalized_val_not_bragging_all, writer)
elapsed = time.time() - start_time  
writer.add_text('Time', 'Training time:' + str(elapsed))

# evaluate model and report metrics
with torch.no_grad():
    err_tr = torch.linalg.norm(lhs_ref_tr.flatten() - lhs_pred_tr.flatten())/torch.linalg.norm(lhs_ref_tr.flatten())*100
    writer.add_text('lhs_err_tr', str(err_tr.item()))
    err_val = torch.linalg.norm(lhs_ref_val.flatten() - lhs_pred_val.flatten())/torch.linalg.norm(lhs_ref_val.flatten())*100
    writer.add_text('lhs_err_val', str(err_val.item()))

# save data and mdl
scipy.io.savemat('pred_lhs.mat',{'lhs_pred_tr':lhs_pred_tr.detach().cpu().numpy(), 'lhs_ref_tr':lhs_ref_tr.cpu().numpy(),
'lhs_pred_val':lhs_pred_val.detach().cpu().numpy(), 'lhs_ref_val':lhs_ref_val.cpu().numpy(),
})

torch.save(free_coeff, 'free_coeff.pt')
torch.save(sparse_coeff, 'sparse_coeff.pt')

""" prune coeff  """
""" run on the training machine """
# find the medians of ensembles
sparse_coeff_med, _ = torch.median(sparse_coeff, dim=0, keepdim=True)

# a networkwise grid search from 0 to 100 percentile
threshold_list = []
sparse_coeff_all = sparse_coeff_med.detach().to('cpu').numpy().flatten()
threshold_list = [np.percentile(np.abs(sparse_coeff_all), My_Percentile) for My_Percentile in range(0,101)]

# hard-threshold sparse coeff
Eq_err_list = []
No_Nonzero_list = []
i_list = [] 
with torch.no_grad():
    for i, threshold in enumerate(threshold_list):
        sparse_coeff_med, _ = HardThreshold_Eq(sparse_coeff_med, threshold)

        lhs_pred_tr_list = []
        lhs_ref_tr_list = []
        for data_normalized_tr in data_normalized_tr_all:
            _, lhs_pred_tr, lhs_ref_tr = forward_pass(data_normalized_tr, free_coeff, sparse_coeff_med, 'Median')  
            lhs_pred_tr_list.append(lhs_pred_tr)
            lhs_ref_tr_list.append(lhs_ref_tr)
        lhs_pred_tr = torch.cat(lhs_pred_tr_list, 1)
        lhs_ref_tr = torch.cat(lhs_ref_tr_list, 1)

        err_tr = torch.linalg.norm(lhs_ref_tr.flatten() - lhs_pred_tr.flatten())/torch.linalg.norm(lhs_ref_tr.flatten())*100
        Eq_err_list.append(err_tr.item())
        No_Nonzero = CountNonZeros_Eq(sparse_coeff_med)
        No_Nonzero_list.append(No_Nonzero)
        i_list.append(i)

    scipy.io.savemat('Eq_prune.mat', {'Eq_err':np.stack(Eq_err_list), 'No_Nonzero_all':np.stack(No_Nonzero_list), 'i_all':np.stack(i_list)})

#%%
""" print eq.  """
""" run after pruning. """
# load trained mdl
free_coeff = torch.load('free_coeff.pt', map_location=device)
sparse_coeff = torch.load('sparse_coeff.pt', map_location=device)

# find the medians of ensembles
sparse_coeff_med, _ = torch.median(sparse_coeff, dim=0, keepdim=True)

# find std
sparse_coeff_std = torch.std(sparse_coeff, dim=0, keepdim=True)

# a networkwise grid search from 0 to 100 percentile
threshold_list = []
sparse_coeff_all = sparse_coeff_med.detach().to('cpu').numpy().flatten()
threshold_list = [np.percentile(np.abs(sparse_coeff_all), My_Percentile) for My_Percentile in range(0,101)]

# hard-threshold sparse coeff
best_ind = 94
sparse_coeff_med, sparse_ind = HardThreshold_Eq(sparse_coeff_med, threshold_list[best_ind])

sparse_coeff_std[sparse_ind] = 0

with torch.no_grad():
    # create normalized symbols
    symbol_normalized = {k: sym.symbols(k)*(v.item()) for _, (k, v) in enumerate(data_normalized_coeff.items())}
    
    # standardized regressors to compare rank
    # symbol_normalized = {k: sym.symbols(k) for _, (k, v) in enumerate(data_normalized_coeff.items())}

    # free bases
    free_bases_sym = [symbol_normalized['c_v'], symbol_normalized['f_c'], symbol_normalized['c_l'], symbol_normalized['d_a_d']*symbol_normalized['c_v'],
        symbol_normalized['d_a_d']*symbol_normalized['f_c'], symbol_normalized['d_a_d']*symbol_normalized['c_l']]

    # sparse bases
    sparse_bases_sym = BuildLib([1, sym.sqrt(symbol_normalized['d_t_d']), sym.sqrt(symbol_normalized['b_d']), sym.sqrt(symbol_normalized['w_d'])], 4)
    sparse_bases_c_v_sym = [basis*free_bases_sym[0] for basis in sparse_bases_sym[1:]]
    sparse_bases_f_c_sym = [basis*free_bases_sym[1] for basis in sparse_bases_sym[1:]]
    sparse_bases_c_l_sym = [basis*free_bases_sym[2] for basis in sparse_bases_sym[1:]]
    
    # all bases
    bases_sym = sym.Matrix(free_bases_sym + sparse_bases_c_v_sym + sparse_bases_f_c_sym + sparse_bases_c_l_sym).transpose()

    # concatenated coeff
    coeff = concatenate_coeff(free_coeff, sparse_coeff_med, free_coeff_flag = 'Median')

    # linear regression
    lhs_pred_sym = bases_sym@(coeff.squeeze(0))

    print('L.H.S. of the discovered Equation')
    scale = 0.0991080627061296
    print(symbol_normalized['v']/scale)
    print('R.H.S. of the discovered Equation')
    print(sym.expand(lhs_pred_sym)[0, 0]/scale)

    coeff_std = concatenate_coeff(free_coeff, sparse_coeff_std, free_coeff_flag = 'Std')
    print('std of discovered rhs') 
    lhs_pred_sym_std = bases_sym@(coeff_std.squeeze(0))
    print(sym.expand(lhs_pred_sym_std)[0, 0]/scale)

#%%
# predict ensemble again using sparse eq coeff
with torch.no_grad():
    sparse_coeff[sparse_ind.expand(sparse_coeff.shape[0], -1, -1)] = 0

    lhs_pred_all, lhs_ref_all = nonbragging_UQ_pred(data_normalized_all, free_coeff, sparse_coeff)
    scipy.io.savemat('pred_lhs_prune_all.mat',{'lhs_pred_all':lhs_pred_all.detach().cpu().numpy(), 'lhs_ref_all':lhs_ref_all.cpu().numpy(),
    })

    lhs_pred_val, lhs_ref_val = nonbragging_UQ_pred(data_normalized_val_not_bragging_all, free_coeff, sparse_coeff)
    scipy.io.savemat('pred_lhs_prune_val.mat',{'lhs_pred_val':lhs_pred_val.detach().cpu().numpy(), 'lhs_ref_val':lhs_ref_val.cpu().numpy(),
    })


#%%
""" one-sample t-test to check if regressor coeff is statistically significant """
# https://en.wikipedia.org/wiki/Student%27s_t-test#One-sample_t-test
coeff_mean = coeff.squeeze(0)/scale
coeff_std = coeff_std.squeeze(0)/scale

nz_ind = coeff_std.nonzero(as_tuple=True)
coeff_mean = coeff_mean[nz_ind]
coeff_std = coeff_std[nz_ind]

test_stat = torch.abs((coeff_mean - 0))/(coeff_std/np.sqrt(free_coeff.shape[0]))
DOF = free_coeff.shape[0] - 1
t_score_baseline = scipy.stats.t.ppf(q=1-.05/2,df=DOF) # significance level is 5%

if torch.all(test_stat > t_score_baseline):
    print('All coeffs are significant')
else:
    print('Some coeffs are insignificant')

""" save probabilistic coeff """
## denormalize coeff
# free bases
free_bases_norm_coeff = [data_normalized_coeff['c_v'], data_normalized_coeff['f_c'], data_normalized_coeff['c_l'],
 data_normalized_coeff['d_a_d']*data_normalized_coeff['c_v'],
    data_normalized_coeff['d_a_d']*data_normalized_coeff['f_c'], data_normalized_coeff['d_a_d']*data_normalized_coeff['c_l']]

# sparse bases
sparse_bases_norm_coeff = BuildLib([1, torch.sqrt(data_normalized_coeff['d_t_d']), torch.sqrt(data_normalized_coeff['b_d']), torch.sqrt(data_normalized_coeff['w_d'])], 4)
sparse_bases_c_v_sym = [basis*free_bases_norm_coeff[0] for basis in sparse_bases_norm_coeff[1:]]
sparse_bases_f_c_sym = [basis*free_bases_norm_coeff[1] for basis in sparse_bases_norm_coeff[1:]]
sparse_bases_c_l_sym = [basis*free_bases_norm_coeff[2] for basis in sparse_bases_norm_coeff[1:]]

# all bases
bases_norm_coeff = torch.stack(free_bases_norm_coeff + sparse_bases_c_v_sym + sparse_bases_f_c_sym + sparse_bases_c_l_sym, 0).reshape((1, -1))

coeff_UQ = concatenate_coeff(free_coeff, sparse_coeff, None).squeeze(-1)
coeff_UQ_denorm = coeff_UQ*bases_norm_coeff/scale
scipy.io.savemat('UQ_coeff.mat',{'coeff_UQ':coeff_UQ_denorm.detach().cpu().numpy(),
    })

""" wrap-up """
writer.flush()
writer.close()

#%%