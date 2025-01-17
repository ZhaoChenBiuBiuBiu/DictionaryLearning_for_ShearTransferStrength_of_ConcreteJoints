import torch
import itertools
import numpy as np
from tqdm import tqdm 


def forward_pass(data_normalized, free_coeff, sparse_coeff, free_coeff_flag = None, loss_fn = torch.nn.MSELoss()):
    v = data_normalized['v']
    f_c = data_normalized['f_c']
    c_v = data_normalized['c_v']
    c_l = data_normalized['c_l']
    d_a_d = data_normalized['d_a_d']
    d_t_d = data_normalized['d_t_d']
    b_d = data_normalized['b_d']
    w_d = data_normalized['w_d']

    lhs_ref = v
    ## construct a library
    # free bases
    free_bases = [c_v, f_c, c_l, d_a_d*c_v, d_a_d*f_c, d_a_d*c_l]
    
    # sparse bases
    sparse_bases = BuildLib([torch.ones_like(d_a_d), torch.sqrt(d_t_d), torch.sqrt(b_d), torch.sqrt(w_d)], 4)
    sparse_bases_c_v = [basis*c_v for basis in sparse_bases[1:]]
    sparse_bases_f_c = [basis*f_c for basis in sparse_bases[1:]]
    sparse_bases_c_l = [basis*c_l for basis in sparse_bases[1:]]

    bases = torch.cat(free_bases + sparse_bases_c_v + sparse_bases_f_c + sparse_bases_c_l, dim=-1) 

    ## linear regression
    coeff = concatenate_coeff(free_coeff, sparse_coeff, free_coeff_flag) # (bragging, candidates, 1)
    lhs_pred = bases@coeff

    ## loss
    loss_data = loss_fn(lhs_pred, lhs_ref)

    return loss_data, lhs_pred, lhs_ref

def concatenate_coeff(free_coeff, sparse_coeff, free_coeff_flag):
    if free_coeff_flag == None:
        coeff = torch.cat([torch.sigmoid(free_coeff[:, :2, :]), free_coeff[:, 2:, :], sparse_coeff], dim=1)
    elif free_coeff_flag == 'Median':
        coeff = torch.cat([torch.median(torch.sigmoid(free_coeff[:, :2, :]), 0, keepdim = True)[0], torch.median(free_coeff[:, 2:, :], 0, keepdim = True)[0],
         sparse_coeff],
         dim=1)
    elif free_coeff_flag == 'Std':
        coeff = torch.cat([torch.std(torch.sigmoid(free_coeff[:, :2, :]), 0, keepdim = True), torch.std(free_coeff[:, 2:, :], 0, keepdim = True), sparse_coeff], dim=1)
    return coeff

def BuildLib(candidates, order):
    combos = list(itertools.combinations_with_replacement('0123', order))
    Lib = []
    for combo in combos:
        candidate_product = 1
        for sample in range(order):
            candidate_product *= candidates[int(combo[sample])]

        Lib.append(candidate_product)

    return Lib

def ell_half_norm(w):
    a = 0.01
    # torch.where gives NaN in backpropagation
    # regu_weight = torch.where(torch.abs(w) >= a, torch.sqrt(torch.abs(w)), torch.sqrt(-w**4/8/(a**3) + 3*w**2/4/a + 3*a/8))
    
    big_ind = torch.nonzero(torch.abs(w) >= a, as_tuple=True)
    regu_weight_big = torch.sum(torch.sqrt(torch.abs(w[big_ind])))

    small_ind = torch.nonzero(torch.abs(w) < a, as_tuple=True)
    regu_weight_small = torch.sum(torch.sqrt(-w[small_ind]**4/8/(a**3) + 3*w[small_ind]**2/4/a + 3*a/8))
   
    return regu_weight_big+regu_weight_small

def HardThreshold_Eq(eq_coeff, threshold, GradMask = False):
    with torch.no_grad():
        # apply hard threshold to SNN weights            
        ind1 = torch.abs(eq_coeff) < threshold
        eq_coeff[ind1] = 0
        # eq_coeff.requires_grad = True
        # # fix zero weights as zeros: Create Gradient mask
        # if GradMask == True:
        #     gradient_mask1 = torch.ones_like(eq_coeff)
        #     gradient_mask1[ind1] = 0
        #     eq_coeff.register_hook(lambda grad: grad.mul_(gradient_mask1))
    return eq_coeff, ind1

def CountNonZeros_Eq(eq_coeff):
    No_Nonzero = np.count_nonzero(eq_coeff.detach().to('cpu').numpy())
    return No_Nonzero

def train(free_coeff, sparse_coeff, data_loss_coeff_core, epoch, data_normalized_tr_all, data_normalized_val_not_bragging_all, writer):
    optimizer = torch.optim.Adam([free_coeff, sparse_coeff, data_loss_coeff_core], lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    act_fn = torch.nn.Softplus()
    for epoch in tqdm(range(epoch)): # 10000
        ## training forward pass
        loss_data_tr_list = []
        lhs_pred_tr_list = []
        lhs_ref_tr_list = []

        for data_normalized_tr in data_normalized_tr_all:
            # free_coeff_flag need to be consistent w sparse_coeff
            loss_data_tr, lhs_pred_tr, lhs_ref_tr = forward_pass(data_normalized_tr, free_coeff, sparse_coeff, None, loss_fn) 
            loss_data_tr_list.append(loss_data_tr) 
            lhs_pred_tr_list.append(lhs_pred_tr)
            lhs_ref_tr_list.append(lhs_ref_tr)
        
        ## Structured sparsity regularization
        loss_sparsity = ell_half_norm(torch.linalg.norm(sparse_coeff, dim=0))*1e-1

        ## total loss
        data_loss_coeff = act_fn(data_loss_coeff_core) + 1.0 # > 1 
        loss_data_tr_all = torch.sum(torch.stack(loss_data_tr_list)*data_loss_coeff) # (dataNo, 1) -> (1,)
        loss_tr = loss_data_tr_all + loss_sparsity

        ## val forward pass
        with torch.no_grad():
            # it's ok to have inconsistency in loss function here.
            sparse_coeff_med, _ = torch.median(sparse_coeff, dim=0, keepdim=True)
            loss_data_val_list = []
            lhs_pred_val_list = []
            lhs_ref_val_list = []
            for data_normalized_val_not_bragging in data_normalized_val_not_bragging_all:
                loss_data_val, lhs_pred_val, lhs_ref_val = forward_pass(data_normalized_val_not_bragging, free_coeff, sparse_coeff_med, 'Median', loss_fn) 
                loss_data_val_list.append(loss_data_val) 
                lhs_pred_val_list.append(lhs_pred_val) # (bragging, samples)
                lhs_ref_val_list.append(lhs_ref_val)

            loss_data_val_all = torch.sum(torch.stack(loss_data_val_list)*data_loss_coeff) # (dataNo, 1) -> (1,)

        writer.add_scalars('loss_data', {'tr':loss_data_tr_all.item(), 'val':loss_data_val_all.item()}, epoch)
        writer.add_scalar('loss_sparsity', loss_sparsity.item(), epoch)
        writer.add_scalars('data_loss_coeff', {str(i):data_loss_coeff[i].item() for i in range(data_loss_coeff.shape[0])}, epoch)

        optimizer.zero_grad()
        loss_tr.backward()
        optimizer.step()

    lhs_pred_tr = torch.cat(lhs_pred_tr_list, 1)
    lhs_ref_tr = torch.cat(lhs_ref_tr_list, 1)
    lhs_pred_val = torch.cat(lhs_pred_val_list, 1)
    # print(lhs_ref_val_list)
    lhs_ref_val = torch.cat(lhs_ref_val_list, 0)
    return lhs_pred_tr, lhs_ref_tr, lhs_pred_val, lhs_ref_val

def preprocess(data_torch, p_c_flag, device, data_normalized_coeff, bragging):
    # filter data based on p_c
    if p_c_flag != None:
        if p_c_flag:
            # p_c > 2200
            ind_valid = torch.nonzero(data_torch['p_c']>2200, as_tuple=True)
        else:
            # p_c <= 2200
            ind_valid = torch.nonzero(data_torch['p_c']<=2200, as_tuple=True)

        if ind_valid[0].nelement() == 0:
            return None, None, None, None 

        data_torch = {k:v[ind_valid].reshape((-1, 1)) for _, (k, v) in enumerate(data_torch.items())}

    # normalize data
    data_normalized = {k: v*data_normalized_coeff[k] for _, (k, v) in enumerate(data_torch.items())}

    # random split train/val data by 80%/20%
    split_ind = torch.randperm(data_normalized['c_v'].shape[0], device = device)

    if split_ind.shape[0]-1 == int(0.8*split_ind.shape[0]):
        tr_ind = split_ind[:-1]
        val_ind = split_ind[[-1]]
    else:
        tr_ind = split_ind[:int(0.8*split_ind.shape[0])]
        val_ind = split_ind[int(0.8*split_ind.shape[0]):]

    data_normalized_tr_not_bragging = {k: v[tr_ind] for _, (k, v) in enumerate(data_normalized.items())}
    data_normalized_val_not_bragging = {k: v[val_ind] for _, (k, v) in enumerate(data_normalized.items())}

    # bragging
    data_normalized_tr_list = {k: [] for _, (k, v) in enumerate(data_normalized.items())}
    for _ in range(bragging): # bragging. robust bagging/bootstrap aggregation
        # random split train/val data by 80%/20%
        boot_ind = torch.randint(data_normalized_tr_not_bragging['c_v'].shape[0], (data_normalized_tr_not_bragging['c_v'].shape[0],))
        for _, (k, v) in enumerate(data_normalized_tr_not_bragging.items()):
            data_normalized_tr_list[k].append(v[boot_ind])

    data_normalized_tr = {k: torch.stack(v, dim=0) for _, (k, v) in enumerate(data_normalized_tr_list.items())} # (bootstrapping, samples, 1)

    return data_normalized_tr_not_bragging, data_normalized_val_not_bragging, data_normalized_tr, data_normalized

def nonbragging_UQ_pred(data_normalized_all, free_coeff, sparse_coeff):
    lhs_pred_all_list = []
    lhs_ref_all_list = []
    for data_normalized in data_normalized_all:
        _, lhs_pred_all, lhs_ref_all = forward_pass(data_normalized, free_coeff, sparse_coeff, None)
        lhs_pred_all_list.append(lhs_pred_all)
        lhs_ref_all_list.append(lhs_ref_all)
    lhs_pred_all = torch.cat(lhs_pred_all_list, 1)
    lhs_ref_all = torch.cat(lhs_ref_all_list, 0)
    return lhs_pred_all, lhs_ref_all
