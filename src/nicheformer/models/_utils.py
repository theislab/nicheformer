import os
import torch
import random
import numpy as np
    
def complete_masking(batch, p, n_tokens):
    
    padding_token = 1
    cls_token = 3

    indices = batch['X']

    indices = torch.where(indices == 0, torch.tensor(padding_token), indices) # 0 is originally the padding token, we change it to 1
    batch['X'] = indices

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p) # mask indices with probability p
    
    masked_indices = indices * mask # masked_indices 
    masked_indices = torch.where(indices != padding_token, masked_indices, indices) # we just mask non-padding indices
    mask = torch.where(indices == padding_token, torch.tensor(padding_token), mask) # in the model we evaluate the loss of mask position 0
    # so we make the mask of all PAD tokens to be 1 so that it's not taken into account in the loss computation
    
    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices, indices) # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, torch.tensor(padding_token), mask) # we change the mask so that it doesn't mask any CLS token

    # 80% of masked indices are masked
    # 10% of masked indices are a random token
    # 10% of masked indices are the real token

    random_tokens = torch.randint(10, n_tokens, size=masked_indices.shape, device=masked_indices.device)
    random_tokens = random_tokens * torch.bernoulli(torch.ones_like(random_tokens)*0.1).type(torch.int64) 

    masked_indices = torch.where(masked_indices == 0, random_tokens, masked_indices) # put random tokens just in the previously masked tokens

    same_tokens = indices.clone()
    same_tokens = same_tokens * torch.bernoulli(torch.ones_like(same_tokens) * 0.1).type(torch.int64)

    masked_indices = torch.where(masked_indices == 0, same_tokens, masked_indices) # put same tokens just in the previously masked tokens

    batch['masked_indices'] = masked_indices
    batch['mask'] = mask
    
    attention_mask = (masked_indices == padding_token)
    batch['attention_mask'] = attention_mask.type(torch.bool)

    return batch


def set_seed(seed):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"