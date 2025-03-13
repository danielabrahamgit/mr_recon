import torch
import numpy as np
import cupy as cp
import time

import matplotlib
matplotlib.use('Webagg')
import matplotlib.pyplot as plt

from mr_recon.utils import np_to_torch, torch_to_np
from mr_recon.indexing import ravel
from itertools import count, groupby
from tqdm import tqdm

@torch.jit.script
def fnv_hash_int_torch_jit(values: torch.Tensor, 
                           size: int, 
                           d: int = 0x811c9dc5) -> torch.Tensor:
    # Initialize hash with the FNV offset basis.
    hash_val = values.new_full(values.size(), d)
    # The eight shifts for the 8 bytes of a 64-bit integer.
    shifts = [56, 48, 40, 32, 24, 16, 8, 0]
    for shift in shifts:
        byte = (values >> shift) & 0xff
        hash_val = ((hash_val * 0x01000193) ^ byte) & 0xffffffff
    return hash_val % size

def fnv_hash_int_torch_vectorized(values, size, d=None):
    """
    Vectorized FNV hash for a tensor of 64-bit integers.

    Parameters:
    -----------
    values : torch.Tensor
        A tensor of dtype torch.int64 with shape (...,)
    size : int 
        The modulus to apply at the end (i.e. the hash will be in range [0, size-1]).
    d : int 
        The initial hash value (default is 0x811c9dc5) with shape (...,)

    Returns:
    --------
    torch.Tensor 
        A tensor of hashed values modulo 'size', with the same shape as 'values'.
    """
    # Ensure values is a 64-bit integer tensor.
    assert values.dtype == torch.int64
    
    if d is None:
        d = torch.full_like(values, 0x811c9dc5, dtype=torch.int64)
    
    # Initialize hash with the FNV offset basis.
    hash_val = d.clone()
    
    # For each of the 8 bytes in the 64-bit integer (big-endian order)
    for shift in [56, 48, 40, 32, 24, 16, 8, 0]:
        # Extract the corresponding byte.
        byte = (values >> shift) & 0xff
        # Multiply by FNV prime, XOR with the byte, and mask to 32 bits.
        hash_val = ((hash_val * 0x01000193) ^ byte) & 0xffffffff

    return hash_val % size

def fnv_hash_int(value, size, d=0x811c9dc5):
    """Calculates a distinct hash function for a given 64-bit integer.

    Each value of the integer d results in a different hash value. The return
    value is the modulus of the hash and size.

    """
    # Use the FNV algorithm from http://isthe.com/chongo/tech/comp/fnv/
    # The unsigned integer is first converted to a 8-character byte string.
    for c in value.to_bytes(8, 'big'):
        d = ((d * 0x01000193) ^ c) & 0xffffffff

    return d % size

def gen_minimal_perfect_hash(dictionary, _hash_func=fnv_hash_int):
    """Computes a minimal perfect hash table using the given Python dictionary.

    It returns a tuple (intermediate, values). intermediate and values are both
    lists. intermediate contains the intermediate table of indices needed to
    compute the index of the value in values; a tuple of (flag, d) is stored, where
    d is either a direct index, or the input for another call to the hash function.
    values contains the values of the dictionary.

    """
    size = len(dictionary)

    # Step 1: Place all of the keys into buckets
    buckets = [[] for __ in dictionary]
    intermediate = [(False, 0)] * size
    values = [None] * size

    for key in dictionary:
        buckets[_hash_func(key, size)].append(key)

    # Step 2: Sort the buckets and process the ones with the most items first.
    buckets.sort(key=len, reverse=True)
    # Only look at buckets of length greater than 1 first; partitioned produces
    # groups of buckets of lengths > 1, then those of length 1, then the empty
    # buckets (we ignore the last group).
    partitioned = (g for k, g in groupby(buckets, key=lambda b: len(b) != 1))
    for bucket in next(partitioned, ()):
        # Try increasing values of d until we find a hash function
        # that places all items in this bucket into free slots
        for d in count(1):
            slots = {}
            for key in bucket:
                slot = _hash_func(key, size, d=d)
                if values[slot] is not None or slot in slots:
                    break
                slots[slot] = dictionary[key]
            else:
                # all slots filled, update the values table; False indicates
                # these values are inputs into the hash function
                intermediate[_hash_func(bucket[0], size)] = (False, d)
                for slot, value in slots.items():
                    values[slot] = value
                break

    # The next group is buckets with only 1 item. Process them more quickly by
    # directly placing them into a free slot.
    freelist = (i for i, value in enumerate(values) if value is None)

    for bucket, slot in zip(next(partitioned, ()), freelist):
        # These are 'direct' slot references
        intermediate[_hash_func(bucket[0], size)] = (True, slot)
        values[slot] = dictionary[bucket[0]]

    return (intermediate, values)

def perfect_hash_lookup_torch(key, intermediate, values, _hash_func=fnv_hash_int_torch_vectorized):
    "Look up a value in the hash table defined by intermediate and values"
    direct, d = intermediate[_hash_func(key, len(intermediate))].T
    vals_direct = values[d]
    vals_indirect = values[_hash_func(key, len(values), d=d)] 
    return torch.where(direct.type(torch.bool), vals_direct, vals_indirect)

def perfect_hash_lookup(key, intermediate, values, _hash_func=fnv_hash_int):
    "Look up a value in the hash table defined by intermediate and values"
    direct, d = intermediate[_hash_func(key, len(intermediate))]
    return values[d if direct else _hash_func(key, len(values), d=d)]


torch_dev = torch.device(5)
# torch_dev = torch.device('cpu')

# Sparse tensor example
d = 4
M = 100_000
grd_size = (50,)*d

# Random indices and values
inds = torch.randint(0, grd_size[0], (d, M), device=torch_dev).unique(dim=1)
M = inds.shape[1]
inds_flt = ravel(inds, grd_size, 0)
vals = torch.randn(inds.shape[1], device=torch_dev)

# Create sparse tensor and regular tensor
sparse_tsr = torch.sparse_coo_tensor(inds_flt[None,], vals, size=(np.prod(grd_size).item(),), device=torch_dev)
dense_tsr = torch.zeros(grd_size, device=torch_dev)
dense_tsr[tuple(inds)] = vals
dense_tsr_flt = dense_tsr.flatten()

# Build MPH
testdata = {inds_flt[m].item(): m
            for m in range(M)}
tables = gen_minimal_perfect_hash(testdata)
for key, value in testdata.items():
    assert perfect_hash_lookup(key, *tables) == value
    
# Convert to torch 
interm, arr = tables
interm = torch.tensor(interm, device=torch_dev, dtype=torch.int64)
arr = torch.tensor(arr, device=torch_dev, dtype=torch.int64)

# Test indexing speed for both
num_inds = M
runs = 1000
time_dense = 0
time_sparse = 0
time_mph = 0
for _ in tqdm(range(runs)):
    
    rnd_inds = inds[:, torch.randperm(M, device=torch_dev)[:num_inds]]
    # rnd_inds = inds
    rnd_inds_flt = ravel(rnd_inds, grd_size, 0)
    torch.cuda.synchronize()
    
    # Time MPH
    start = time.perf_counter()
    one_d_inds = perfect_hash_lookup_torch(rnd_inds_flt, interm, arr)
    vals_mph = vals[one_d_inds]
    torch.cuda.synchronize()
    end = time.perf_counter()
    time_mph += end - start
    
    # Time dense
    start = time.perf_counter()
    vals_dense = dense_tsr[tuple(rnd_inds)]
    # vals_dense = dense_tsr_flt.index_select(0, rnd_inds_flt)
    torch.cuda.synchronize()
    end = time.perf_counter()
    time_dense += end - start
    
    # Time sparse
    start = time.perf_counter()
    vals_sparse = torch.zeros((rnd_inds.shape[1],), device=torch_dev, dtype=vals.dtype)
    ind_select_sparse = sparse_tsr.index_select(dim=0, index=rnd_inds_flt).coalesce()
    vals_sparse[ind_select_sparse.indices()[0]] = ind_select_sparse.values() 
    torch.cuda.synchronize()
    end = time.perf_counter()
    time_sparse += end - start

    # assert vals_sparse.isclose(vals_dense).all()
    # assert vals_mph.isclose(vals_dense).all()

print(f'Dense time = {1e3*time_dense/runs:.4f} ms')
print(f'Sparse time = {1e3*time_sparse/runs:.4f} ms')
print(f'MPH time = {1e3*time_mph/runs:.4f} ms')
print(f'\nMemory Dense = {dense_tsr.element_size() * dense_tsr.numel() / (2 ** 20):.4f} MB')
print(f'Memory Sparse = {sparse_tsr.element_size() * M / (2 ** 20):.4f} MB')
print(f'Memory MPH = {interm.element_size() * interm.numel() / (2 ** 30) + arr.element_size() * arr.numel() / (2 ** 20):.4f} MB')