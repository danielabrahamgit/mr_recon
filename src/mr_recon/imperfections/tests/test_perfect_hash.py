# Easy Perfect Minimal Hashing
# By Steve Hanov. Released to the public domain.
# Adapted to Python 3 best practices and 64-bit integer keys by Martijn Pieters
#
# Based on:
# Edward A. Fox, Lenwood S. Heath, Qi Fan Chen and Amjad M. Daoud,
# "Practical minimal perfect hash functions for large databases", CACM, 35(1):105-121
# also a good reference:
# Compress, Hash, and Displace algorithm by Djamal Belazzougui,
# Fabiano C. Botelho, and Martin Dietzfelbinger
import random
import torch
from mr_recon.indexing import ravel
from itertools import count, groupby

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


# Sparse tensor example
torch_dev = torch.device('cpu')
d = 5
M = 100_000
grd_size = (50,)*d

# Random indices and values
inds = torch.randint(0, grd_size[0], (d, M), device=torch_dev, dtype=torch.int64).unique(dim=1)
M = inds.shape[1]
inds_flt = ravel(inds, grd_size, 0)
vals = torch.randn(M, device=torch_dev)
# inds_un = torch.stack(torch.unravel_index(inds_flt, grd_size), dim=0) # == inds

# Build MPH
testdata = {inds_flt[m].item(): m
            for m in range(M)}
tables = gen_minimal_perfect_hash(testdata)
for key, value in testdata.items():
    assert perfect_hash_lookup(key, *tables) == value
    
# Convert to torch 
interm, values = tables
interm = torch.tensor(interm, device=torch_dev, dtype=torch.int64)
values = torch.tensor(values, device=torch_dev, dtype=torch.int64)

breakpoint()


test_keys = inds_flt[:100]
test_gt = [perfect_hash_lookup(key.item(), *tables) for key in test_keys]
test = perfect_hash_lookup_torch(test_keys, interm, values)
for i in range(test.shape[0]):
    assert test[i].item() == test_gt[i]
