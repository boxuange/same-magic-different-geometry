"""Minimal exact bridge-tensor helpers, extracted from the S35/S50 conventions.

The stabilizer tensor, CNOT permutations, and integer graph contractions use
integer arithmetic. Six frozen binary64 weighted Hermitians are data inputs,
not redrawn in the verifier. The saved arrays are promoted to dyadics in
s50_exact_binary64_physical.py.
"""
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
import numpy as np
import opt_einsum as oe

Q=1<<36
EDGES=((0,1),(1,2),(0,2),(2,3),(3,4),(4,5),(3,5))
I=np.eye(2,dtype=np.int64)
X=np.array([[0,1],[1,0]],dtype=np.int64)
Z=np.diag([1,-1]).astype(np.int64)

def local_integer_tensor():
    z=np.zeros(32,dtype=np.int64);z[0]=1
    for word in ('XZZXI','IXZZX','XIXZZ','ZXIXZ'):
        g=reduce(np.kron,[{'I':I,'X':X,'Z':Z}[w] for w in word])
        z=z+g@z
    other=reduce(np.kron,[X]*5)@z
    t=np.column_stack((z,other))
    assert np.array_equal(t.T@t,16*np.eye(2,dtype=np.int64))
    return t

def weighted_integer_hermitians(seed=20261009):
    assert seed==20261009, 'Public S50 proof is for the fixed, archived seed only'
    H=np.load(Path(__file__).resolve().parent/'bridge_weighted_H_binary64.npy',allow_pickle=False)
    assert H.shape==(6,32,32)
    assert all(np.array_equal(H[v],H[v].conj().T) for v in range(6))
    re=[np.rint(Q*h.real).astype(np.int64) for h in H]
    im=[np.rint(Q*h.imag).astype(np.int64) for h in H]
    assert all(np.array_equal(h,h.T) for h in re)
    assert all(np.array_equal(h,-h.T) for h in im)
    return re,im,[h for h in H]

def cnot_permutation(n,control,target):
    assert n>0 and 0<=control<n and 0<=target<n and target!=control
    idx=np.arange(1<<n)
    cb=1<<(n-1-control);tb=1<<(n-1-target)
    return idx^np.where(idx&cb,tb,0)

@contextmanager
def bridge_network():
    yield

def integer_graph_contraction(chosen=range(6),*,modified=None):
    chosen=tuple(sorted(chosen));selected=set(chosen)
    adj={v:sorted(i for i,e in enumerate(EDGES) if v in e) for v in range(6)}
    internal={i for i,(a,b) in enumerate(EDGES) if a in selected and b in selected}
    ext=[];cut=[];log=[];args=[];counter=100
    local=local_integer_tensor()
    for v in chosen:
        logical_leg=1000+v
        legs=[logical_leg];log.append(logical_leg)
        for e in adj[v]:
            if e in internal: legs.append(2000+e)
            else:
                legs.append(3000+e)
                if e not in cut:cut.append(e)
        for _ in range(5-len(adj[v])):
            legs.append(counter);ext.append(counter);counter+=1
        data=modified[1] if modified is not None and v==modified[0] else local
        args.extend([np.moveaxis(data.reshape((2,)*6),-1,0),legs])
    cut.sort()
    p=oe.contract(*args,ext+log+[3000+e for e in cut],optimize='greedy')
    return p.reshape(1<<len(ext),1<<(len(log)+len(cut))),len(internal)
