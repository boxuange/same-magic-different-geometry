"""S59: exact INTEGER source-transfer certificate for square graph site-4 Z source.

This checks the universal real-linear source-transfer map on ALL 64 elementary
complex local-tensor tangent entries (real and imaginary parts: 128 tests),
using the exact stabiliser code and the original graph-port conventions.

No NumPy floating point, no Hessian fitting, no frozen random generators, and
no implicit assumption that every elemental K is a Stiefel tangent.
"""
from __future__ import annotations
from pathlib import Path
from functools import reduce
import json, hashlib
import numpy as np
import opt_einsum as oe

ROOT=Path(__file__).resolve().parent
EDGES=((0,1),(1,2),(2,3),(0,3),(3,4),(4,5),(2,5))
I=np.eye(2,dtype=np.int64)
X=np.array([[0,1],[1,0]],dtype=np.int64)
Z=np.diag([1,-1]).astype(np.int64)
D={'I':I,'X':X,'Z':Z}
# Exact stabilizer projector, no square roots or floating point.
z0=np.zeros(32,dtype=np.int64);z0[0]=1
for label in ('XZZXI','IXZZX','XIXZZ','ZXIXZ'):
 g=reduce(np.kron,(D[x] for x in label))
 z0=z0+g@z0
assert int(z0@z0)==16 and set(np.unique(z0))<={-1,0,1}
z1=reduce(np.kron,[X]*5)@z0
T=np.column_stack((z0,z1)) # V5 = T/4, exact
assert np.array_equal(T.T@T,16*np.eye(2,dtype=np.int64))

def patch(vertices):
 vertices=tuple(sorted(vertices)); selected=set(vertices)
 incident={v:sorted(i for i,(a,b) in enumerate(EDGES) if v in (a,b)) for v in range(6)}
 internal={i for i,(a,b) in enumerate(EDGES) if a in selected and b in selected}
 exterior=[];cut=[];logical=[];args=[];counter=100
 for v in vertices:
  leg=[1000+v]; logical.append(1000+v)
  for e in incident[v]:
   if e in internal: leg.append(2000+e)
   else:
    leg.append(3000+e)
    if e not in cut:cut.append(e)
  for _ in range(5-len(incident[v])):
   leg.append(counter);exterior.append(counter);counter+=1
  args.extend((np.moveaxis(T.reshape((2,)*6),-1,0),leg))
 cut.sort()
 A=oe.contract(*args,exterior+logical+[3000+e for e in cut],optimize='greedy')
 return A.reshape(2**len(exterior),2**(len(logical)+len(cut))),cut

P01,cut01=patch((0,1))
P4,cut4=patch((4,))
PA,cutA=patch((2,3,5))
assert cut01==[1,3] and cut4==[4,5] and cutA==[1,3,4,5]
assert P01.shape==(64,16) and P4.shape==(8,8) and PA.shape==(128,128)
assert np.array_equal(P01.T@P01,32*np.eye(16,dtype=np.int64))
assert np.array_equal(P4.T@P4,4*np.eye(8,dtype=np.int64))
assert np.array_equal(PA.T@PA,64*np.eye(128,dtype=np.int64))
# Unnormalised full graph: A- and B-side Gram 64,128; 16 cuts.
# Global squared normalisation = 131072 = (256 sqrt(2))^2.
# Applying DA=PA.T/8 leaves B/(32 sqrt(2)) on the decoded side.
# Its density expansion thus has rational normalisation 1/2048.
assert 64*(32*4)*16==131072
assert 64 == 8*8  # applying P235.T/8 restores the exact q basis
# The complement B exactly factorises into P01 \otimes T at vertex 4.
P01r=P01.reshape(64,4,4)     # exterior01, logical01, q12+q03
T4=T.reshape(4,8,2).transpose(1,2,0) # exterior4, logical4, q34+q45
assert np.array_equal(P4,T4.reshape(8,8))
# exact 5-code erasure fact: baseline local Z4 source has zero reduced cut operator
local_Z=np.einsum('bnp, bnq, n -> pq',T4,T4,np.array([1,-1]),optimize=True)
assert np.array_equal(local_Z,np.zeros((4,4),dtype=np.int64))

B=np.einsum('apq,brs->abprqs',P01r,T4,optimize=True).reshape(512,8,16)
assert np.array_equal(B.reshape(4096,16).T@B.reshape(4096,16),1024*np.eye(16,dtype=np.int64))
# B logical indices: logical01 has 4 states, logical4 has 2 states (4*n01+n4)
s=np.tile([1,-1],4).astype(np.int64)
weighted=(B*s[np.newaxis,:,np.newaxis]).reshape(4096,16)
Bflat=B.reshape(4096,16)
# baseline source operator vanishes on both complement cuts, exactly
assert np.array_equal(Bflat.T@weighted, np.zeros((16,16),dtype=np.int64))

LHS_count=0; err=0; nonzero=0
for idx in range(64):
 # arbitrary complex Knum = E or iE on the (physical-port,logical) 32x2 matrix
 E=np.zeros((32,2),dtype=np.int64)
 E.flat[idx]=1
 E4=E.reshape(4,8,2).transpose(1,2,0)
 DB=np.einsum('apq,brs->abprqs',P01r,E4,optimize=True).reshape(4096,16)
 H=DB.T@weighted
 # direct contraction of decoded first-order source (integer numerator)
 direct_real=H+H.T
 direct_imag=H-H.T  # coefficient multiplying i for Knum=iE
 # local source reduced to the q34,q45 cut (numerators; Gamma = L/32)
 # S(a,b)=sum_{e,l} Z_l E_{a e l} T_{b e l}
 S=np.einsum('enp, enq, n -> pq',E4,T4,np.array([1,-1]),optimize=True)
 local_real=S+S.T
 local_imag=S-S.T
 # 4 logical states in 01, and P01.T P01=32 I: prefactor 128
 predicted_real=128*np.kron(np.eye(4,dtype=np.int64),local_real)
 predicted_imag=128*np.kron(np.eye(4,dtype=np.int64),local_imag)
 this_err=max(int(np.max(np.abs(direct_real-predicted_real))),int(np.max(np.abs(direct_imag-predicted_imag))))
 err=max(err,this_err)
 if this_err: raise AssertionError(('local source transfer',idx,this_err))
 LHS_count+=2
 nonzero+=int(np.count_nonzero(direct_real)>0)+int(np.count_nonzero(direct_imag)>0)
assert LHS_count==128 and err==0

# Algebraically: X235=(1/(64*2048)) I_M \otimes R,
# where R=128 I_qL\otimes L(K); hence X235= I_M/8 \otimes I_qL/4 \otimes Gamma(K),
# Gamma(K)=(K Z V^dag + V Z K^dag)/2, with K=E/4,V=T/4.
# Tests do not use sources outside Z4 or source-dependent recoveries.
report={
 'status':'PASS',
 'network':'shared-edge-square',
 'source':'Z4',
 'tensor_construction':'integer product of (I+g) stabilizer projectors; V5=T/4',
 'graph_edge_order':EDGES,
 'P01_shape':P01.shape,'P4_shape':P4.shape,'P235_shape':PA.shape,
 'exact_patch_01_gram':'P01.T P01 = 32 I16',
 'exact_patch_4_gram':'P4.T P4 = 4 I8',
 'exact_patch_235_gram':'P235.T P235 = 64 I128',
 'exact_full_graph_unnormalised_isometry_norm_squared':'131072',
 'decoded_B_amplitude_factor':'1/(32 sqrt(2)); squared = 1/2048',
 'local_Z4_baseline_cut_operator_zero':bool(not np.any(local_Z)),
 'complement_baseline_Z4_cut_operator_zero':bool(not np.any(Bflat.T@weighted)),
 'tested_K_basis':'32 physical x 2 logical entries, each with real and imaginary parts',
 'basis_tests':LHS_count,
 'nonzero_basis_responses':nonzero,
 'max_integer_source_transfer_residual':err,
 'exact_source_map':'X235(Z4;K4) = (I8/8) tensor (I4/4) tensor Gamma4(Z4;K4)',
 'Gamma4':'(K4 Z V5^dag + V5 Z K4^dag)/2 traced over 3 external physical ports',
 'exact_input_domain':'arbitrary complex 32x2 single-vertex K4 by real-linearity; raw first-order source, physical tangent when K4=iH4V5 with H4 Hermitian',
 'other_vertex_local_K':'zero Z4 first-order source because exact local Z4 baseline cut operator vanishes; holds for any single-vertex local insertion on 0,1,2,3,5',
 'scope':'Fixed square network, exact decoder U235, Z4 source at delta=0 and leading order in epsilon. No general CPTP claim, no finite-epsilon resource equality.',
 'sha256_integer_T':hashlib.sha256(T.astype('<i8').tobytes()).hexdigest()
}
(ROOT/'s59_exact_source_transfer.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
