"""Exact Pauli-CNOT combinatorics and dyadic S55 source Gram, without entropy Hessians.
All Clifford checks are equality tests on pairs of integer matrices representing
Gaussian-integer Pauli strings. Scalar coefficients are fractions throughout.
"""
from pathlib import Path
import json, itertools, hashlib
from functools import reduce
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parent
S58=ROOT/'S58'
with open(S58/'square_pauli_exact_dyadic.json') as f: previous=json.load(f)
beta={'IX':Fraction(previous['beta_I_X_exact']),
      'IY':Fraction(previous['beta_I_Y_exact']),
      'IZ':Fraction(previous['beta_I_Z_exact'])}
I=np.eye(2,dtype=np.int64)
X=np.array([[0,1],[1,0]],dtype=np.int64)
Z=np.diag([1,-1]).astype(np.int64)
R=X@Z
REAL={'I':I,'X':X,'Y':R,'Z':Z}
def pauli_pair(word):
    mat=reduce(np.kron,[REAL[z] for z in word]); n=word.count('Y')%4
    if n==0:return mat,np.zeros_like(mat)
    if n==1:return np.zeros_like(mat),mat
    if n==2:return -mat,np.zeros_like(mat)
    return np.zeros_like(mat),-mat
names=[''.join(w) for w in itertools.product('IXYZ',repeat=4)]
pairs={s:pauli_pair(s) for s in names}
def cnot_perm(n,c,t):
    ix=np.arange(1<<n,dtype=int)
    return ix^np.where(ix&(1<<(n-1-c)),1<<(n-1-t),0)
routes=[None,(0,3),(3,0),(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(0,2),(2,0),(1,3),(3,1)]
features=[a+b for a in 'XYZ' for b in 'IXYZ']
F=[[Fraction(0) for j in routes] for q in features]
verified=0; vanished_orthogonal=0; active_positions=[]
for j,route in enumerate(routes):
 perm=np.arange(16) if route is None else cnot_perm(4,*route)
 for input2 in [a+b for a in 'IXYZ' for b in 'IXYZ' if a+b!='II']:
  inword='II'+input2
  rr,ri=pairs[inword]; rr=rr[np.ix_(perm,perm)];ri=ri[np.ix_(perm,perm)]
  found=[(q,sgn) for q,(qr,qi) in pairs.items() for sgn in (+1,-1)
           if np.array_equal(rr,sgn*qr) and np.array_equal(ri,sgn*qi)]
  assert len(found)==1,(j,input2,found)
  output,sgn=found[0];verified+=1
  # For 235->25: q1 and q2 must be identity.
  # For 25->5: additionally q0 must be identity.
  if output[1]!='I' or output[2]!='I' or output[0]=='I':
   continue
  # The contribution survived the first channel and was erased by the second.
  if input2 not in beta:
   vanished_orthogonal+=1
   raise AssertionError(('new extra source-dependent feature',j,input2,output))
  key=output[0]+output[3]
  F[features.index(key)][j]+=4*sgn*beta[input2]
  active_positions.append([j,input2,output,int(sgn)])
assert verified==13*15
B=[[sum(F[q][j]*F[q][k] for q in range(12)) for k in range(13)] for j in range(13)]
nonzero=[(i,j) for i in range(13) for j in range(13) if B[i][j]]
assert nonzero==[(1,1),(2,2)],nonzero
expected11=Fraction(previous['B_11_exact'])
expected22=Fraction(previous['B_22_exact'])
assert B[1][1]==expected11 and B[2][2]==expected22
assert B[1][1]>0 and B[2][2]>0
# Exact witness at the archived binary64 centre
ctl=np.load(S58/'S55/s55_frozen_control.npy')
x=[Fraction(float(z)) for z in ctl]
value=B[1][1]*x[1]*x[1]+B[2][2]*x[2]*x[2]
assert value==Fraction(previous['W_at_selected_binary64_x_exact'])
# Centre is not the Banach fixed-point solution; need evaluate rigorous bound separately.
report={'status':'PASS','exact':True,'source':'Z4','network':'square',
 'Clifford_pauli_routes_checked':verified,
 'Gaussian_integer_matrix_max_residual':0,
 'rank_13x13_Gram':2,
 'only_nonzero_entries':[[i,j] for i,j in nonzero],
 'B_11_fraction':str(B[1][1]),'B_22_fraction':str(B[2][2]),
 'B_11_decimal':float(B[1][1]),'B_22_decimal':float(B[2][2]),
 'W_saved_binary64_centre_fraction':str(value),
 'W_saved_binary64_centre_decimal':float(value),
 'active_pauli_routings':active_positions,
 'scope':'Fixed 13 CNOT routes, single source Z4, Pauli filtering. Exact after the integer site4 transfer identity, input Hermitian dyadic beta, and exact patch erasure maps. No claim on resource equalities at finite epsilon.'}
(ROOT/'s59_exact_clifford_gram.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
