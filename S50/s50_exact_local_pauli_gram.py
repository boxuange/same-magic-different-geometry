"""S50 exact dyadic 15-Pauli microscopic witness on S39 bridge.
Input: integer [[5,1,3]] tensor and vertex4 weighted Hermitian at denominator 2^36.
No six-tensor geometry response, no entropy Hessian, no floating point in Gram entries.
"""
import json,sys,itertools,math,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
from minimal_bridge_core import local_integer_tensor, weighted_integer_hermitians, Q, cnot_permutation
CTRL=[None,(0,3),(3,0),(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(0,2),(2,0),(1,3),(3,1)]
I=np.eye(2,dtype=np.int64);X=np.array([[0,1],[1,0]],dtype=np.int64);YI=np.array([[0,-1],[1,0]],dtype=np.int64);Z=np.diag([1,-1]).astype(np.int64)
D={'I':I,'X':X,'Y':YI,'Z':Z}

def pauli_int(s):
 a=np.array([[1]],np.int64);p=0
 for z in s: a=np.kron(a,D[z]);p+=int(z=='Y')
 return a,p%4

def local_15_numerators(seed=20261009):
 hr,hi,_=weighted_integer_hermitians(seed)
 V=local_integer_tensor().reshape(4,8,2)
 Kr=(-(hi[4]@V.reshape(32,2))).reshape(4,8,2)
 Ki=(hr[4]@V.reshape(32,2)).reshape(4,8,2)
 sign=np.array([1,-1],dtype=np.int64)
 # Rnum uses Vint and Knum; actual local source matrix is Rnum/(32*Q).
 Rr=np.einsum('qel,l,rel->qr',Kr,sign,V,optimize=True)+np.einsum('qel,l,rel->qr',V,sign,Kr,optimize=True)
 Ri=np.einsum('qel,l,rel->qr',Ki,sign,V,optimize=True)-np.einsum('qel,l,rel->qr',V,sign,Ki,optimize=True)
 assert np.array_equal(Rr,Rr.T) and np.array_equal(Ri,-Ri.T)
 assert int(np.trace(Rr))==0
 out={}
 for letters in itertools.product('IXYZ',repeat=2):
  w=''.join(letters)
  P,y=pauli_int(w)
  prod=P@Rr if y%2==0 else P@Ri
  co=int(np.trace(prod)) * (1 if y%4 in (0,1) else -1)
  if w=='II': assert co==0
  else:out[w]=co
 return out

def gate_maps():
 labels=[''.join(c) for c in itertools.product('IXYZ',repeat=4)]
 mats={p:pauli_int(p) for p in labels}
 maps=[]
 for ctrl in CTRL:
  perm=np.arange(16) if ctrl is None else cnot_permutation(4,*ctrl)
  one={}
  for right in itertools.product('IXYZ',repeat=2):
   p='II'+''.join(right)
   mat,phase=mats[p]
   conj=mat[np.ix_(perm,perm)]
   # conjugation of Pauli word by CNOT maps to signed Pauli; imaginary factor y is preserved modulo phase.
   matches=[]
   for out,(t,tphase) in mats.items():
    if (phase-tphase)%2:continue
    for real_sign in (1,-1):
     if np.array_equal(conj,real_sign*t):
      physical_sign=real_sign*(1 if (phase-tphase)%4==0 else -1)
      matches.append((out,physical_sign))
   assert len(matches)==1,(ctrl,p,matches)
   one[p[2:]]=matches[0]
  maps.append(one)
 return maps,labels

def exact_micro():
 t0=time.time();nums=local_15_numerators(); den=32*Q
 maps,labels=gate_maps(); features=[{} for _ in CTRL]
 for i,one in enumerate(maps):
  for p,n in nums.items():
   img,sgn=one[p]
   if img[:2]!='II':features[i][img]=sgn*n
 # integer Gram from Python unbounded integers
 G=[[sum(features[i].get(w,0)*features[j].get(w,0) for w in set(features[i])|set(features[j])) for j in range(13)] for i in range(13)]
 GG=np.asarray(G,dtype=np.float64)/den**2
 archived=np.load(str(ROOT/'historical_20261009_grams.npz'))['witness']
 err=float(np.max(abs(GG-archived)))
 eig=np.linalg.eigvalsh(GG)
 print('local 15 nonzero',[p for p,n in nums.items() if n],flush=True)
 print('DENOM',den,'ARCHIVE_MAX_GAP',err,'POS_EIG',eig[5:],'elapsed',time.time()-t0,flush=True)
 # exact 4 nontrivial 2x2 blocks: order [1,9],[2,10],[5,11],[6,12].
 blocks={f'{a}_{b}':[[G[a][a],G[a][b]],[G[b][a],G[b][b]]] for a,b in [(1,9),(2,10),(5,11),(6,12)]}
 exact_dets={k:int(v[0][0]*v[1][1]-v[0][1]**2) for k,v in blocks.items()}
 print('exact positive determinants?',{k:(v>0) for k,v in exact_dets.items()})
 print('max off-block entry int',max(abs(G[i][j]) for i in range(13) for j in range(13) if i!=j and {i,j} not in [set(v) for v in [(1,9),(2,10),(5,11),(6,12)]]))
 print('repeated blocks under reversed coordinates?',[[v for v in row] for row in blocks['1_9']]==[[blocks['5_11'][1][1],blocks['5_11'][1][0]],[blocks['5_11'][0][1],blocks['5_11'][0][0]]])
 assert err<1e-10
 assert all(v>0 for v in exact_dets.values())
 assert all(G[i][j]==0 for i in range(13) for j in range(13) if (i in (0,3,4,7,8) or j in (0,3,4,7,8)))
 assert all(blocks[a][i][j]==blocks[b][1-i][1-j] for a,b in [('1_9','5_11'),('2_10','6_12')] for i in (0,1) for j in (0,1))
 out={'physical_model':'S39 exact dyadic weighted Hermitian site4 and six perfect tensors','site4_H_entry_denominator':Q,'local_transfer_coefficient_denominator':den,'local_transfer_15_pauli_integer_numerators':nums,
   'clifford_gate_labels':CTRL,'exact_two_by_two_blocks_numerator':blocks,'exact_principal_minor_determinants':{k:str(v) for k,v in exact_dets.items()},'rank_exact':8,
   'exact_gram_numerator':[[str(z) for z in row] for row in G],'max_difference_vs_archived_B':err,'positive_eigenvalues_numeric':eig[5:].tolist(),'computation_s':time.time()-t0}
 (ROOT/'s50_exact_local_micro_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
 return out
if __name__=='__main__':exact_micro()
