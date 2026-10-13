"""Exact rational (binary64 interpreted as dyadics) local-Pauli witness,
combined with S39's resource Banach-ball existence certificate.
Uses original six Hermitian weighted inputs as exact binary64 rationals,
and exact [[5,1,3]] integer tensor. Does not contract six-tensor Hessian.
"""
from pathlib import Path
import sys,json
from fractions import Fraction as F
import numpy as np
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from s50_exact_local_pauli_gram import gate_maps,local_integer_tensor,weighted_integer_hermitians

def local_binary64_coeffs():
 H=weighted_integer_hermitians(20261009)[2][4]
 Hr=H.real;Hi=H.imag
 exponents=[float(x).as_integer_ratio()[1].bit_length()-1 for x in np.concatenate((Hr.ravel(),Hi.ravel()))]
 power=max(exponents);hden=1<<power
 rh=np.array([int(float(x).as_integer_ratio()[0]*(hden//float(x).as_integer_ratio()[1])) for x in Hr.ravel()],dtype=object).reshape(32,32)
 ih=np.array([int(float(x).as_integer_ratio()[0]*(hden//float(x).as_integer_ratio()[1])) for x in Hi.ravel()],dtype=object).reshape(32,32)
 assert all(rh[i,j]==rh[j,i] and ih[i,j]==-ih[j,i] for i in range(32) for j in range(32))
 v=local_integer_tensor().astype(object)
 kr=-(ih@v);ki=rh@v
 v=v.reshape(4,8,2);kr=kr.reshape(4,8,2);ki=ki.reshape(4,8,2)
 rr=np.zeros((4,4),object);ri=np.zeros((4,4),object)
 for q in range(4):
  for r in range(4):
   rr[q,r]=sum((kr[q,e,l]*v[r,e,l]+v[q,e,l]*kr[r,e,l])*(1 if l==0 else -1) for e in range(8) for l in range(2))
   ri[q,r]=sum((ki[q,e,l]*v[r,e,l]-v[q,e,l]*ki[r,e,l])*(1 if l==0 else -1) for e in range(8) for l in range(2))
 assert all(rr[q,r]==rr[r,q] and ri[q,r]==-ri[r,q] for q in range(4) for r in range(4))
 from s50_exact_local_pauli_gram import pauli_int
 from itertools import product
 nums={}
 for t in product('IXYZ',repeat=2):
  w=''.join(t);P,y=pauli_int(w)
  pm=rr if y%2==0 else ri
  num=int(np.trace(P.astype(object)@pm))*(1 if y%4 in (0,1) else -1)
  if w=='II':assert num==0
  else:nums[w]=num
 return nums,32*hden,power

def main():
 from s50_exact_local_pauli_gram import CTRL
 nums,den,power=local_binary64_coeffs()
 maps,_=gate_maps()
 f=[{} for _ in CTRL]
 for i,mapping in enumerate(maps):
  for key,num in nums.items():
   w,sgn=mapping[key]
   if w[:2]!='II':f[i][w]=num*sgn
 G=[[sum(f[i].get(w,0)*f[j].get(w,0) for w in f[i].keys()|f[j].keys()) for j in range(13)] for i in range(13)]
 # Structural integer checks, exact rational witness model.
 blocks=[(1,9),(2,10),(5,11),(6,12)]
 assert all(G[i][j]==0 for i in range(13) for j in range(13) if (i in (0,3,4,7,8) or j in (0,3,4,7,8)))
 assert all(G[i][j]==0 for i in range(13) for j in range(13) if i!=j and all({i,j}!={a,b} for a,b in blocks))
 assert all(G[a][a]==G[d][d] and G[b][b]==G[c][c] and G[a][b]==G[c][d] for (a,b),(c,d) in [((1,9),(5,11)),((2,10),(6,12))])
 det={f'{a}_{b}':str(G[a][a]*G[b][b]-G[a][b]*G[a][b]) for a,b in blocks}
 assert all(int(v)>0 for v in det.values())
 baseline=np.load(str(ROOT/'historical_20261009_grams.npz'))['witness']
 gap=max(abs(float(F(G[i][j],den**2))-float(baseline[i,j])) for i in range(13) for j in range(13))
 # Archived binary64 array comparison is diagnostic; exact equality uses
 # S43 patch factorisation and exact 256-Pauli Kraus-channel identity.
 archived=json.loads(Path(str(ROOT/'resource_manifold_audit.json')).read_text())['20261009']
 center=[F(*float(x).as_integer_ratio()) for x in archived['numerical_continuation_not_exact_roots'][1]['coefficients']]
 piv=[0,1,4,6,7,9,10,11,12]
 r=F(37,2000000)
 bvec=[sum(F(G[i][j],den**2)*center[j] for j in range(13)) for i in range(13)]
 wc=sum(center[i]*bvec[i] for i in range(13))
 werr1=2*r*sum(abs(bvec[j]) for j in piv)
 werr2=r*r*sum(abs(G[i][j]) for i in piv for j in piv)/den**2
 lower=wc-werr1-werr2
 assert lower>F(1,100000)
 # Each potential matched coefficient x* in the certified Banach pivot ball
 # has W(x*) > lower and W(e0)=0 (exact).
 # The exact witness is positive for all combinations and zero on the 5 inactive coordinates.
 eig=np.linalg.eigvalsh(np.asarray(G,dtype=float)/float(den**2))
 print('EXACT BINARY64 LOCAL MODEL',power,'max archive gap',gap,'rank',8)
 print('EXACT DYADIC BALL WITNESS LOWER',float(lower),'center',float(wc),'linear radius error',float(werr1),'quadratic error',float(werr2))
 print('positive eig pairs',eig[5:]);print('det strictly >0',True)
 out={'scope':'S39 historical binary64 Hermitian inputs interpreted as exact dyadics; vertex4 local source; 13 paired cut-CNOT directions','h4_dyadic_denominator_exponent':power,
  'local_15_pauli_coefficient_denominator_decimal':str(den),'local_15_pauli_integer_numerators':{k:str(v) for k,v in nums.items()},
  'exact_integer_numerator_witness_gram':[[str(x) for x in row] for row in G],
  'four_block_positive_principal_minor_determinants':det,'double_eigenvalue_structure_exact':True,'rank_8_exact':True,
  'max_numeric_discrepancy_vs_archived_witness_gram':gap,
  'exact_Banach_ball_witness_lower_fraction':str(lower),'exact_Banach_ball_witness_lower_decimal':float(lower),
  'banach_center_witness_fraction':str(wc),'banach_radius_linear_error_fraction':str(werr1),'banach_radius_quadratic_error_fraction':str(werr2),
  'strict_above_1e-5':bool(lower>F(1,100000))}
 (ROOT/'s50_exact_binary64_physical_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
