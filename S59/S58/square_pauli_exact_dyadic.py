"""Independently recalculate the three local Pauli coefficients from frozen H4 as exact dyadic rationals.
No float arithmetic after ingesting H4 and integer five-qubit stabiliser tensor.
"""
from fractions import Fraction as Q
import numpy as np,json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent/'S55'/'s55_frozen_binary64_weighted_H.npy'
H=np.load(p)[4]
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'S50'))
from minimal_bridge_core import local_integer_tensor
T=local_integer_tensor()
assert set(np.unique(T))<=set([-1,0,1])
R=[[Q(float(H[i,j].real)) for j in range(32)] for i in range(32)]
I=[[Q(float(H[i,j].imag)) for j in range(32)] for i in range(32)]
# K = i H V5, and V5 = T/4: retain the numerator, Knum=i H T.
Kr=[[sum((-I[i][j]*int(T[j,l]) for j in range(32)),Q(0)) for l in range(2)] for i in range(32)]
Ki=[[sum((R[i][j]*int(T[j,l]) for j in range(32)),Q(0)) for l in range(2)] for i in range(32)]
# reduced gamma(q45, q45') = sum_{q34, external, logical} gamma_full
# gamma_full = 1/2 [(iHV5) Z V5^dagger + V5 Z (iHV5)^dagger]
gamma_re=[[Q(0) for _ in range(2)] for _ in range(2)]
gamma_im=[[Q(0) for _ in range(2)] for _ in range(2)]
for a in range(2):
 for b in range(2):
  for external in range(8):
   for c in range(2):
    i=(a<<4)+(b<<3)+external
    for ell in range(2):
     s=1 if ell==0 else -1
     # terms A[i] conj(V[j]) + V[i] conj(A[j]); A= (Kr+iKi)/4, V=T/4
     gamma_re[b][c] += Q(s,32)*(Kr[i][ell]*int(T[(a<<4)+(c<<3)+external,ell])+int(T[i,ell])*Kr[(a<<4)+(c<<3)+external][ell])
     gamma_im[b][c] += Q(s,32)*(Ki[i][ell]*int(T[(a<<4)+(c<<3)+external,ell])-int(T[i,ell])*Ki[(a<<4)+(c<<3)+external][ell])
# beta_Ia = 1/4 Tr(sigma_a gamma), gamma = Tr_q34 Gamma
bX=(gamma_re[0][1]+gamma_re[1][0])/4
bY=(gamma_im[1][0]-gamma_im[0][1])/4 # sign check: Tr(Y gamma) = i(gamma01 - gamma10), i(2i Im(gamma01))=-2 Im; yes gamma10_im-gamma01_im /4
bZ=(gamma_re[0][0]-gamma_re[1][1])/4
B11=16*(bY*bY+bZ*bZ)
B22=16*(bX*bX+bY*bY)
root=p.parent
x=np.load(root/'s55_frozen_control.npy')
x1=Q(float(x[1]));x2=Q(float(x[2]))
W=B11*x1*x1+B22*x2*x2
f=lambda q: f'{q.numerator}/{q.denominator}'
report={'input_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'beta_I_X_exact':f(bX),'beta_I_Y_exact':f(bY),'beta_I_Z_exact':f(bZ),
        'beta_I_X_decimal':float(bX),'beta_I_Y_decimal':float(bY),'beta_I_Z_decimal':float(bZ),
        'B_11_exact':f(B11),'B_22_exact':f(B22),
        'B_11_decimal':float(B11),'B_22_decimal':float(B22),
        'positive_definite_active_2x2_exact':bool(B11>0 and B22>0),
        'rank_2_when_13x13_B_is_diag_1_2':bool(B11>0 and B22>0),
        'W_at_selected_binary64_x_exact':f(W), 'W_at_selected_binary64_x_decimal':float(W),
        'caveat':'Exact dyadic beta and B scalar arithmetic. Structural identity B=diag(0,B11,B22,0,..) uses the separate exact Clifford/erasure algebra and local-source factorisation, which must be audited independently.'}
out=Path(str(Path(__file__).resolve().parent/'square_pauli_exact_dyadic.json'))
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if 'exact' not in k},indent=2))
