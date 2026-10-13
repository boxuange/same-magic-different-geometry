"""Both nested square-network Pauli erasure maps, checked in exact integer arithmetic.

Patches are reconstructed using the independent integer stabiliser routine in
s59_exact_source_transfer.py. Tests each matter-identity Pauli basis element.
"""
from pathlib import Path
import numpy as np, itertools, json
from functools import reduce
from s59_exact_source_transfer import patch, X, Z, I
ROOT=Path(__file__).resolve().parent
P235,_=patch((2,3,5))
P25,_=patch((2,5))
P5,_=patch((5,))
assert np.array_equal(P25.T@P25,16*np.eye(32,dtype=np.int64))
assert np.array_equal(P5.T@P5,4*np.eye(8,dtype=np.int64))
# With exterior leg counts: vertex 2 has two, 3 two, 5 three.
# Arrange rows of 235 as (vertex3 physical, remaining vertex2+5 physical).
blocks235=P235.reshape(4,4,8,128).transpose(1,0,2,3).reshape(4,32,128)
Knum=np.stack([P25.T@block for block in blocks235])
assert np.array_equal(sum(k.T@k for k in Knum),1024*np.eye(128,dtype=np.int64))
# On 25->5, vertex2 has 2 physical boundary qubits, vertex5 has 3.
blocks25=P25.reshape(4,8,32)
Lnum=np.stack([P5.T@block for block in blocks25])
assert np.array_equal(sum(k.T@k for k in Lnum),64*np.eye(32,dtype=np.int64))
R=X@Z
paulis={'I':I,'X':X,'Z':Z,'R':R}
def pauli(word):return reduce(np.kron,(paulis[c] for c in word))
checked1=0;kept1=0
for q0,q1,q2,q3 in itertools.product('IXZR',repeat=4):
 A=pauli('III'+q0+q1+q2+q3)
 got=sum(k@A@k.T for k in Knum)
 keep=(q1=='I' and q2=='I')
 expect=4096*pauli('II'+q0+'I'+q3) if keep else np.zeros((32,32),dtype=np.int64)
 if not np.array_equal(got,expect):
  raise AssertionError(('235->25',q0,q1,q2,q3,np.max(abs(got-expect))))
 checked1+=1;kept1+=int(keep)
checked2=0;kept2=0
for qa,qb,qc in itertools.product('IXZR',repeat=3):
 A=pauli('II'+qa+qb+qc)
 got=sum(k@A@k.T for k in Lnum)
 keep=(qa=='I' and qb=='I')
 expect=256*pauli('I'+qc+'I') if keep else np.zeros((8,8),dtype=np.int64)
 if not np.array_equal(got,expect):raise AssertionError(('25->5',qa,qb,qc))
 checked2+=1;kept2+=int(keep)
assert (checked1,kept1,checked2,kept2)==(256,16,64,4)
report={'status':'PASS','integer_exact':True,
'P235_gram':'64 I128','P25_gram':'16 I32','P5_gram':'4 I8',
'235_to_25':{'Kraus':'K_s = P25.T (P235)_s / 32','basis_tested':checked1,
 'basis_retained':kept1,'basis_erased':checked1-kept1,
 'identity':'Phi(I8 tensor P0 P1 P2 P3) = 4 delta(P1,I) delta(P2,I) (I4 tensor P0 tensor I2 tensor P3)'},
'25_to_5':{'Kraus':'L_s = P5.T (P25)_s / 8','basis_tested':checked2,
 'basis_retained':kept2,'basis_erased':checked2-kept2,
 'identity':'Phi(I4 tensor P12 P23 P45) = 4 delta(P12,I) delta(P23,I) (I2 tensor P45 tensor I2)'},
'max_integer_residual':0}
(ROOT/'s59_nested_erasure.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
