"""Exact integer 235->35 stabilizer patch CPTP channel acts as Pauli conditional expectation
on matter-identity, cut-Pauli subalgebra. Full 4^4 Pauli test; no randomness.
"""
import sys,itertools,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
ROOT=HERE
from minimal_bridge_core import bridge_network, integer_graph_contraction

P={'I':np.eye(2,dtype=np.int64),'X':np.array([[0,1],[1,0]],dtype=np.int64),
   'Y':np.array([[0,-1],[1,0]],dtype=np.int64), 'Z':np.diag([1,-1]).astype(np.int64)}
def tensor(label):
 a=np.array([[1]],dtype=np.int64)
 for p in label:a=np.kron(a,P[p])
 return a

def run():
 t0=time.time();results=[]
 with bridge_network():
  p235,_=integer_graph_contraction((2,3,5))
  p35,_=integer_graph_contraction((3,5))
  assert np.array_equal(p235.T@p235,64*np.eye(128,dtype=np.int64))
  assert np.array_equal(p35.T@p35,16*np.eye(32,dtype=np.int64))
  kn=np.stack([p35.T@p235.reshape(4,32,128)[i] for i in range(4)])
  assert np.array_equal(np.einsum('iab,iac->bc',kn,kn,optimize=True),1024*np.eye(128,dtype=np.int64))
  mx=0;failed=[];byclass={}
  for w in itertools.product('IXYZ',repeat=4):
   lab=''.join(w)
   q=tensor(lab)
   o=np.kron(np.eye(8,dtype=np.int64),q)
   # signed permutation row index, no dense O contraction; kn@o[:,j] for each j.
   perm=np.argmax(abs(o),axis=0);sign=o[perm,np.arange(128)]
   assert np.all(np.sum(abs(o),axis=0)==1)
   ans=np.zeros((32,32),dtype=np.int64)
   for k in kn:
    ans+=(k[:,perm]*sign[None,:])@k.T
   # 1/1024 from Kraus; input identity response is 4x output identity.
   target=4096*np.kron(np.eye(8,dtype=np.int64),tensor(lab[2:])) if lab[:2]=='II' else np.zeros((32,32),dtype=np.int64)
   err=int(np.max(np.abs(ans-target)));mx=max(mx,err)
   if err:failed.append((lab,err, int(np.max(abs(ans)))))
   key='left_identity' if lab[:2]=='II' else 'left_nonidentity'
   byclass[key]=byclass.get(key,0)+1
  print('exact_256 Pauli erasure max discrepancy',mx,'failures',failed[:8],'byclass',byclass,'time',time.time()-t0,flush=True)
  out={'scope':'exact 235->35 physical partial trace plus patch decoding; for matter-identity cut Pauli operators only','num_pauli_tested':256,'left_identity_16_survive':byclass['left_identity'],'left_nonidentity_240_erased':byclass['left_nonidentity'],'max_integer_discrepancy':mx,'Kraus_numerator_denominator':32,'success':not failed}
  (HERE/'s50_exact_pauli_erasure_certificate.json').write_text(json.dumps(out,indent=2)+'\n')
  assert not failed
if __name__=='__main__':run()
