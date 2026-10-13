"""S56 rational checker: frozen-input hashes, archived Gram error enclosures,
and conservative Banach fixed-point inequalities.

PROOF SCOPE: independently recomputes rational inequalities *from the archived
Gram centre and archived physical-error certificates*. Does not regenerate the
underlying exact integer physical Gram calculations or prove those input
error certificates correct from scratch. The archive is a prerequisite.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
import sympy as sp

R=Path(__file__).resolve().parent
c=json.loads((R/'s56_frozen_physical_gram_certificate.json').read_text())
for name,file in [('input_frozen_H_sha256','s55_frozen_binary64_weighted_H.npy'),('input_archived_8_grams_sha256','s55_resources_grams.npz'),('input_archived_witness_sha256','s55_validated_witness_gram.npy')]:
    assert hashlib.sha256((R/file).read_bytes()).hexdigest()==c[name]
assert c['status']=='CERTIFIED' and len(c['results'])==9
for k,v in c['results'].items():
    gap=F(v.get('exact_Q_model_vs_archived_max_gap',v.get('exact_mixed_Q39_Q32_model_vs_archive_max_gap',v.get('exact_Q32_witness_vs_archive_gap','0'))))
    extra=F(v.get('frozen_dyadic_H_to_Q_error_upper',v.get('frozen_H_to_exact_CI_upper',v.get('frozen_H_to_exact_source_witness_upper','0'))))
    upper=F(v['total_strict_error_upper'])
    assert gap>=0 and extra>=0 and upper>=gap+extra, (k,upper,gap,extra)
    assert upper<F(1,100000000),(k,upper)
print('PASS S56 all nine certificate upper bounds below 1e-8',flush=True)

z=np.load(R/'s55_resources_grams.npz')
meta=json.loads((R/'s55_resource_selected_before_geometry.json').read_text())
keys=meta['diagnostics'];assert len(keys)==8 and len(set(keys))==8
x=np.load(R/'s55_frozen_control.npy',allow_pickle=False)
assert len(x)==13
# Table III's signed integers, 2^-60 denominator: this makes the published
# centre *literal*, not an implicit numerical input generated during checking.
ns=[1153347777385426176,-14331272386324468,16957741683710130,
3223495552357277,-12331965103681522,4932532582426477,
5032506806642781,-7101093299867069,-5025445579681072,
8161958593594992,5723992090989284,-13818253249840658,
8149529531088830]
X=sp.Matrix([sp.Rational(n,2**60) for n in ns]);
assert all(sp.Rational(*float(x[j]).as_integer_ratio())==X[j] for j in range(13))
rat=lambda y:sp.Rational(*float(y).as_integer_ratio())
G=[sp.Matrix(13,13,lambda i,j:rat(z[k][i,j])) for k in keys]
piv=[0,9,7,3,6,4,2,8,5]
free=sorted(set(range(13))-set(piv)); assert free==[1,10,11,12]
J=sp.Matrix([[2*(g*X)[j] for j in piv] for g in G]+[[1]*9]);assert J.det(method='domain-ge')!=0
A=J.inv();M=max(sum(abs(A[i,j]) for j in range(9)) for i in range(9))
res=sp.Matrix([(X.T*g*X)[0]-g[0,0] for g in G]+[sum(X)-1])
h=max(abs(v) for v in A*res)
s=sum(abs(v) for v in X)
L=2*max(sum(abs(g[i,j]) for i in piv for j in piv) for g in G)
r=sp.Rational(1,100000);delta=sp.Rational(1,100000000)
q=M*(L*r+18*delta*(s+9*r)); d=h+M*delta*(s*s+1)
assert q<1 and d+q*r<r
# Nontrivial: candidate free coordinate 1 != zero.
assert X[1] != 0
print('PASS S56 fixed-point chart',flush=True)
print('  inverse inf norm',float(M),'radius',float(r),'q',float(q),'self-map',float(d+q*r),flush=True)
print('SCOPE: input physical Gram envelopes are archived, NOT regenerated here',flush=True)
