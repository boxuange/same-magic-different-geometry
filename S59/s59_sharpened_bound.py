"""Exact rational witness lower bound within the archived S56 physical Banach pivot chart.
Does not recreate the S56 physical Gram error enclosures or existence certificate;
it only strengthens their witness inequality assuming that certified chart.
"""
from pathlib import Path
from fractions import Fraction as F
import json,numpy as np
ROOT=Path(__file__).resolve().parent
report0=json.loads((ROOT/'s59_exact_clifford_gram.json').read_text())
B1=F(report0['B_11_fraction']); B2=F(report0['B_22_fraction'])
x=np.load(ROOT/'S58/S55/s55_frozen_control.npy')
x1=F(float(x[1])); x2=F(float(x[2])); r=F(1,100000)
# Exact physical-pivot chart of S56 uses free idx=[1,10,11,12] and pivot idx=[0,9,7,3,6,4,2,8,5].
free=[1,10,11,12]
pivots=[0,9,7,3,6,4,2,8,5]
assert 1 in free and 2 in pivots and abs(x2)>r
Wlower=B1*x1*x1 + B2*(abs(x2)-r)**2
Wcenter=B1*x1*x1 + B2*x2*x2
# Strict positive rational bound 2.4209e-6.
threshold=F(24209,10**10)
assert Wlower>threshold
assert Wcenter-Wlower>0
out={'status':'PASS',
'exact_arithmetic':True,
'external_dependency':'Uses S56 physically certified fixed-point pivot chart and 1e-5 pivot radius; does NOT independently reproduce S56 Gram enclosures.',
'free_coordinates':free,'pivot_coordinates':pivots,
'pivot_infinity_radius':str(r),
'x1_fixed_decimal':float(x1),'x2_centre_decimal':float(x2),
'W_centre_decimal':float(Wcenter),
'W_lower_exact_fraction':str(Wlower),
'W_lower_decimal':float(Wlower),
'strict_bound':'W_square(x*) - W_square(e0) > 2.4209e-6',
'zero_reference_from_micro_gram':True,
'comparison_to_existing_S56_bound':'S56 retained bound 2.39671149689e-6; this improves witness only, not resource-matching existence proof.',
'certification_logic':'Because B11 and B22 are exact positive rational coefficients, x1 is free/fixed, and |x2-x2centre|<=1e-5, use W>= B11*x1^2 + B22*(|x2centre|-1e-5)^2.',
'scope':'Fixed physical frozen binary64 input model; the single Z4 response witness; depends on independently previously certified S56 existence chart.'}
(ROOT/'s59_sharpened_bound.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
