#!/usr/bin/env python3
"""Compact four-stage verification workflow; see README for evidence limits."""
from pathlib import Path
import subprocess, sys, os, json, hashlib
ROOT=Path(__file__).resolve().parent
jobs=[
 ('S39', 'verify_full_frozen_code_s39.py'),
 ('S50', 's50_exact_local_pauli_gram.py'),
 ('S50', 's50_exact_binary64_physical.py'),
 ('S50', 's50_exact_pauli_erasure.py'),
 ('S56', 'verify_s56_certified_chart.py'),
 ('S59/S58', 'square_pauli_exact_dyadic.py'),
 ('S59', 'run_s59.py'),
]
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
report={'verification_scope':'S39/S56 use archived physical Gram enclosures; S50/S59 recompute explicitly stated integer/dyadic microscopic operations', 'steps':[]}
for folder,script in jobs:
    wd=ROOT/folder
    print(f'[{len(report["steps"])+1}/{len(jobs)}] {folder}/{script}',flush=True)
    result=subprocess.run([sys.executable,script],cwd=wd,env=env,text=True,capture_output=True)
    record={'stage':folder,'script':script,'exit_code':result.returncode,'stdout_tail':result.stdout[-1800:],'stderr_tail':result.stderr[-1000:]}
    report['steps'].append(record)
    if result.returncode:
        print(result.stdout,flush=True)
        print(result.stderr,file=sys.stderr,flush=True)
        raise SystemExit(f'FAIL: {folder}/{script}')
    print(' PASS',flush=True)
assert all(x['exit_code']==0 for x in report['steps'])
s59=json.loads((ROOT/'S59/S59_REPLAY_STATUS.json').read_text())
assert s59['status']=='PASS' and s59['step_count']==4
report['status']='PASS'
report['pipeline_steps']=len(jobs)
report['S59_nested_steps']=4
(ROOT/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('ALL PASS — 7 command groups, including 4 S59 sub-checks.',flush=True)
print('LIMIT: S39/S56 physical-error inputs are audited archive certificates, not reconstructed from full exact tensor orbit.',flush=True)
