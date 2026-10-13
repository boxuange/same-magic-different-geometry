"""Run every new S59 certificate. Original S58 data are included but not recomputed here."""
import subprocess,sys,os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
steps=['s59_exact_source_transfer.py','s59_nested_erasure.py','s59_exact_clifford_gram.py','s59_sharpened_bound.py']
results={}
for name in steps:
 print('RUN',name,flush=True)
 result=subprocess.run([sys.executable,str(ROOT/name)],cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True)
 if result.returncode:
  print(result.stdout, result.stderr)
  raise SystemExit(f'{name} FAILED')
 results[name]={'result':'PASS','sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}
 print('  PASS',flush=True)
for name in ('s59_exact_source_transfer.json','s59_nested_erasure.json','s59_exact_clifford_gram.json','s59_sharpened_bound.json'):
 assert json.loads((ROOT/name).read_text())['status']=='PASS'
report={'status':'PASS','step_count':len(steps),'passed_steps':results,
 'scope':'This S59 replay checks direct integer transfer, both integer nested erasures, Clifford/dyadic rank two, and exact lower bound CONDITIONAL on previously archived S56 certified existence. It does not regenerate S56 physical resource Gram enclosures.'}
(ROOT/'S59_REPLAY_STATUS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
