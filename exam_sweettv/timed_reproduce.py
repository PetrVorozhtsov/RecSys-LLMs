import json, subprocess, sys, time
from pathlib import Path
root=Path(__file__).parent
start=time.perf_counter()
folder=root/'outputs_reproduction'
with (root/'reproduction.log').open('w',encoding='utf-8') as log:
    completed=subprocess.run([sys.executable,str(root/'run.py'),'--data-dir',str(root/'data'),'--out-dir',str(folder)],stdout=log,stderr=subprocess.STDOUT)
result={'returncode':completed.returncode,'pipeline_seconds':time.perf_counter()-start,'identical':{}}
if completed.returncode==0:
    for name in ['submission.csv','validation_results.json','audit.json','examples.json']:
        result['identical'][name]=(folder/name).read_bytes()==(root/'outputs'/name).read_bytes()
(root/'outputs/reproduction_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
