import csv, json, time, platform, os, ctypes
from pathlib import Path
import run

root=Path(__file__).parent
start=time.perf_counter()
with (root/'data/submission.csv').open(encoding='utf-8-sig',newline='') as f: template=list(csv.reader(f))
with (root/'outputs/submission.csv').open(encoding='utf-8-sig',newline='') as f: submitted=list(csv.reader(f))
assert submitted[0]==template[0]
assert [r[0] for r in submitted[1:]]==[r[0] for r in template[1:]]
assert len(submitted)-1==1260
assert all(len(r[1].split())==len(set(r[1].split()))==5 for r in submitted[1:])
assert run.map5({'u':['A','B','C','D','E']},{'u':['A','C']})==((1+2/3)/2)
ex=json.loads((root/'outputs/examples.json').read_text())
for user,d in ex.items():
    for method,ids in d['predictions'].items():
        assert len(ids)==len(set(ids))==5
        assert not (set(ids)&set(d['history']))
baseline=['10','20','30','40','50']
assert run.top5([],baseline,set())==baseline
# Capture the actual content cold-start discrepancy without changing the submitted model.
meta={s:{'category':'','genres':'','year':'','director':'','actors':''} for s in ['1','2','3','4','5','6']}
empty_content=run.top5(run.cosine_profile({},meta,set(),set(meta)),baseline,set())
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('avail',ctypes.c_ulonglong),('page',ctypes.c_ulonglong),('avail_page',ctypes.c_ulonglong),('virt',ctypes.c_ulonglong),('avail_virt',ctypes.c_ulonglong),('extended',ctypes.c_ulonglong)]
mem=Memory(); mem.length=ctypes.sizeof(mem); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
result={'checks':'passed','submission_rows':1260,'submission_header':submitted[0], 'unique_ids_per_row':5,'example_lists_checked':6,'hand_example_ap5':(1+2/3)/2,'cold_content_example':empty_content,'expected_popularity_fallback':baseline,'verification_seconds':time.perf_counter()-start,'python':platform.python_version(),'cpu':os.environ.get('PROCESSOR_IDENTIFIER','unavailable'),'logical_cpus':os.cpu_count(),'ram_gib':round(mem.total/1024**3,2)}
(root/'outputs/report_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
