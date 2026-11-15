import hashlib, json, os, re, subprocess
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
build=repo/'build_llama_production_20261023'
p=build/'Testing/ocr-im2col-85e55y1b'
before=p/'before'
old={ (r['model'],r['batches']):r for r in json.loads((p/'im2col-model-regression.json').read_text()) if r['backend']=='vulkan'}
rows=[]
for role in ['det','rec-ch','rec-en']:
 for dtype in ['f32','f16','q8_0']:
  for batches in ([1] if role=='det' else [1,3]):
   model=f'ppocr-{role}-ch-{dtype}.gguf' if role=='det' else f'ppocr-{role}-{dtype}.gguf'
   key=f'vulkan-{role}-{dtype}-n{batches}-spatial'
   binary=p/(key+'.bin')
   command=[str(build/'bin/aicore_tests/test_ocr_runtime'), str(Path.home()/'cloudViewer_data/extract/ocr_fixtures'),str(Path.home()/'cloudViewer_data/extract/ocr_models'/model), 'det' if role=='det' else 'rec', 'Vulkan', '-1', 'dump='+str(binary),'1.0','2','10','4']
   env=os.environ.copy();env['LD_LIBRARY_PATH']=str(build/'bin')+':/usr/local/cuda/lib64';env['AICORE_OCR_PROBE_TILE']=str(batches)
   with (p/(key+'.log')).open('w') as log:
    result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
   lines=(p/(key+'.log')).read_text().splitlines()
   metrics=[json.loads(s) for s in lines if s.startswith('{') and 'p50_ms' in s]
   digest=hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else ''
   prior=old[model,batches]
   row=dict(backend='vulkan',model=model,batches=batches,command=command,exit_code=result.returncode,metrics=metrics,sha256=digest,before=prior['before'],bitwise_equal=digest==prior['before']['sha256'])
   rows.append(row);(p/'vulkan-im2col-model-regression.json').write_text(json.dumps(rows,indent=2)+'\n')
   print(model,batches,result.returncode,row['bitwise_equal'],metrics,flush=True)
raise SystemExit(0 if all(r['exit_code']==0 and r['bitwise_equal'] for r in rows) else 1)
