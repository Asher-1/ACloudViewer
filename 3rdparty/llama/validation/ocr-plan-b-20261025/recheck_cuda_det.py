import json,os,subprocess,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[2]; build=repo/'build_llama_production_20261023'
scratch=build/'Testing/ocr-cudagraphs-recheck'
base=repo/'build_app/Testing/ocr-storage-reference'
fixture=Path.home()/'cloudViewer_data/extract/ocr_fixtures'
rows=[]
for quant in ['f16','q8_0']:
 model=Path.home()/'cloudViewer_data/extract/ocr_models'/f'ppocr-det-ch-{quant}.gguf' if quant=='f16' else base/'public-q8-repaired-7xyhrgpy/extract/pending/ocr_models/ppocr-det-ch-q8_0.gguf'
 report=scratch/f'cuda-det-{quant}.json'
 args=dict(source_root=base/'public-q8-gpu-truth-vpx8osmk/extract',source_pins=repo/'core/AICore/tools/ocr_paddle_sources.json',source_python=base/'ocr-paddle-clean-gzzutrtn/bin/python',source_gpu_python=base/'ocr-paddle-gpu-minimal-4s9ttxya/bin/python',source_role='det_ch',model=model,input=fixture/'det_ch_input.npy',input_sha256='0e9bedae1dc5c6c248f08222c002ecd0cf8027b02b86a69ab82a0639f2a65060',reference=fixture/'det_ch_reference.npy',reference_sha256='4ebf682711f8fd49732ce281041efb004fb9b472d5cf58becc9909384da61eb7',probe=build/'bin/aicore_tests/test_ocr_runtime',backend='cuda',report=report,warmups=2,runs=10,threads=4,measurement_profile='release',gpu_idle_timeout=60,max_diff=.6,mean_diff=.0005)
 command=[sys.executable,str(repo/'core/AICore/scripts/run_ocr_paddle_probe.py')]
 for key,value in args.items():command.extend(['--'+key.replace('_','-'),str(value)])
 with report.with_suffix('.log').open('w') as log:result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
 rows.append(dict(quant=quant,exit_code=result.returncode,report=str(report)))
 print(rows[-1],flush=True)
(scratch/'cuda-det-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(any(x['exit_code'] for x in rows))
