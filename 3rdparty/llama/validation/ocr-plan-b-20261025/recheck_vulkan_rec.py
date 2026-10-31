import json,os,subprocess,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[2]; build=repo/'build_llama_production_20261023'
scratch=build/'Testing/ocr-cudagraphs-recheck'
base=repo/'build_app/Testing/ocr-storage-reference'
fixture=Path.home()/'cloudViewer_data/extract/ocr_fixtures'
rows=[]
for quant in ['q8_0']:
 model=Path.home()/'cloudViewer_data/extract/ocr_models'/f'ppocr-rec-ch-{quant}.gguf' if quant=='f16' else Path.home()/'cloudViewer_data/extract/ocr_models/ppocr-rec-ch-q8_0.gguf'
 report=scratch/f'vulkan-rec-{quant}.json'
 args=dict(source_root=base/'public-q8-gpu-truth-vpx8osmk/extract',source_pins=repo/'core/AICore/tools/ocr_paddle_sources.json',source_python=base/'ocr-paddle-clean-gzzutrtn/bin/python',source_gpu_python=base/'ocr-paddle-gpu-minimal-4s9ttxya/bin/python',source_role='rec_ch',model=model,input=fixture/'rec_ch_input.npy',input_sha256='634b6c42cb75bb2fa93ae6a24a4985ff6a7871c1213fc3fb188ff6e95a56682e',reference=fixture/'rec_ch_reference.npy',reference_sha256='6ae5c9d54a2206ae31f9a67212ff6d4893aafe3aaa7c78251bd1a5b4d95cef05',probe=build/'bin/aicore_tests/test_ocr_runtime',backend='vulkan',report=report,warmups=2,runs=10,threads=4,measurement_profile='release',gpu_idle_timeout=60,max_diff=.005)
 command=[sys.executable,str(repo/'core/AICore/scripts/run_ocr_paddle_probe.py')]
 for key,value in args.items():command.extend(['--'+key.replace('_','-'),str(value)])
 with report.with_suffix('.log').open('w') as log:result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
 rows.append(dict(quant=quant,exit_code=result.returncode,report=str(report)))
 print(rows[-1],flush=True)
(scratch/'vulkan-rec-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(any(x['exit_code'] for x in rows))
