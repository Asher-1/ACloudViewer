#!/usr/bin/env python3
"""Compare public ABI output with pinned upstream native and official PyTorch.

Truth is generated outside the source tree from official checkpoints. Native
baseline timings bracket the candidate on the same host, weights, input, math
mode and thread count. No C++ task internals are imported by the probe.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
from pathlib import Path
import re
import statistics
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace
import numpy as np
SNAPSHOT = Path(__file__).resolve().parents[2] / 'src/tasks/relation/UPSTREAM.json'
REVISION = json.loads(SNAPSHOT.read_text())['commit']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def metrics(output):
    values = {k: (int(v) if k == 'fingerprint' else float(v)) for k,v in re.findall(r'^([a-zA-Z_0-9]+)=([0-9.eE+-]+)$', output, re.M)}
    for key, sequence in re.findall(r'^([a-zA-Z_0-9]*iteration_ms)=([^\n]+)$', output, re.M):
        values[key] = [float(value) for value in sequence.split(',')]
    return values


def timing_distribution(candidate, before, after, repeats, text=False):
    key = 'e2e_iteration_ms' if text else 'forward_iteration_ms'
    series = [candidate[key], before['iteration_ms'], after['iteration_ms']]
    if any(len(values) != repeats or
           any(not math.isfinite(v) or v <= 0 for v in values)
           for values in series):
        raise ValueError('invalid or missing timed iteration samples')
    result = {}
    for percentile, statistic in (
            ('p50', statistics.median),
            ('p95', lambda values: sorted(values)[math.ceil(len(values)*.95)-1])):
        actual = statistic(series[0])
        baseline = statistics.mean(statistic(values) for values in series[1:])
        result['candidate_'+percentile+'_ms'] = actual
        result['upstream_'+percentile+'_ms'] = baseline
        result['performance_'+percentile+'_ratio'] = actual/baseline
    return result


def summarize_rows(rows):
    """Expose the worst sample for every accuracy and performance gate."""
    result={'latency_ms':statistics.median(r['latency_ms'] for r in rows)}
    for key, value in rows[0].items():
        if isinstance(value,(int,float)) and key not in ('sample','latency_ms'):
            values=[r[key] for r in rows]
            lower_is_worse=key.endswith(('_agreement','_recall','_jaccard'))
            result[key]=min(values) if lower_is_worse else max(values)
    return result


def parity_failures(rows, text=False, performance_limit=1.05):
    checks = [('native_max_diff', '<=', .0002 if text else .001),
              ('performance_ratio', '<=', performance_limit)]
    if text:
        checks.append(('official_max_diff', '<=', .003))
    else:
        checks += [('native_pair_max_diff', '<=', .001),
                   ('native_top1_agreement', '>=', 1),
                   ('native_pair_set_recall', '>=', 1),
                   ('native_pair_set_jaccard', '>=', 1),
                   ('official_mean_error_excess', '<=', 1e-5),
                   ('official_top1_drop', '<=', 1e-6),
                   ('official_pair_recall_drop', '<=', 1e-6)]
    failures = []
    for row in rows:
        row_checks = checks + ([('typed_score_max_diff', '<=', 1e-5)]
                              if 'typed_score_max_diff' in row else [])
        for key, op, limit in row_checks:
            value = row.get(key, float('nan'))
            passed = math.isfinite(value) and (
                value <= limit if op == '<=' else value >= limit)
            if not passed:
                failures.append(
                    f"sample {row['sample']}: {key}={value:.9g} requires {op} {limit:.9g}")
    return failures


def timing_auditor(backend):
    if backend == 'cpu' or shutil.which('nvidia-smi') is None:
        return None
    path=Path(__file__).resolve().parents[1]/'tagging/upstream_metrics_probe.py'
    spec=importlib.util.spec_from_file_location('aicore_tagging_timing_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

    def compute_processes():
        output=subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
            text=True)
        return {int(pid) for pid in output.splitlines() if pid.strip().isdigit()}

    evaluation=SimpleNamespace(ROOT=Path.cwd(),gpu_process_ids=compute_processes)
    module.include_graphics_audit(evaluation)
    return module,evaluation


def run(command, folder, label, env, auditor=None):
    if auditor is not None:
        module,evaluation=auditor
        log=folder/(label+'.log')
        module.execute_audited(command,log,evaluation,set(),env=env)
        return metrics(log.read_text().split('\n--- stderr ---\n',1)[0])
    result = subprocess.run(list(map(str,command)), env=env, capture_output=True, text=True, check=False)
    (folder/(label+'.log')).write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError(f'{label} exited {result.returncode}: '+(result.stdout+result.stderr)[-3000:])
    return metrics(result.stdout)


def load_raw(path):
    b=path.read_bytes();magic,k,v=struct.unpack_from('<8sII',b)
    if magic!=b'RAFOv1\0\0' or len(b)!=16+k*(16+4*v):raise ValueError('invalid raw output')
    out={key:np.frombuffer(b,'<f4' if key=='pairs' else '<i4',k,16+j*4*k).copy() for j,key in enumerate(('valid','sub','obj','pairs'))}
    out['logits']=np.frombuffer(b,'<f4',k*v,16+16*k).reshape(k,v).copy();return out


def compare(reference,candidate,exact_pairs=False):
    def keys(r): return {(int(r['sub'][i]),int(r['obj'][i])):i for i in np.flatnonzero(r['valid']) if r['sub'][i]!=r['obj'][i]}
    a,b=keys(reference),keys(candidate)
    if exact_pairs and a.keys()!=b.keys(): raise ValueError('candidate pair set differs from native reference')
    matched=sorted(a.keys() & b.keys())
    if not matched: raise ValueError('no matched object pairs')
    ia=[a[k] for k in matched];ib=[b[k] for k in matched]
    x,y=reference['logits'][ia],candidate['logits'][ib]
    if x.shape!=y.shape or not np.isfinite(y).all():raise ValueError('invalid candidate logits')
    error=np.abs(x.astype(np.float64)-y)
    pair_error=np.abs(reference['pairs'][ia].astype(np.float64)-candidate['pairs'][ib])
    return {'max_diff':float(error.max()),'mean_diff':float(error.mean()),
            'pair_mean_diff':float(pair_error.mean()),'pair_max_diff':float(pair_error.max()),
            'top1_agreement':float(np.mean(x.argmax(-1)==y.argmax(-1))),
            'pair_set_recall':len(matched)/len(a),'pair_set_jaccard':len(matched)/len(a.keys() | b.keys())}


def compare_edges(reference,candidate):
    if 'predictions' in reference and 'relations' not in reference:
        reference=dict(reference);reference['relations']=reference['predictions']
    metrics={}
    for stream in ('relations','spatial','semantic'):
        if stream not in reference and stream not in candidate:continue
        def keyed(rows):return {(r['subject'],r['object'],r['predicate_id']):r for r in rows}
        a,b=keyed(reference.get(stream,[])),keyed(candidate.get(stream,[]))
        matched=a.keys() & b.keys()
        if a.keys()!=b.keys():raise ValueError('typed '+stream+' edge set differs from upstream')
        if any(a[k]['predicate']!=b[k]['predicate'] for k in matched):
            raise ValueError('predicate name differs from upstream')
        metrics[stream+'_pair_set_recall']=len(matched)/len(a) if a else 1.0
        metrics[stream+'_pair_set_jaccard']=len(matched)/len(a.keys() | b.keys()) if a or b else 1.0
        metrics[stream+'_predicate_agreement']=sum(
            a[k]['predicate_id']==b[k]['predicate_id'] for k in matched)/len(matched) if matched else 1.0
        score_error=[abs(a[k]['score']-b[k]['score']) for k in matched]
        metrics[stream+'_score_max_diff']=max(score_error,default=0.0)
        metrics[stream+'_score_mean_diff']=statistics.mean(score_error) if score_error else 0.0
    metrics['score_max_diff']=max((v for k,v in metrics.items() if k.endswith('_score_max_diff')),default=0.0)
    return metrics


def edge_top1_agreement(edges, reference):
    rows={(int(reference['sub'][i]),int(reference['obj'][i])):i
          for i in np.flatnonzero(reference['valid'])
          if reference['sub'][i]!=reference['obj'][i]}
    values=[]
    for edge in edges:
        i=rows.get((int(edge['subject']),int(edge['object'])))
        if i is not None:
            values.append(int(edge['predicate_id'])==int(reference['logits'][i].argmax()))
    if not values:raise ValueError('typed output has no pairs covered by framework truth')
    return float(np.mean(values))


def upstream_build_dir(upstream, backend):
    build=upstream/f'cpp_ggml/build-{backend}'
    if build.is_symlink() or not build.is_dir():
        raise RuntimeError(f'upstream {backend} build directory is missing or is a symlink: {build}')
    cache=build/'CMakeCache.txt'
    if not cache.is_file():
        raise RuntimeError(f'upstream {backend} build has no CMakeCache.txt: {build}')
    home=re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$',cache.read_text(),re.M)
    expected=(upstream/'cpp_ggml').resolve()
    if home is None or Path(home.group(1)).resolve()!=expected:
        raise RuntimeError(f'upstream {backend} build was configured from a different source tree: {build}')
    binary=build/'relateanything-ggml'
    if binary.is_symlink() or not binary.is_file():
        raise RuntimeError(f'upstream {backend} executable is missing or is a symlink: {binary}')
    return build


def verify_upstream(upstream, binary, env):
    snapshot=json.loads(SNAPSHOT.read_text())
    revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if revision!=snapshot['commit']:raise RuntimeError('upstream revision differs from recorded engineering baseline')
    for relative,expected in snapshot['files'].items():
        if digest(upstream/relative)!=expected:
            raise RuntimeError('upstream source differs from recorded snapshot: '+relative)
    info=json.loads(subprocess.check_output([str(binary),'--build-info'],env=env,text=True))
    if info['ggml_commit']!=snapshot['ggml_commit']:
        raise RuntimeError('upstream binary uses a different ggml revision')
    for source,expected in info['source']['files'].items():
        if digest(source)!=expected:raise RuntimeError('upstream build is stale: '+source)
    return revision,info


def loaded_libraries(binary, env):
    if platform.system()!='Linux':return {}
    output=subprocess.check_output(['ldd',str(binary)],env=env,text=True)
    paths=re.findall(r'=> (\S+) \(',output)
    return {p:digest(p) for p in paths if Path(p).name.startswith(
        ('libAICore','libggml','libcublas','libcudart'))}


def runtime_environments(env, build, backend):
    candidate=env.copy();native=env.copy()
    for variable in ('GGML_BACKEND_PATH','LD_LIBRARY_PATH','DYLD_LIBRARY_PATH'):
        native.pop(variable,None)
    if backend=='cuda':
        cache=(build/'CMakeCache.txt').read_text()
        paths=[]
        for name in ('CUDA_cudart_LIBRARY','CUDA_cublas_LIBRARY','CUDA_cublasLt_LIBRARY'):
            match=re.search(r'^'+name+r':FILEPATH=(.*)$',cache,re.M)
            if match is None or not Path(match.group(1)).is_file():
                raise RuntimeError('upstream CUDA runtime library is missing: '+name)
            folder=str(Path(match.group(1)).resolve().parent)
            if folder not in paths:paths.append(folder)
        # Match the toolkit recorded by the native build. Removing every
        # library path can silently select an older system cuBLAS with the
        # same soname and change both arithmetic and performance.
        native['LD_LIBRARY_PATH']=os.pathsep.join(paths)
        candidate['LD_LIBRARY_PATH']=os.pathsep.join(
            paths+([candidate['LD_LIBRARY_PATH']] if candidate.get('LD_LIBRARY_PATH') else []))
    return candidate,native


def backend_modules(folder):
    paths=set()
    for log in folder.glob('*/candidate.log'):
        paths.update(re.findall(r'loaded \S+ backend from (\S+)',log.read_text()))
    return {p:digest(p) for p in sorted(paths)}


def official_reference(upstream, family, sample, catalog, directory, text=False, texts=None, custom=False, masks=None, scores=None, image_override=None):
    sys.path.insert(0,str(upstream))
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assets=upstream/'cpp_ggml/models/pytorch'/family
    checkpoint=assets/('text_student.pt' if text else 'model.pth')
    checkpoint_digest=digest(checkpoint)
    image_path = Path(image_override) if image_override is not None else catalog.parent/sample['image']
    identity={'upstream_revision':REVISION,'checkpoint_sha256':checkpoint_digest,
              'torch_version':torch.__version__,'catalog_sha256':digest(catalog),
              'sample_image':sample['image'] if not text else None,
              'sample_image_sha256':digest(image_path) if not text else None,
              'texts':texts,'custom':custom,
              'masks_sha256':hashlib.sha256(masks.tobytes()).hexdigest() if masks is not None else None,
              'scores':scores.tolist() if scores is not None else None,
              'generator_sha256':digest(Path(__file__)),'cuda':torch.cuda.is_available()}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=directory.parent.parent/'framework-references';cache.mkdir(exist_ok=True)
    cached=cache/(key+'.npz');provenance=cache/(key+'.json')
    if cached.exists() and provenance.exists():
        saved=json.loads(provenance.read_text())
        if saved['identity']==identity and saved['reference_sha256']==digest(cached):
            with np.load(cached,allow_pickle=False) as data:reference={k:data[k].copy() for k in data.files}
            (directory/'reference-provenance.json').write_text(json.dumps(saved,indent=2))
            return reference
    if text:
        from relsgg.text.student import PredicateTextStudent
        checkpoint=assets/'text_student.pt'
        model=PredicateTextStudent.from_checkpoint(str(checkpoint),device='cpu')
        # The native API ensembles these three prompts before normalizing.
        # Compare the same public encoding contract in the framework.
        templates=['{p}', 'one object is {p} another object',
                   'a photo of something {p} something']
        values=torch.nn.functional.normalize(sum(
            model.encode_texts([template.format(p=p) for p in texts])
            for template in templates), dim=-1).numpy().astype(np.float32)
        reference={'values':values}
    else:
        from relsgg import RelateAnything
        from PIL import Image
        checkpoint=assets/'model.pth'
        model=RelateAnything.from_checkpoint(str(checkpoint),predicates=texts if custom else None,device='cuda' if torch.cuda.is_available() else 'cpu',full_vocabulary=not custom)
        captured={};hook=model.model.register_forward_hook(lambda m,i,o:captured.update(out=o))
        boxes=np.asarray([o['box_xyxy'] for o in sample['objects']],np.float32)
        with Image.open(image_path) as image:model.predict(image.convert('RGB'),boxes,masks=masks,box_scores=scores)
        hook.remove();out=captured['out'];reference={k:out[v][0].detach().float().cpu().numpy() for k,v in (('logits','logits'),('pairs','pair_logits'),('sub','sub_idx'),('obj','obj_idx'),('valid','valid_mask'))}
    np.savez(directory/'official.npz',**reference)
    np.savez(cached,**reference)
    saved={'identity':identity,'reference_sha256':digest(cached),'reference_artifact':str(cached)}
    provenance.write_text(json.dumps(saved,indent=2))
    (directory/'reference-provenance.json').write_text(json.dumps(saved,indent=2))
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return reference


def _reference_worker(channel, args, kwargs):
    try:
        channel.send((True, official_reference(*args, **kwargs)))
    except Exception:
        channel.send((False, traceback.format_exc()))
    finally:
        channel.close()


def isolated_official_reference(*args, **kwargs):
    # Framework truth must release its CUDA context before either timed native
    # or candidate process starts. A spawned process also keeps PyTorch thread
    # pools and loaded libraries out of the host timing environment.
    context=multiprocessing.get_context('spawn')
    receiver,sender=context.Pipe(duplex=False)
    process=context.Process(target=_reference_worker,args=(sender,args,kwargs))
    process.start();sender.close()
    try:
        successful,value=receiver.recv()
        process.join()
        if not successful:
            raise RuntimeError('official framework reference failed:\n'+value)
        if process.exitcode:
            raise RuntimeError('official framework reference exited '+str(process.exitcode))
        return value
    except EOFError as error:
        raise RuntimeError('official framework reference terminated without output') from error
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate();process.join()


def build_native_text_probe(upstream, backend, folder, env):
    folder=folder.resolve()
    build=upstream_build_dir(upstream,backend)
    flags=(build/'CMakeFiles/relateanything-ggml.dir/flags.make').read_text()
    includes=re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M).group(1)
    cflags=re.search(r'^CXX_FLAGS = (.*)$',flags,re.M).group(1)
    link=shlex.split((build/'CMakeFiles/relateanything-ggml.dir/link.txt').read_text())
    source=Path(__file__).with_name('native_text_probe.cpp').resolve()
    obj=folder/'native-text.o';binary=folder/'native-text-probe'
    commands=[[link[0],*shlex.split(includes),*shlex.split(cflags),'-c',str(source),'-o',str(obj)]]
    # Preserve the exact upstream linker closure, including its private ggml ABI.
    output_index=link.index('-o')+1;link[output_index]=str(binary)
    link=[str(obj) if token.endswith('src/main.cpp.o') else token for token in link]
    commands.append(link)
    for command in commands:
        result=subprocess.run(command,cwd=build,env=env,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('native text probe build: '+result.stderr[-3000:])
    return binary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--backend',choices=('cpu','cuda','vulkan'),required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeat',type=int,default=5);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--samples',type=int,nargs='+',default=[0,50,60]);p.add_argument('--text',action='store_true')
    p.add_argument('--regions',action='store_true',help='Exercise binary masks and detector confidences')
    p.add_argument('--custom',action='store_true',help='Exercise the matching native predicate student in the relation graph')
    p.add_argument('--student',type=Path);p.add_argument('--decompose',action='store_true')
    p.add_argument('--no-strict',action='store_true',help='Match the upstream CLI default arithmetic mode')
    p.add_argument('--contracts',action='store_true');p.add_argument('--performance-limit',type=float,default=1.05)
    args=p.parse_args();args.threads=args.threads if args.threads>0 else min(8,os.cpu_count() or 1);upstream_text=os.environ.get('AICORE_TEST_RELATE_UPSTREAM')
    if not upstream_text: raise RuntimeError('Set AICORE_TEST_RELATE_UPSTREAM to the pinned RelateAnything-GGML checkout')
    upstream=Path(upstream_text).resolve()
    binary=upstream_build_dir(upstream,args.backend)/'relateanything-ggml';args.output.parent.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix=args.output.stem+'-',dir=args.output.parent))
    env,native_env=runtime_environments(os.environ,upstream_build_dir(upstream,args.backend),args.backend)
    revision,build_info=verify_upstream(upstream,binary,native_env)
    auditor=timing_auditor(args.backend)
    family=args.model.name.split('-text-student')[0] if args.text else args.model.name.rsplit('-',1)[0]
    catalog=json.loads(args.catalog.read_text()); rows=[]
    text_binary=build_native_text_probe(upstream,args.backend,folder,native_env) if args.text else None
    for index in ([0] if args.text else args.samples):
        directory=folder/str(index);directory.mkdir();s=catalog['samples'][index]
        decoded_ppm=None
        if not args.text:
            decoded_ppm=directory/'input.ppm'
            run([args.probe,'--decode-only','--catalog',args.catalog,
                 '--sample',index,'--pixels',decoded_ppm],directory,
                'decode-input',env)
        common=[binary,'--backend',args.backend,'--threads',args.threads]
        if not args.no_strict:common+=['--strict-f32']
        command=[args.probe,'--backend',args.backend,'--threads',args.threads,'--model',args.model,'--warmup',args.warmup,'--repeat',args.repeat,'--output',directory/'candidate.rafo']
        if args.no_strict:command+=['--no-strict']
        texts=None;masks=None;scores=None
        if args.text:
            texts=['above','holding','next to','café','上面','&amp;','a'*150,'']
            (directory/'texts.json').write_text(json.dumps(texts,ensure_ascii=False))
            command+=['--texts',directory/'texts.json','--tokenizer',args.tokenizer]
            native=[text_binary,args.model,args.tokenizer,directory/'texts.json',args.backend,args.threads,args.warmup,args.repeat,directory/'upstream.f32',int(not args.no_strict)]
        else:
            boxes=np.asarray([o['box_xyxy'] for o in s['objects']],np.float32)
            request='RAIP1\n'+json.dumps(str(decoded_ppm.resolve()))+'\n'+str(len(boxes))+'\n'+'\n'.join(' '.join(format(float(v),'.17g') for v in row) for row in boxes)+'\n'
            if args.regions:
                from PIL import Image
                with Image.open(args.catalog.parent/s['image']) as img:w,h=img.size
                masks=np.zeros((len(boxes),h,w),np.uint8)
                for j,(x1,y1,x2,y2) in enumerate(boxes):masks[j,max(0,int(y1)):min(h,int(y2)),max(0,int(x1)):min(w,int(x2))]=1
                masks.tofile(directory/'masks.u8');scores=np.linspace(.3,.99,len(boxes),dtype=np.float32)
                (directory/'scores.json').write_text(json.dumps(scores.tolist()))
                request=request.replace('RAIP1\n','RAIP2\n',1)+'1\n'+' '.join(format(float(v),'.17g') for v in scores)+'\n'+json.dumps(str(directory/'masks.u8'))+'\n'
                command+=['--masks',directory/'masks.u8','--scores',directory/'scores.json']
            (directory/'request.raip').write_text(request)
            run(common+['--preprocess-only','--input',directory/'request.raip','--output',directory/'input.raim'],directory,'preprocess',native_env)
            native=common+['--model',args.model,'--input',directory/'input.raim','--output',directory/'upstream.rafo','--raw-logits','--warmup',args.warmup,'--repeats',args.repeat]
            if args.custom:
                if not args.student or not args.tokenizer:raise ValueError('--custom requires --student and --tokenizer')
                texts=['riding','holding','next to','above']
                (directory/'predicates.json').write_text(json.dumps(texts))
                native+=['--texts',directory/'predicates.json','--text-student',args.student,'--tokenizer',args.tokenizer]
                command+=['--predicate-texts',directory/'predicates.json','--student',args.student,'--tokenizer',args.tokenizer]
            if args.decompose:native+=['--decompose'];command+=['--decompose']
            command+=['--catalog',args.catalog,'--sample',index]
            if args.contracts and index==args.samples[0]: command+=['--contracts']
            if args.contracts or args.decompose or args.custom or args.regions:command+=['--edges',directory/'candidate-edges.json']
        ref=isolated_official_reference(upstream,family,s,args.catalog,directory,text=args.text,texts=texts,custom=args.custom,masks=masks,scores=scores,image_override=decoded_ppm)
        before=run(native,directory,'upstream-before',native_env,auditor)
        candidate=run(command,directory,'candidate',env,auditor)
        after=run(native,directory,'upstream-after',native_env,auditor)
        if args.text:
            x=np.fromfile(directory/'candidate.rafo','<f4').reshape(-1,512);y=np.fromfile(directory/'upstream.f32','<f4').reshape(-1,512)
            row={'native_max_diff':float(np.max(np.abs(x-y))),'official_max_diff':float(np.max(np.abs(x-ref['values']))),'official_mean_diff':float(np.mean(np.abs(x-ref['values']))),'latency_ms':candidate['latency_ms'],'performance_ratio':candidate['latency_ms']/statistics.mean([before['latency_ms'],after['latency_ms']]),'upstream_before_ms':before['latency_ms'],'upstream_after_ms':after['latency_ms']}
        else:
            actual=load_raw(directory/'candidate.rafo');baseline_raw=load_raw(directory/'upstream.rafo')
            native_values=compare(baseline_raw,actual,exact_pairs=True);truth=compare(ref,actual);upstream_truth=compare(ref,baseline_raw)
            baseline=statistics.mean([before['latency_ms'],after['latency_ms']])
            row={**{'native_'+k:v for k,v in native_values.items()},**{'official_'+k:v for k,v in truth.items()},
                'performance_ratio':candidate['forward_mean_ms']/baseline,'latency_ms':candidate['latency_ms'],
                'official_mean_error_excess':max(0.0,truth['mean_diff']-upstream_truth['mean_diff']),
                'official_pair_mean_error_excess':max(0.0,truth['pair_mean_diff']-upstream_truth['pair_mean_diff']),
                'official_pair_max_error_excess':max(0.0,truth['pair_max_diff']-upstream_truth['pair_max_diff']),
                'official_top1_drop':max(0.0,upstream_truth['top1_agreement']-truth['top1_agreement']),
                'official_pair_recall_drop':max(0.0,upstream_truth['pair_set_recall']-truth['pair_set_recall']),
                'upstream_before_ms':before['latency_ms'],'upstream_after_ms':after['latency_ms'],'forward_ms':candidate['forward_mean_ms']}
            if (directory/'candidate-edges.json').exists():
                edge_command=list(native);edge_command.remove('--raw-logits')
                edge_command[edge_command.index('--output')+1]=directory/'upstream-edges.json'
                edge_command[edge_command.index('--warmup')+1]=0;edge_command[edge_command.index('--repeats')+1]=1
                run(edge_command,directory,'upstream-typed-edges',native_env)
                upstream_edges=json.loads((directory/'upstream-edges.json').read_text())
                candidate_edges=json.loads((directory/'candidate-edges.json').read_text())
                typed=compare_edges(upstream_edges,candidate_edges)
                row.update({'typed_'+k:v for k,v in typed.items()})
                if not args.decompose:
                    row['upstream_edge_top1_agreement']=edge_top1_agreement(
                        upstream_edges.get('predictions',[]),ref)
                    row['official_edge_top1_agreement']=edge_top1_agreement(
                        candidate_edges.get('relations',[]),ref)
                    row['official_edge_top1_drop']=max(
                        0.0, row['upstream_edge_top1_agreement']-
                        row['official_edge_top1_agreement'])
        timing_key='e2e_iteration_ms' if args.text else 'forward_iteration_ms'
        row['timing_samples_complete']=timing_key in candidate
        if row['timing_samples_complete']:
            row.update(timing_distribution(candidate,before,after,args.repeat,args.text))
        row['sample']=index;row['candidate']=candidate;row['upstream_before']=before;row['upstream_after']=after;rows.append(row)
    summary=summarize_rows(rows)
    failures=parity_failures(rows,args.text,args.performance_limit)
    report={'schema':1,'metrics':summary,'failures':failures,'fingerprint':str(rows[0]['candidate']['fingerprint']),
            'model_sha256':digest(args.model),'backend':args.backend,'upstream_revision':revision,
            'upstream_binary_sha256':digest(binary),'probe_sha256':digest(args.probe),'catalog_sha256':digest(args.catalog),
            'upstream_snapshot_sha256':digest(SNAPSHOT),'upstream_build_info':build_info,
            'candidate_libraries':loaded_libraries(args.probe,env),
            'upstream_libraries':loaded_libraries(binary,native_env),
            'candidate_backend_modules':backend_modules(folder),
            'gpu_overlap_audit':{'enabled':auditor is not None,
                'scope':'NVIDIA compute and active graphics processes',
                'logs':str(folder)},
            'host':{'system':platform.platform(),'machine':platform.machine(),'cpu_count':os.cpu_count(),
                    'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None},
            'math_mode':'default' if args.no_strict else 'strict_f32','custom_predicates':args.custom,'binary_masks_and_scores':args.regions,
            'decompose':args.decompose,'threads':args.threads,'warmup':args.warmup,'repeats':args.repeat,'artifacts':str(folder),'rows':rows}
    args.output.write_text(json.dumps(report,indent=2))
    print('latency_ms='+str(summary['latency_ms']))
    print('fingerprint='+str(rows[0]['candidate']['fingerprint']))
    for key,value in summary.items():
        if key!='latency_ms':print(key+'='+str(value))
    if failures:
        raise RuntimeError('upstream parity failed: '+'; '.join(failures))
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
