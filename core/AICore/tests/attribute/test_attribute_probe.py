#!/usr/bin/env python3
"""Public multi-attribute real-model probe; the independent reference is an external, pinned-input original-framework output."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


class Options(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('abi_version', C.c_uint32),
                ('model_path', C.c_char_p), ('device', C.c_char_p), ('threads', C.c_int32)]


class View(C.Structure):
    _fields_ = [('data', C.c_void_p), ('width', C.c_int32), ('height', C.c_int32),
                ('stride', C.c_size_t), ('format', C.c_int)]


class Item(C.Structure):
    _fields_ = [('name', C.c_char_p), ('value', C.c_char_p), ('score', C.c_float),
                ('probability_index', C.c_int32)]


class Result(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('abi_version', C.c_uint32),
                ('label', C.c_char_p), ('width', C.c_int32), ('height', C.c_int32),
                ('attributes', C.POINTER(Item)), ('attribute_count', C.c_size_t),
                ('probabilities', C.POINTER(C.c_float)), ('probability_count', C.c_size_t),
                ('private_owner', C.c_void_p)]


class Preset(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('abi_version', C.c_uint32),
                ('filename', C.c_char_p), ('display_name', C.c_char_p),
                ('label', C.c_char_p), ('source_sha256', C.c_char_p),
                ('probability_count', C.c_size_t), ('attribute_count', C.c_size_t)]


class Timings(C.Structure):
    _fields_ = [('abi_version', C.c_uint32), ('valid_fields', C.c_uint32)] + [
                (key + '_ms', C.c_double) for key in ('preprocess', 'inference', 'postprocess', 'serialization', 'e2e')]


def bind(path):
    lib = C.CDLL(str(path))
    for name, result, parameters in (
            ('default_options', Options, []), ('create', C.c_void_p, [C.POINTER(Options)]),
            ('free', None, [C.c_void_p]), ('ready', C.c_int, [C.c_void_p]),
            ('last_error', C.c_char_p, [C.c_void_p]), ('context_device', C.c_char_p, [C.c_void_p]),
            ('run', C.c_int, [C.c_void_p, C.POINTER(View), C.POINTER(C.POINTER(Result))]),
            ('result_free', None, [C.POINTER(Result)]),
            ('last_pipeline_timings', C.c_int, [C.c_void_p, C.POINTER(Timings)])):
        function = getattr(lib, 'aicore_attribute_' + name)
        function.restype, function.argtypes = result, parameters
    return lib


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def infer(lib, args, image):
    padded = np.zeros((image.shape[0], image.shape[1] + 7, 3), np.uint8)
    padded[:, :image.shape[1]] = image
    view = View(padded.ctypes.data, image.shape[1], image.shape[0], padded.strides[0], 4)
    before = hashlib.sha256(padded.tobytes()).hexdigest()
    options = lib.aicore_attribute_default_options()
    options.model_path, options.device, options.threads = str(args.model).encode(), args.backend.encode(), args.threads
    context = lib.aicore_attribute_create(C.byref(options))
    values, attributes, durations, identities = [], [], [], []
    try:
        if not lib.aicore_attribute_ready(context):
            raise RuntimeError(lib.aicore_attribute_last_error(context).decode())
        device = lib.aicore_attribute_context_device(context).decode()
        if args.backend.lower() not in device.lower():
            raise RuntimeError('the requested backend was not actually used: ' + device)
        label = None
        for index in range(args.warmups + args.repeats):
            output = C.POINTER(Result)()
            if lib.aicore_attribute_run(context, C.byref(view), C.byref(output)) != 0 or not output:
                raise RuntimeError(lib.aicore_attribute_last_error(context).decode())
            try:
                result = output.contents
                if result.struct_size != C.sizeof(Result) or result.abi_version != 1 or (
                        result.width, result.height) != (image.shape[1], image.shape[0]):
                    raise RuntimeError('output ABI/full-image geometry mismatch')
                current = np.ctypeslib.as_array(result.probabilities, shape=(result.probability_count,)).copy()
                attrs = {result.attributes[i].name.decode(): result.attributes[i].value.decode()
                         for i in range(result.attribute_count)}
                if not np.isfinite(current).all() or not attrs:
                    raise RuntimeError('invalid probabilities or empty attributes')
                timing = Timings(1, 0)
                if lib.aicore_attribute_last_pipeline_timings(context, C.byref(timing)) != 0 or timing.valid_fields != 23:
                    raise RuntimeError('missing the common typed pipeline timing')
                if index >= args.warmups:
                    values.append(current)
                    attributes.append(attrs)
                    durations.append(timing.e2e_ms)
                    identities.append(hashlib.sha256(current.tobytes() + json.dumps(attrs, sort_keys=True).encode()).hexdigest())
                    label = result.label.decode()
            finally:
                lib.aicore_attribute_result_free(output)
    finally:
        lib.aicore_attribute_free(context)
    if hashlib.sha256(padded.tobytes()).hexdigest() != before:
        raise RuntimeError('caller pixels were modified')
    return values, attributes, durations, identities, device, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--source-identity', type=Path)
    parser.add_argument('--source-identity-sha256')
    parser.add_argument('--backend', required=True)
    parser.add_argument('--quant', choices=('f32', 'f16', 'q8_0'), required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--raw-out', type=Path)
    parser.add_argument('--json-out', type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.threads < 1:
        parser.error('invalid sampling protocol')
    if digest(args.reference) != args.reference_sha256:
        raise ValueError('independent original-framework output SHA-256 mismatch')
    if bool(args.source_identity) != bool(args.source_identity_sha256):
        parser.error('source identity and its SHA-256 must be provided together')
    source = None
    if args.source_identity:
        if digest(args.source_identity) != args.source_identity_sha256:
            raise ValueError('independent source identity SHA-256 mismatch')
        source = json.loads(args.source_identity.read_text())
        if not source.get('source_revision') or not isinstance(source.get('attributes'), dict) or not source['attributes']:
            raise ValueError('independent source identity lacks a revision/attribute reference')
    reference = np.load(args.reference).reshape(-1)
    if not np.isfinite(reference).all():
        raise ValueError('original-framework output is not finite')
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError('input decoding failed')
    library = args.library.absolute()
    lib = bind(library)
    values, attributes, durations, hashes, device, label = infer(lib, args, image)
    attributes_match = None
    if source is not None:
        lib.aicore_attribute_preset_count.restype = C.c_int
        lib.aicore_attribute_preset_at.argtypes = [C.c_int]
        lib.aicore_attribute_preset_at.restype = C.POINTER(Preset)
        presets = [lib.aicore_attribute_preset_at(i).contents for i in range(lib.aicore_attribute_preset_count())]
        if not any(entry.label.decode() == label and entry.source_sha256.decode() == source.get('source_sha256')
                   for entry in presets) or source.get('family') != label:
            raise ValueError('independent source identity does not match the public task recipe')
        expected = np.asarray(source.get('probabilities'), dtype=np.float32).reshape(-1)
        if not np.array_equal(expected, reference):
            raise ValueError('the attribute reference and the probability reference are not the same source output')
        attributes_match = all(row == source['attributes'] for row in attributes)
    if any(row.shape != reference.shape for row in values):
        raise ValueError('original probability output shape mismatch')
    errors = np.maximum.reduce([np.abs(row.astype(np.float64) - reference) for row in values])
    threshold = {'f32': .001, 'f16': .01, 'q8_0': .02}[args.quant]
    stable = len(set(hashes)) == 1
    ordered = sorted(durations)
    report = {'pass': bool(stable and errors.max() <= threshold and attributes_match is not False), 'stable': stable,
        'attributes_match': attributes_match,
        'source_sha256': source.get('source_sha256') if source else None,
        'source_revision': source.get('source_revision') if source else None,
        'source_identity_sha256': args.source_identity_sha256,
        'maximum_output_error': float(errors.max()), 'per_output_error': errors.tolist(),
        'threshold': threshold, 'quantization': args.quant, 'device': device, 'label': label,
        'attributes': attributes, 'output_count': int(reference.size),
        'e2e_p50_ms': ordered[int(np.ceil(.5 * len(ordered))) - 1],
        'e2e_p95_ms': ordered[int(np.ceil(.95 * len(ordered))) - 1],
        'e2e_samples_ms': durations, 'output_sha256': hashes[0],
        'runtime_sha256': digest(library), 'model_sha256': digest(args.model),
        'image_sha256': digest(args.image), 'reference_sha256': args.reference_sha256,
        'warmups': args.warmups, 'repeats': args.repeats, 'threads': args.threads,
        'public_attribute_api': True,
        'scope': 'numeric/stability agreement between the public task and a pinned original-framework output; when a pinned source identity is supplied all attributes are verified as well, otherwise attribute truth is unverified; not a release qualification'}
    if args.raw_out:
        values[0].tofile(args.raw_out)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
