"""Independent checkpoint accuracy and per-output integration error.

Absolute limits follow the upstream GGML-native G1 contract in
cpp_ggml/scripts/write_acceptance_status.py. The limits apply to the pooled
matched logits and object pairs in the selected input set; a three-source
integration probe is not the upstream 50-image benchmark.
"""
import math

import numpy as np


def pair_rows(output):
    rows = {}
    for index in np.flatnonzero(output['valid']):
        key = (int(output['sub'][index]), int(output['obj'][index]))
        if key[0] == key[1]:
            continue
        if key in rows:
            raise ValueError('duplicate object pair in raw output')
        rows[key] = int(index)
    return rows


def checkpoint_metrics(reference, native, candidate):
    truth_rows, native_rows, candidate_rows = map(
        pair_rows, (reference, native, candidate))
    if native_rows.keys() != candidate_rows.keys():
        raise ValueError('candidate pair set differs from native reference')
    matched = sorted(truth_rows.keys() & candidate_rows.keys())
    if not matched:
        raise ValueError('no pairs matched to independent checkpoint')
    truth_indices = [truth_rows[key] for key in matched]
    native_indices = [native_rows[key] for key in matched]
    candidate_indices = [candidate_rows[key] for key in matched]
    truth = reference['logits'][truth_indices].astype(np.float64)
    baseline = native['logits'][native_indices].astype(np.float64)
    actual = candidate['logits'][candidate_indices].astype(np.float64)
    if (truth.shape != actual.shape or baseline.shape != actual.shape or
            not all(np.isfinite(values).all() for values in
                    (truth, baseline, actual))):
        raise ValueError('invalid checkpoint or inference logits')
    error = np.abs(actual - truth)
    added = np.maximum(error - np.abs(baseline - truth), 0)
    truth_pairs = reference['pairs'][truth_indices].astype(np.float64)
    native_pairs = native['pairs'][native_indices].astype(np.float64)
    actual_pairs = candidate['pairs'][candidate_indices].astype(np.float64)
    if not all(np.isfinite(values).all() for values in
               (truth_pairs, native_pairs, actual_pairs)):
        raise ValueError('invalid checkpoint or inference pair logits')
    pair_added = np.maximum(np.abs(actual_pairs - truth_pairs) -
                            np.abs(native_pairs - truth_pairs), 0)
    return {
        'official_logit_squared_error_sum': float(np.square(error).sum()),
        'official_logit_count': int(error.size),
        'official_predicate_matches': int(np.count_nonzero(
            truth.argmax(-1) == actual.argmax(-1))),
        'official_matched_pair_count': len(matched),
        'official_union_pair_count': len(truth_rows.keys() |
                                         candidate_rows.keys()),
        'official_output_error_excess': float(added.max()),
        'official_pair_output_error_excess': float(pair_added.max()),
    }


def pooled_metrics(rows):
    squared_error = sum(row['official_logit_squared_error_sum'] for row in rows)
    logit_count = sum(row['official_logit_count'] for row in rows)
    matched_count = sum(row['official_matched_pair_count'] for row in rows)
    union_count = sum(row['official_union_pair_count'] for row in rows)
    predicate_matches = sum(row['official_predicate_matches'] for row in rows)
    if min(logit_count, matched_count, union_count) <= 0:
        raise ValueError('empty independent checkpoint comparison')
    return {
        'official_logit_squared_error_sum': squared_error,
        'official_logit_count': logit_count,
        'official_predicate_matches': predicate_matches,
        'official_matched_pair_count': matched_count,
        'official_union_pair_count': union_count,
        'official_logits_rmse': math.sqrt(squared_error / logit_count),
        'official_predicate_top1_agreement': predicate_matches / matched_count,
        'official_pairs_jaccard': matched_count / union_count,
        'official_output_error_excess': max(
            row['official_output_error_excess'] for row in rows),
        'official_pair_output_error_excess': max(
            row['official_pair_output_error_excess'] for row in rows),
    }


def absolute_failures(metrics, rmse_limit, top1_limit, jaccard_limit):
    # Limits are supplied by the owning validation manifest, not inferred from
    # whichever models or outputs happen to be cached.
    checks = [('official_logits_rmse', '<=', rmse_limit),
              ('official_predicate_top1_agreement', '>=', top1_limit),
              ('official_pairs_jaccard', '>=', jaccard_limit),
              ('official_output_error_excess', '<=', 1e-5),
              ('official_pair_output_error_excess', '<=', 1e-5)]
    failures = []
    for key, operator, limit in checks:
        value = metrics.get(key, float('nan'))
        passed = math.isfinite(value) and (value <= limit if operator == '<='
                                          else value >= limit)
        if not passed:
            failures.append(f'{key}={value:.9g} requires {operator} {limit:.9g}')
    return failures
