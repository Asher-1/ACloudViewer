# ----------------------------------------------------------------------------
# -                        CloudViewer: Asher-1.github.io                    -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 Asher-1.github.io
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
from scipy.spatial import cKDTree
import pytest
import mltest

# skip all tests if the ml ops were not built
pytestmark = mltest.default_marks

# the supported dtypes for the point coordinates
dtypes = pytest.mark.parametrize('dtype', [np.float32, np.float64])

# the GPU only supports single precision float
gpu_dtypes = [np.float32]


@dtypes
@mltest.parametrize.ml
@pytest.mark.parametrize('num_points_queries', [(10, 5), (31, 33), (33, 31),
                                                (123, 345)])
@pytest.mark.parametrize('radius', [0.1, 0.3])
@pytest.mark.parametrize('hash_table_size_factor', [1 / 8, 1 / 64])
@pytest.mark.parametrize('metric', ['L1', 'L2', 'Linf'])
@pytest.mark.parametrize('ignore_query_point', [False, True])
@pytest.mark.parametrize('return_distances', [False, True])
def test_fixed_radius_search(dtype, ml, num_points_queries, radius,
                             hash_table_size_factor, metric, ignore_query_point,
                             return_distances):

    # skip dtype not supported on GPU
    if mltest.is_gpu_device_name(ml.device) and not dtype in gpu_dtypes:
        return

    rng = np.random.RandomState(123)

    num_points, num_queries = num_points_queries

    points = rng.random(size=(num_points, 3)).astype(dtype)
    if ignore_query_point:
        queries = points
    else:
        queries = rng.random(size=(num_queries, 3)).astype(dtype)

    # kd tree for computing the ground truth
    tree = cKDTree(points, copy_data=True)
    p_norm = {'L1': 1, 'L2': 2, 'Linf': np.inf}[metric]
    gt_neighbors_index = tree.query_ball_point(queries, radius, p=p_norm)

    layer = ml.layers.FixedRadiusSearch(metric=metric,
                                        ignore_query_point=ignore_query_point,
                                        return_distances=return_distances)
    ans = mltest.run_op(
        ml,
        ml.device,
        True,
        layer,
        points,
        queries=queries,
        radius=radius,
        hash_table_size_factor=hash_table_size_factor,
    )

    for i, q in enumerate(queries):
        # check neighbors
        start = ans.neighbors_row_splits[i]
        end = ans.neighbors_row_splits[i + 1]
        q_neighbors_index = ans.neighbors_index[start:end]

        gt_set = set(gt_neighbors_index[i])
        if ignore_query_point:
            gt_set.remove(i)
        assert gt_set == set(q_neighbors_index)

        # check distances
        if return_distances:
            q_neighbors_dist = ans.neighbors_distance[start:end]
            for j, dist in zip(q_neighbors_index, q_neighbors_dist):
                if metric == 'L2':
                    gt_dist = np.sum((q - points[j])**2)
                else:
                    gt_dist = np.linalg.norm(q - points[j], ord=p_norm)
                np.testing.assert_allclose(dist, gt_dist, rtol=1e-7, atol=1e-8)


@mltest.parametrize.ml
def test_fixed_radius_search_empty_point_sets(ml):
    rng = np.random.RandomState(123)

    dtype = np.float32
    radius = 1
    hash_table_size_factor = 1 / 64

    # no query points
    points = rng.random(size=(100, 3)).astype(dtype)
    queries = rng.random(size=(0, 3)).astype(dtype)

    layer = ml.layers.FixedRadiusSearch(return_distances=True)
    ans = mltest.run_op(
        ml,
        ml.device,
        True,
        layer,
        points,
        queries=queries,
        radius=radius,
        hash_table_size_factor=hash_table_size_factor,
    )

    assert ans.neighbors_index.shape == (0,)
    assert ans.neighbors_row_splits.shape == (1,)
    assert ans.neighbors_distance.shape == (0,)

    # no input points
    points = rng.random(size=(0, 3)).astype(dtype)
    queries = rng.random(size=(100, 3)).astype(dtype)

    ans = mltest.run_op(
        ml,
        ml.device,
        True,
        layer,
        points,
        queries=queries,
        radius=radius,
        hash_table_size_factor=hash_table_size_factor,
    )

    assert ans.neighbors_index.shape == (0,)
    assert ans.neighbors_row_splits.shape == (101,)
    np.testing.assert_array_equal(np.zeros_like(ans.neighbors_row_splits),
                                  ans.neighbors_row_splits)
    assert ans.neighbors_distance.shape == (0,)


@dtypes
@mltest.parametrize.ml
@pytest.mark.parametrize('batch_size', [2, 3, 8])
@pytest.mark.parametrize('radius', [0.1, 0.3])
@pytest.mark.parametrize('hash_table_size_factor', [1 / 8, 1 / 64])
@pytest.mark.parametrize('metric', ['L1', 'L2', 'Linf'])
@pytest.mark.parametrize('ignore_query_point', [False, True])
@pytest.mark.parametrize('return_distances', [False, True])
def test_fixed_radius_search_batches(dtype, ml, batch_size, radius,
                                     hash_table_size_factor, metric,
                                     ignore_query_point, return_distances):
    # skip dtype not supported on GPU
    if mltest.is_gpu_device_name(ml.device) and not dtype in gpu_dtypes:
        return

    rng = np.random.RandomState(123)

    # create array defining start and end of each batch
    points_row_splits = np.zeros(shape=(batch_size + 1,), dtype=np.int64)
    queries_row_splits = np.zeros(shape=(batch_size + 1,), dtype=np.int64)
    for i in range(batch_size):
        points_row_splits[i + 1] = rng.randint(15) + points_row_splits[i]
        queries_row_splits[i + 1] = rng.randint(15) + queries_row_splits[i]

    num_points = points_row_splits[-1]
    num_queries = queries_row_splits[-1]

    points = rng.random(size=(num_points, 3)).astype(dtype)
    if ignore_query_point:
        queries = points
        queries_row_splits = points_row_splits
    else:
        queries = rng.random(size=(num_queries, 3)).astype(dtype)

    # kd trees for computing the ground truth
    p_norm = {'L1': 1, 'L2': 2, 'Linf': np.inf}[metric]
    gt_neighbors_index = []
    for i in range(batch_size):
        points_i = points[points_row_splits[i]:points_row_splits[i + 1]]
        queries_i = queries[queries_row_splits[i]:queries_row_splits[i + 1]]

        tree = cKDTree(points_i, copy_data=True)
        gt_neighbors_index.extend([
            list(
                tree.query_ball_point(q, radius, p=p_norm) +
                points_row_splits[i]) for q in queries_i
        ])

    layer = ml.layers.FixedRadiusSearch(metric=metric,
                                        ignore_query_point=ignore_query_point,
                                        return_distances=return_distances)
    ans = mltest.run_op(
        ml,
        ml.device,
        True,
        layer,
        points,
        queries=queries,
        radius=radius,
        points_row_splits=points_row_splits,
        queries_row_splits=queries_row_splits,
        hash_table_size_factor=hash_table_size_factor,
    )

    for i, q in enumerate(queries):
        # check neighbors
        start = ans.neighbors_row_splits[i]
        end = ans.neighbors_row_splits[i + 1]
        q_neighbors_index = ans.neighbors_index[start:end]

        gt_set = set(gt_neighbors_index[i])
        if ignore_query_point:
            gt_set.remove(i)
        assert gt_set == set(q_neighbors_index)

        # check distances
        if return_distances:
            q_neighbors_dist = ans.neighbors_distance[start:end]
            for j, dist in zip(q_neighbors_index, q_neighbors_dist):
                if metric == 'L2':
                    gt_dist = np.sum((q - points[j])**2)
                else:
                    gt_dist = np.linalg.norm(q - points[j], ord=p_norm)
                np.testing.assert_allclose(dist, gt_dist, rtol=1e-7, atol=1e-8)
