import cccorelib
import numpy as np
import pytest


def test_get_set_name():
    sf = cccorelib.ScalarField("numbers")
    assert sf.getName() == "numbers"

    sf.setName("codes")
    assert sf.getName() == "codes"


def test_indexing():
    scalar_field = cccorelib.ScalarField("Codification")
    scalar_field.resize(3)

    scalar_field.setValue(1, 1.0)
    scalar_field[2] = 2.0

    assert scalar_field.getValue(0) == scalar_field[0]
    assert scalar_field.getValue(1) == scalar_field[1]
    assert scalar_field.getValue(2) == scalar_field[2]

    scalar_field.computeMinAndMax()
    assert scalar_field.getMin() == 0.0
    assert scalar_field.getMax() == 2.0

    # Try out of range indices
    with pytest.raises(IndexError):
        _ = scalar_field.getValue(4)

    with pytest.raises(IndexError):
        _ = scalar_field[4]

    with pytest.raises(IndexError):
        scalar_field.setValue(4, 0.0)

    with pytest.raises(IndexError):
        scalar_field[4] = 0.0


def test_as_array():
    scalar_field = cccorelib.ScalarField("Codification")
    scalar_field.resize(10)

    array = scalar_field.asArray()

    # Changes made to the scalar_field reflects on the array
    scalar_field.fill(1.0)
    assert len(array) == 10
    assert np.all(array == 1.0)

    # and vice-versa
    array[:] = 2.0
    assert scalar_field[0] == 2.0
    assert np.all(array == 2.0)


def test_numpy_ufunc_writes_back_to_scalar_field():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(4, 1.0)
    view = scalar_field.asArray()
    np.add(view, [1.0, 2.0, 3.0, 4.0], out=view)
    np.testing.assert_array_equal(scalar_field.toArray(), [2.0, 3.0, 4.0, 5.0])
    view *= 2.0
    np.testing.assert_array_equal(scalar_field.toArray(), [4.0, 6.0, 8.0, 10.0])


def test_numpy_slice_assignment_changes_only_selected_values():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(6)
    view = scalar_field.asArray()
    view[1::2] = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(scalar_field.toArray(), [0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
    view[::-1][::2] = 7.0
    np.testing.assert_array_equal(scalar_field.toArray(), [0.0, 7.0, 0.0, 7.0, 0.0, 7.0])


def test_numpy_fancy_indexing_and_broadcast_assignment():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(4)
    view = scalar_field.asArray()
    view[np.array([True, False, True, False])] = 3.0
    view[[1, 3]] = [4.0, 5.0]
    assert view[-1] == 5.0
    np.testing.assert_array_equal(view[[3, 1]], [5.0, 4.0])
    np.testing.assert_array_equal(scalar_field.toArray(), [3.0, 4.0, 3.0, 5.0])


def test_numpy_conversion_preserves_dtype_and_exports_independent_snapshot():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)
    view = scalar_field.asArray()
    snapshot = np.asarray(view)
    assert snapshot.dtype == np.dtype(view.dtype)
    assert snapshot.dtype.itemsize in (4, 8)
    assert snapshot.flags.owndata
    scalar_field.fill(2.0)
    np.testing.assert_array_equal(snapshot, [1.0, 1.0, 1.0])
    snapshot[:] = 3.0
    assert np.all(view == 2.0)
    np.testing.assert_array_equal(np.asarray(view, dtype=np.float32), [2.0, 2.0, 2.0])


def test_numpy_copy_false_rejects_unsafe_shared_storage():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3)
    view = scalar_field.asArray()
    with pytest.raises(ValueError):
        view.__array__(copy=False)
    with pytest.raises(ValueError):
        scalar_field.__array__(copy=False)
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        with pytest.raises(ValueError):
            np.asarray(view, copy=False)


def test_numpy_view_rejects_changed_field_size():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(4, 1.0)
    view = scalar_field.asArray()
    scalar_field.resize(2)
    with pytest.raises(ValueError):
        np.asarray(view)
    with pytest.raises(ValueError):
        view[:] = 3.0
    np.testing.assert_array_equal(scalar_field.toArray(), [1.0, 1.0])


def test_numpy_view_retains_owner_after_python_reference_release():
    import gc
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(4, 2.0)
    view = scalar_field.asArray()[::-1]
    assert view.dtype.itemsize in (4, 8)
    del scalar_field
    gc.collect()
    view += 1.0
    np.testing.assert_array_equal(view, [3.0, 3.0, 3.0, 3.0])


def test_numpy_invalid_assignment_does_not_partially_write():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(4, 1.0)
    view = scalar_field.asArray()
    with pytest.raises((ValueError, TypeError)):
        view[:] = [2.0, "invalid", 3.0, 4.0]
    np.testing.assert_array_equal(scalar_field.toArray(), [1.0, 1.0, 1.0, 1.0])


def test_numpy_ufunc_multiple_outputs_preserve_aliases():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(4, 1.25)
    view = scalar_field.asArray()
    fractions, integers = view[:2], view[2:]
    result = np.modf(view[:2], out=(fractions, integers))
    assert result[0] is fractions
    assert result[1] is integers
    np.testing.assert_array_equal(scalar_field.toArray(), [0.25, 0.25, 1.0, 1.0])


def test_numpy_ufunc_at_accumulates_repeated_indices():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3)
    view = scalar_field.asArray()
    np.add.at(view, [0, 0, 2], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(scalar_field.toArray(), [3.0, 0.0, 4.0])


def test_numpy_snapshot_survives_field_reallocation():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)
    view = scalar_field.asArray()
    snapshot = np.asarray(view)
    scalar_field.reserve(10000)
    view[:] = 2.0
    np.testing.assert_array_equal(snapshot, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(scalar_field.toArray(), [2.0, 2.0, 2.0])


def test_numpy_truth_value_matches_array_semantics():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(2, 1.0)
    with pytest.raises(ValueError):
        bool(scalar_field.asArray())
    scalar_field.resize(1, 0.0)
    scalar_field.fill(0.0)
    assert not bool(scalar_field.asArray())
    scalar_field.fill(1.0)
    assert bool(scalar_field.asArray())


@pytest.mark.parametrize("mutation", ["resize", "dtype", "strides"])
def test_numpy_foreign_dispatch_never_receives_work_storage(mutation):
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)
    view = scalar_field.asArray()

    class Foreign:
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            target = kwargs["out"][0]
            if mutation == "resize":
                target.resize(1, refcheck=False)
            elif mutation == "dtype":
                target.dtype = np.uint8
            else:
                target.strides = (0,)
            # Stop before the old unchecked C++ commit: the RED must be safe.
            raise RuntimeError("An unsupported callback received private work storage")

    result = view.__array_ufunc__(np.add, "__call__", view, Foreign(), out=(view,))
    assert result is NotImplemented
    np.testing.assert_array_equal(scalar_field.toArray(), [1.0, 1.0, 1.0])


def test_numpy_reentrant_assignment_preserves_non_target_update():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3)
    scalar_field.asArray()[:] = [1.0, 2.0, 3.0]

    class Value:
        def __float__(self):
            scalar_field.setValue(1, 99.0)
            return 7.0

    scalar_field.asArray()[0] = Value()
    np.testing.assert_array_equal(scalar_field.toArray(), [7.0, 99.0, 3.0])


def test_numpy_reentrant_assignment_rejects_target_conflict():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)

    class Value:
        def __float__(self):
            scalar_field.setValue(0, 99.0)
            return 7.0

    with pytest.raises(ValueError):
        scalar_field.asArray()[0] = Value()
    np.testing.assert_array_equal(scalar_field.toArray(), [99.0, 1.0, 1.0])


def test_numpy_reentrant_slice_preserves_non_target_update():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)

    class Value:
        def __float__(self):
            scalar_field.setValue(1, 99.0)
            return 7.0

    scalar_field.asArray()[::2] = Value()
    np.testing.assert_array_equal(scalar_field.toArray(), [7.0, 99.0, 7.0])


def test_numpy_reentrant_slice_rejects_target_conflict():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)

    class Value:
        def __float__(self):
            scalar_field.setValue(2, 99.0)
            return 7.0

    with pytest.raises(ValueError):
        scalar_field.asArray()[::2] = Value()
    np.testing.assert_array_equal(scalar_field.toArray(), [1.0, 1.0, 99.0])


def test_numpy_ufunc_where_preserves_non_target_reentrant_update():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)
    view = scalar_field.asArray()

    def convert(value):
        scalar_field.setValue(1, 99.0)
        return 7.0

    ufunc = np.frompyfunc(convert, 1, 1)
    ufunc(view, out=view, where=[True, False, True], casting="unsafe")
    np.testing.assert_array_equal(scalar_field.toArray(), [7.0, 99.0, 7.0])


def test_numpy_ufunc_checks_all_outputs_before_writing_any_field():
    first = cccorelib.ScalarField("first")
    second = cccorelib.ScalarField("second")
    first.resize(3, 1.0)
    second.resize(3, 2.0)

    def convert(value):
        second.setValue(0, 99.0)
        return (7.0, 8.0)

    ufunc = np.frompyfunc(convert, 1, 2)
    with pytest.raises(ValueError):
        ufunc(first.asArray(), out=(first.asArray(), second.asArray()), casting="unsafe")
    np.testing.assert_array_equal(first.toArray(), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(second.toArray(), [99.0, 2.0, 2.0])


def test_numpy_protocol_rejects_non_ufunc_without_exposing_work_storage():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)
    view = scalar_field.asArray()

    class InvalidUfunc:
        def __call__(self, *inputs, **kwargs):
            kwargs["out"][0].resize(1, refcheck=False)
            raise RuntimeError("A non-ufunc received private work storage")

    result = view.__array_ufunc__(InvalidUfunc(), "__call__", view, out=(view,))
    assert result is NotImplemented
    np.testing.assert_array_equal(scalar_field.toArray(), [1.0, 1.0, 1.0])


@pytest.mark.parametrize("operation", ["length", "scalar", "slice", "snapshot", "assignment"])
def test_numpy_small_access_does_not_allocate_a_full_field_copy(operation):
    import tracemalloc
    scalar_field = cccorelib.ScalarField("large field")
    scalar_field.resize(100000)
    scalar_field.setValue(123, 4.0)
    view = scalar_field.asArray()
    tracemalloc.start()
    try:
        if operation == "length":
            assert len(view) == 100000
        elif operation == "scalar":
            assert view[123] == 4.0
        elif operation == "slice":
            assert len(view[100:103]) == 3
        elif operation == "snapshot":
            np.testing.assert_array_equal(np.asarray(view[123:126]), [4.0, 0.0, 0.0])
        else:
            view[123] = 7.0
            assert scalar_field.getValue(123) == 7.0
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # A float32 field copy alone needs 400000 bytes; scalar operations need no such buffer.
    assert peak < 65536, peak


@pytest.mark.parametrize("index", [slice(None, None, -1), slice(7, 1, -2), slice(2, 2), slice(None, None, 2)])
def test_numpy_basic_slice_composition_matches_values(index):
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(8)
    scalar_field.asArray()[:] = np.arange(8)
    view = scalar_field.asArray()[index][::-1]
    expected = np.arange(8)[index][::-1]
    np.testing.assert_array_equal(view, expected)
    assert len(view) == len(expected)


@pytest.mark.parametrize("index", [1 << 100, -(1 << 100)])
def test_numpy_large_integer_index_raises_index_error(index):
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)
    with pytest.raises(IndexError):
        _ = scalar_field.asArray()[index]
    with pytest.raises(IndexError):
        scalar_field.asArray()[index] = 7.0


def test_numpy_index_callback_is_resolved_before_reading_native_storage():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3, 1.0)
    view = scalar_field.asArray()

    class Index:
        def __index__(self):
            scalar_field.reserve(10000)
            scalar_field.setValue(0, 9.0)
            return 0

    assert view[Index()] == 9.0


def test_numpy_tuple_and_newaxis_indexing_keeps_writable_basic_views():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(4)
    view = scalar_field.asArray()
    view[None][0, 1::2] = [4.0, 5.0]
    np.testing.assert_array_equal(view, [0.0, 4.0, 0.0, 5.0])
    view[..., ::-1][1] = 7.0
    np.testing.assert_array_equal(view, [0.0, 4.0, 7.0, 5.0])


def test_numpy_slice_conflict_comparison_handles_nan_and_signed_zero():
    scalar_field = cccorelib.ScalarField("numbers")
    scalar_field.resize(3)
    view = scalar_field.asArray()
    view[:] = [np.nan, -0.0, 2.0]
    view[::2] = [np.nan, 3.0]
    values = np.asarray(view)
    assert np.isnan(values[0])
    assert np.signbit(values[1])
    assert values[2] == 3.0


def test_numpy_reduction_and_empty_field():
    scalar_field = cccorelib.ScalarField("numbers")
    empty = scalar_field.asArray()
    assert np.asarray(empty).shape == (0,)
    scalar_field.resize(4, 2.0)
    assert np.add.reduce(scalar_field.asArray()) == 8.0
    assert np.sum(scalar_field.asArray()) == 8.0


@pytest.mark.parametrize("container", ["array", "list"])
def test_numpy_where_mutation_rejects_the_entire_transaction(container):
    field = cccorelib.ScalarField("control mutation")
    field.resize(3, 1.0)
    where = [True, False, False]
    if container == "array":
        where = np.asarray(where)

    def convert(value):
        where[:] = [True, True, True]
        field.setValue(2, 99.0)
        return 7.0

    with pytest.raises(ValueError, match="control input changed"):
        np.frompyfunc(convert, 1, 1)(field.asArray(), out=field.asArray(),
                                   where=where, casting="unsafe")
    np.testing.assert_array_equal(field.toArray(), [1.0, 1.0, 99.0])


@pytest.mark.parametrize("container", ["array", "list"])
def test_numpy_at_index_mutation_rejects_the_entire_transaction(container):
    field = cccorelib.ScalarField("control mutation")
    field.resize(3, 1.0)
    indices = [0, 1]
    if container == "array":
        indices = np.asarray(indices)

    def convert(value):
        indices[1] = 2
        field.setValue(2, 99.0)
        return 7.0

    with pytest.raises(ValueError, match="control input changed"):
        np.frompyfunc(convert, 1, 1).at(field.asArray(), indices)
    np.testing.assert_array_equal(field.toArray(), [1.0, 1.0, 99.0])


@pytest.mark.parametrize("where, expected", [(False, 0.0), ([True, False, True], 2.0)])
@pytest.mark.parametrize("scalar_output", [False, True])
def test_numpy_reduce_where_filters_inputs_not_output_writes(where, expected, scalar_output):
    field = cccorelib.ScalarField("reduction")
    field.resize(3, 1.0)
    view = field.asArray()
    output = view[0, ...] if scalar_output else view[:1]
    result = np.add.reduce(view, out=output, keepdims=not scalar_output, where=where)
    assert result is output
    np.testing.assert_array_equal(field.toArray(), [expected, 1.0, 1.0])


@pytest.mark.parametrize("index, expected", [(slice(None), [3.0, 3.0, 3.0]),
                                           (slice(None, None, -2), [3.0, 1.0, 3.0])])
def test_numpy_at_accepts_basic_slice_indices(index, expected):
    field = cccorelib.ScalarField("at slices")
    field.resize(3, 1.0)
    np.add.at(field.asArray(), index, 2.0)
    np.testing.assert_array_equal(field.toArray(), expected)


@pytest.mark.parametrize("operation", ["read", "length", "write", "ufunc"])
def test_numpy_tuple_basic_view_resolves_index_callbacks_only_once(operation):
    field = cccorelib.ScalarField("stable view")
    field.resize(4)
    field.asArray()[:] = [0.0, 1.0, 2.0, 3.0]

    class Index:
        calls = 0

        def __index__(self):
            value = self.calls
            self.calls += 1
            return value

    index = Index()
    child = field.asArray()[(slice(index, None),)]
    assert index.calls == 1
    if operation == "read":
        np.testing.assert_array_equal(np.asarray(child), [0.0, 1.0, 2.0, 3.0])
    elif operation == "length":
        assert len(child) == 4
    elif operation == "write":
        child[:] = 7.0
        np.testing.assert_array_equal(field.toArray(), [7.0, 7.0, 7.0, 7.0])
    else:
        np.add(child, 1.0, out=child)
        np.testing.assert_array_equal(field.toArray(), [1.0, 2.0, 3.0, 4.0])
    assert index.calls == 1


@pytest.mark.parametrize("index", [0, -1, np.int64(1), np.array(1)])
def test_scalar_field_integer_value_boundary_preserves_python_float(index):
    import json
    field = cccorelib.ScalarField("value boundary")
    field.resize(3, 1.25)
    assert type(field[index]) is float
    assert json.dumps(field[index]) == "1.25"
    assert isinstance(field.asArray()[index], np.floating)


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("consumer", ["field", "view"])
def test_numpy_zero_dimensional_integer_access_avoids_full_storage(operation, consumer):
    import tracemalloc
    field = cccorelib.ScalarField("scalar index")
    field.resize(100000, 1.25)
    target = field if consumer == "field" else field.asArray()
    index = np.array(123)
    tracemalloc.start()
    try:
        if operation == "read":
            assert target[index] == 1.25
        else:
            target[index] = 7.0
            assert field.getValue(123) == 7.0
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 65536, peak


@pytest.mark.parametrize("operation", ["read", "write", "overlap", "reverse", "strided", "multi_output"])
def test_numpy_short_ufunc_uses_the_owner_window_not_full_storage(operation):
    import tracemalloc
    field = cccorelib.ScalarField("short ufunc")
    field.resize(100000, 1.25)
    view = field.asArray()
    first, second = view[10:13], view[13:16]
    tracemalloc.start()
    try:
        if operation == "read":
            result = np.add(first, 1.0)
            np.testing.assert_array_equal(result, [2.25, 2.25, 2.25])
        elif operation == "write":
            np.add(first, 1.0, out=first)
            assert field.getValue(10) == 2.25
        elif operation == "overlap":
            np.add(view[10:13], 1.0, out=view[11:14])
            assert field.getValue(11) == 2.25
            assert field.getValue(10) == 1.25
        elif operation == "reverse":
            target = view[12:9:-1]
            np.add(target, 1.0, out=target)
            assert field.getValue(10) == 2.25
        elif operation == "strided":
            target = view[10:16:2]
            np.add(target, 1.0, out=target)
            assert field.getValue(10) == 2.25
            assert field.getValue(11) == 1.25
        else:
            np.modf(first, out=(first, second))
            assert field.getValue(10) == 0.25
            assert field.getValue(13) == 1.0
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 65536, peak


@pytest.mark.parametrize("where", [[1, 0, 1], [1.0, 0.0, 1.0], (1, 0, 1), [[1, 0, 1]]])
def test_numpy_where_sequence_preserves_numpy_boolean_conversion(where):
    field = cccorelib.ScalarField("where sequence")
    field.resize(3, 1.0)
    view = field.asArray()
    expected = np.ones(3, dtype=view.dtype)
    if isinstance(where[0], list):
        view = view[None]
        expected = expected[None]
    np.add(expected, 2.0, out=expected, where=where)
    np.add(view, 2.0, out=view, where=where)
    np.testing.assert_array_equal(np.asarray(view), expected)


def test_numpy_where_integer_array_keeps_numpy_safe_cast_rejection():
    field = cccorelib.ScalarField("where array")
    field.resize(3, 1.0)
    view = field.asArray()
    where = np.array([1, 0, 1])
    with pytest.raises(TypeError):
        np.add(np.ones(3), 2.0, out=np.ones(3), where=where)
    with pytest.raises(TypeError):
        np.add(view, 2.0, out=view, where=where)
    np.testing.assert_array_equal(field.toArray(), [1.0, 1.0, 1.0])


@pytest.mark.parametrize("kind", ["slice", "tuple", "newaxis"])
@pytest.mark.parametrize("operation", ["read", "write", "at"])
def test_numpy_basic_slice_callback_order_matches_numpy(kind, operation):
    calls = []

    class Bound:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def __index__(self):
            calls.append(self.name)
            return self.value

    index = slice(Bound("start", 1), Bound("stop", 4), Bound("step", 1))
    if kind == "tuple":
        index = (index,)
    elif kind == "newaxis":
        index = (slice(None), index)
    field = cccorelib.ScalarField("slice callback order")
    field.resize(5, 0.0)
    view = field.asArray()
    view[:] = np.arange(5)
    expected = np.arange(5, dtype=view.dtype)
    if kind == "newaxis":
        view, expected = view[None], expected[None]
    if operation == "read":
        reference = expected[index]
    elif operation == "write":
        expected[index] = 7.0
    else:
        np.add.at(expected, index, 7.0)
    reference_calls = calls.copy()
    calls.clear()
    if operation == "read":
        np.testing.assert_array_equal(np.asarray(view[index]), reference)
    elif operation == "write":
        view[index] = 7.0
        np.testing.assert_array_equal(np.asarray(view), expected)
    else:
        np.add.at(view, index, 7.0)
        np.testing.assert_array_equal(np.asarray(view), expected)
    assert calls == reference_calls


@pytest.mark.parametrize("kind", ["slice", "tuple", "newaxis"])
def test_numpy_zero_slice_step_rejects_before_other_bound_callbacks(kind):
    calls = []

    class Bound:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def __index__(self):
            calls.append(self.name)
            return self.value

    index = slice(Bound("start", 1), Bound("stop", 4), Bound("step", 0))
    field = cccorelib.ScalarField("zero step order")
    field.resize(5, 1.0)
    view, expected = field.asArray(), np.ones(5)
    if kind == "tuple":
        index = (index,)
    elif kind == "newaxis":
        index = (slice(None), index)
        view, expected = view[None], expected[None]
    with pytest.raises(ValueError):
        _ = expected[index]
    reference_calls = calls.copy()
    calls.clear()
    with pytest.raises(ValueError):
        _ = view[index]
    assert calls == reference_calls == ["step"]


@pytest.mark.parametrize("order", ["=", "<", ">"])
def test_numpy_assignment_preserves_explicit_input_byteorder_conversion(order):
    field = cccorelib.ScalarField("input byteorder")
    field.resize(3, 0.0)
    view = field.asArray()
    values = np.array([1.25, -2.5, 3.75], dtype=view.dtype.newbyteorder(order))
    reference = np.zeros(3, dtype=view.dtype)
    reference[:] = values
    view[:] = values
    np.testing.assert_array_equal(np.asarray(view), reference)


@pytest.mark.parametrize("index", [slice(10, 13), (slice(10, 13),)])
def test_numpy_builtin_slice_reuses_immutable_index_without_normalization_storage(index):
    import tracemalloc
    field = cccorelib.ScalarField("immutable slice")
    field.resize(20, 1.0)
    view = field.asArray()
    tracemalloc.start()
    try:
        selection = view[index]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(selection) == 3
    assert peak <= 128, peak


@pytest.mark.parametrize("operation", ["slice", "snapshot", "assignment", "ufunc"])
def test_numpy_single_slice_tuple_avoids_full_owner_storage(operation):
    import tracemalloc
    field = cccorelib.ScalarField("tuple window")
    field.resize(100000, 1.25)
    view = field.asArray()
    index = (slice(12, 9, -1),)
    tracemalloc.start()
    try:
        if operation == "slice":
            assert len(view[index]) == 3
        elif operation == "snapshot":
            np.testing.assert_array_equal(np.asarray(view[index]), [1.25, 1.25, 1.25])
        elif operation == "assignment":
            view[index] = 7.0
            assert field.getValue(10) == 7.0
            assert field.getValue(9) == 1.25
        else:
            target = view[index]
            np.add(target, 1.0, out=target)
            assert field.getValue(10) == 2.25
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 65536, peak


def test_numpy_nested_slice_tuple_retains_numpy_index_rejection():
    field = cccorelib.ScalarField("nested tuple")
    field.resize(3, 1.0)
    index = ((slice(None),),)
    with pytest.raises(IndexError):
        _ = np.ones(3)[index]
    with pytest.raises(IndexError):
        _ = field.asArray()[index]


def test_numpy_dense_output_does_not_allocate_an_all_true_write_mask():
    import tracemalloc
    field = cccorelib.ScalarField("dense output")
    field.resize(200000, 1.25)
    view = field.asArray()
    expected = np.full(200000, 2.25, dtype=view.dtype)
    tracemalloc.start()
    try:
        np.add(view, 1.0, out=view)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(np.asarray(view), expected)
    # Two ScalarType buffers suffice when every work-window element is a write target.
    assert peak < 2 * 200000 * view.dtype.itemsize + 65536, peak


def test_numpy_readonly_reduction_does_not_allocate_commit_buffers():
    import tracemalloc
    field = cccorelib.ScalarField("readonly reduction")
    field.resize(100000, 1.0)
    view = field.asArray()
    tracemalloc.start()
    try:
        assert np.add.reduce(view) == 100000.0
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100000 * view.dtype.itemsize * 1.4 + 65536, peak
