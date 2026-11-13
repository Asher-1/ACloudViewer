"""Exercise production bindings through the official CLI, not artificial registrations."""
import json
import locale
import os
from pathlib import Path
import shutil
import subprocess

import pytest


CASES = {
    "histogram_returns_independent_list": """
cloud = pycc.ccPointCloud("histogram")
index = cloud.addScalarField("values")
field = cloud.getScalarField(index)
field.resize(4, 1.0)
field.setValue(1, 2.0)
field.computeMinAndMax()
bins = field.getHistogram()
assert isinstance(bins, list), type(bins)
assert sum(bins) == 4, bins
assert all(isinstance(value, int) for value in bins)
bins.append(999)
assert sum(field.getHistogram()) == 4
""",
    "empty_histogram_returns_list": """
cloud = pycc.ccPointCloud("empty histogram")
field = cloud.getScalarField(cloud.addScalarField("values"))
assert field.getHistogram() == []
assert isinstance(field.getHistogram(), list)
""",
    "waveform_decode_returns_list": """
wave = pycc.ccWaveform(1)
wave.setDataDescription(0, 1)
descriptor = pycc.WaveformDescriptor()
descriptor.numberOfSamples = 1
descriptor.bitsPerSample = 8
descriptor.digitizerGain = 2.0
descriptor.digitizerOffset = -1.0
# One sample fits the bound uint8_t argument; avoid an out-of-bounds pointer fixture.
values = wave.decodeSamples(descriptor, 7)
assert isinstance(values, list), type(values)
assert values == [13.0], values
""",
    "empty_waveform_decode_returns_list": """
wave = pycc.ccWaveform(0)
descriptor = pycc.WaveformDescriptor()
values = wave.decodeSamples(descriptor, 0)
assert isinstance(values, list), type(values)
assert values == []
""",
    "empty_proxy_decode_returns_list": """
wave = pycc.ccWaveform(0)
descriptor = pycc.WaveformDescriptor()
proxy = pycc.ccWaveformProxy(wave, descriptor, 0)
assert not proxy.isValid()
values = proxy.decodeSamples()
assert isinstance(values, list), type(values)
assert values == []
""",
    "ascii_accepts_list_without_mutating_input": """
values = [1.25, -2.5]
assert pycc.ccWaveform.ToASCII(str(output), values, 3)
assert values == [1.25, -2.5]
assert output.read_text().splitlines() == [
    "//time(ps);intensity", "0;1.250000", "3;-2.500000"]
""",
    "ascii_accepts_empty_list": """
assert pycc.ccWaveform.ToASCII(str(output), [], 3)
assert output.read_text().splitlines() == ["//time(ps);intensity"]
""",
    "ascii_accepts_numpy_array": """
import numpy as np
values = np.array([1.25, -2.5], dtype=np.float64)
assert pycc.ccWaveform.ToASCII(str(output), values, 3)
assert values.tolist() == [1.25, -2.5]
assert output.read_text().splitlines() == [
    "//time(ps);intensity", "0;1.250000", "3;-2.500000"]
""",
    "ascii_accepts_tuple": """
assert pycc.ccWaveform.ToASCII(str(output), (1.25, -2.5), 3)
assert output.read_text().splitlines() == [
    "//time(ps);intensity", "0;1.250000", "3;-2.500000"]
""",
    "ascii_accepts_generator": """
values = (value for value in [1.25, -2.5])
assert pycc.ccWaveform.ToASCII(str(output), values, 3)
assert list(values) == []
assert output.read_text().splitlines() == [
    "//time(ps);intensity", "0;1.250000", "3;-2.500000"]
""",
    "ascii_rejects_invalid_values_before_writing": """
for values in (None, 42, "12", b"12", {}, [1, "bad"]):
    try:
        pycc.ccWaveform.ToASCII(str(output), values, 3)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid values were accepted")
    assert not output.exists()
""",
    "ascii_consumes_generator_before_element_error": """
values = (value for value in [1, "bad", 3])
try:
    pycc.ccWaveform.ToASCII(str(output), values, 3)
except TypeError:
    pass
else:
    raise AssertionError("invalid generator was accepted")
assert list(values) == []
assert not output.exists()
""",
    "qstring_preserves_embedded_nul": """
name = "\\u70b9\\u4e91\\0\\u6807\\u6ce8"
cloud = pycc.ccPointCloud(name)
assert cloud.getName() == name, repr(cloud.getName())
""",
    "qstring_rejects_wrong_type_without_poisoning_next_call": """
try:
    pycc.ccPointCloud(42)
except TypeError:
    pass
else:
    raise AssertionError("integer name was accepted")
name = "\\u6709\\u6548\\u540d\\u79f0"
assert pycc.ccPointCloud(name).getName() == name
""",
    "python_plugin_actions_cross_cpp_callback": """
class Consumer(pycc.PythonPluginInterface):
    def getActions(self):
        return [pycc.Action("\\u70b9\\u4e91\\u64cd\\u4f5c", lambda: None)]
consumer = Consumer()
actions = pycc.PythonPluginInterface.getActions(consumer)
assert isinstance(actions, list), type(actions)
assert len(actions) == 1
""",
    "indices_vector_is_mutated_by_cpp_search": """
cloud = cccorelib.PointCloud()
assert cloud.reserve(3)
for x in (0.0, 1.0, 2.0):
    cloud.addPoint(cccorelib.CCVector3(x, 0.0, 0.0))
tree = cccorelib.KDTree()
assert tree.buildFromCloud(cloud)
indices = cccorelib.KDTree.IndicesVector()
assert tree.radiusSearch([0.0, 0.0, 0.0], 1.0, 0.01, indices) == 1
assert list(indices) == [1], list(indices)
""",
    "numpy_foreign_callback_cannot_mutate_work_array": """
import numpy as np
field = cccorelib.ScalarField("numbers")
field.resize(3, 1.0)
view = field.asArray()
class Foreign:
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise AssertionError("unsupported callback received private work arrays")
assert view.__array_ufunc__(np.add, "__call__", view, Foreign(), out=(view,)) is NotImplemented
np.testing.assert_array_equal(field.toArray(), [1.0, 1.0, 1.0])
""",
    "numpy_assignment_reentry_preserves_updates": """
import numpy as np
field = cccorelib.ScalarField("numbers")
field.resize(3, 1.0)
class Value:
    target = 1
    def __float__(self):
        field.setValue(self.target, 99.0)
        return 7.0
field.asArray()[0] = Value()
np.testing.assert_array_equal(field.toArray(), [7.0, 99.0, 1.0])
field.fill(1.0)
value = Value()
value.target = 0
try:
    field.asArray()[0] = value
except ValueError:
    pass
else:
    raise AssertionError("reentrant target update was overwritten")
np.testing.assert_array_equal(field.toArray(), [99.0, 1.0, 1.0])
""",
    "numpy_ufunc_where_preserves_reentrant_updates": """
import numpy as np
field = cccorelib.ScalarField("numbers")
field.resize(3, 1.0)
view = field.asArray()
def convert(value):
    field.setValue(1, 99.0)
    return 7.0
np.frompyfunc(convert, 1, 1)(view, out=view, where=[True, False, True], casting="unsafe")
np.testing.assert_array_equal(field.toArray(), [7.0, 99.0, 7.0])
""",
    "numpy_short_access_avoids_full_field_allocation": """
import numpy as np
import tracemalloc
field = cccorelib.ScalarField("large field")
field.resize(100000, 1.0)
view = field.asArray()
tracemalloc.start()
try:
    assert len(view) == 100000
    assert view[-1] == 1.0
    assert len(view[10:13]) == 3
    assert np.asarray(view[10:13]).shape == (3,)
    view[10] = 7.0
    _, peak = tracemalloc.get_traced_memory()
finally:
    tracemalloc.stop()
assert peak < 65536, peak
assert field.getValue(10) == 7.0
""",
    "vector2_container_is_opaque_and_usable_by_geometry": """
points = cccorelib.CCVector2List()
for x, y in ((0, 0), (2, 0), (2, 2), (0, 2)):
    points.append(cccorelib.CCVector2(x, y))
assert len(points) == 4
points[0].x = 0.5
assert points[0].x == 0.5
assert cccorelib.ManualSegmentationTools.isPointInsidePoly(cccorelib.CCVector2(1, 1), points)
""",
    "vector2_legacy_list_geometry_input_is_preserved": """
points = [cccorelib.CCVector2(x, y) for x, y in ((0, 0), (2, 0), (2, 2), (0, 2))]
assert cccorelib.ManualSegmentationTools.isPointInsidePoly(cccorelib.CCVector2(1, 1), points)
assert not cccorelib.ManualSegmentationTools.isPointInsidePoly(cccorelib.CCVector2(3, 3), tuple(points))
if cccorelib.Delaunay2dMesh.Available():
    mesh = cccorelib.Delaunay2dMesh()
    assert mesh.buildMesh(points, 4, "")
    assert mesh.size() == 2
""",
    "scalar_container_has_a_production_opaque_binding": """
values = cccorelib.GenericDistribution.ScalarContainer()
for value in (1.0, 2.0, 3.0):
    values.append(value)
assert not isinstance(values, list)
values[1] = 4.0
assert list(values) == [1.0, 4.0, 3.0]
distribution = cccorelib.NormalDistribution()
assert distribution.computeRobustParameters(values, 2.0)
assert distribution.isValid()
# Support sequences explicitly at this const-input boundary, not in the opaque caster.
legacy = [1.0, 2.0, 3.0]
assert distribution.computeRobustParameters(legacy, 2.0)
assert legacy == [1.0, 2.0, 3.0]
assert distribution.getMu() == 2.0
""",
    "indices_vector_keeps_reference_semantics": """
indices = cccorelib.KDTree.IndicesVector()
indices.append(4)
indices.append(7)
indices[0] = 9
assert list(indices) == [9, 7]
assert not isinstance(indices, list)
""",
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("unicode_paths", [False, True])
def test_production_binding_contract(case, unicode_paths, acloudviewer_exe, tmp_path):
    cli = shutil.which("cli-anything-acloudviewer")
    assert cli is not None, "The official cli-anything-acloudviewer toolchain is required"
    assert Path(acloudviewer_exe).is_file(), "A built ACloudViewer binary is required"
    if unicode_paths:
        tmp_path /= "\u8def\u5f84 with spaces"
        tmp_path.mkdir()
    output = tmp_path / "waveform.txt"
    marker = tmp_path / "executed.json"
    script = tmp_path / "binding_case.py"
    previous_locale = locale.setlocale(locale.LC_CTYPE)
    try:
        if unicode_paths:
            # Reproduce a non-UTF-8 text locale without changing filesystem encoding.
            locale.setlocale(locale.LC_CTYPE, "C")
        script.write_text(
            "import cccorelib, pycc, json, locale, sys\nfrom pathlib import Path\n"
            + f"output = Path({str(output)!r})\n"
            + CASES[case]
            + f"\nPath({str(marker)!r}).write_text(json.dumps({{'case': {case!r}, 'child_locale': locale.setlocale(locale.LC_CTYPE), 'filesystem_encoding': sys.getfilesystemencoding()}}), encoding='utf-8')\n",
            encoding="utf-8",
        )
    finally:
        locale.setlocale(locale.LC_CTYPE, previous_locale)
    env = dict(os.environ, ACV_BINARY=str(Path(acloudviewer_exe).resolve()),
               PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen",
               XDG_CONFIG_HOME=str(tmp_path / "config"),
               XDG_DATA_HOME=str(tmp_path / "data"),
               XDG_DATA_DIRS=str(tmp_path / "system-data"))
    if unicode_paths:
        # Exercise a real C-locale child while retaining Unicode filesystem arguments.
        env.update(LC_ALL="C", LANG="C", PYTHONUTF8="1", PYTHONCOERCECLOCALE="0")
    completed = subprocess.run(
        [cli, "--json", "--mode", "headless", "plugin", "run", "PYTHON_SCRIPT",
         "--arg=" + str(script)],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["exit_code"] == 0, report
    assert marker.is_file(), "The script did not finish; empty stdout is not passing evidence"
    observed = json.loads(marker.read_text(encoding="utf-8"))
    assert observed["case"] == case
    if unicode_paths:
        assert observed["child_locale"] == "C", observed
        assert observed["filesystem_encoding"] == "utf-8", observed
