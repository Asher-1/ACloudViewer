# Frequently Asked Questions

## Installation and startup

<details>
<summary>Q: Startup reports "Only part of a ReadProcessMemory or WriteProcessMemory request was completed".</summary>

Try launching the application again.
</details>

<details>
<summary>Q: Startup reports `ImportError: No module named expat; use SimpleXMLTreeBuilder instead`.</summary>

Conda may update a system library such as expat beyond the version supported by a prebuilt Python module. Try installing the compatible version:

```bash
conda install expat=2.5.0 -y
```
</details>

<details>
<summary>Q: The canvas cursor moves slowly under Fedora KDE.</summary>

On Fedora KDE, under either Wayland or X11, force Qt to use XCB:

```bash
python -m anylabeling --qt-platform xcb
```

Alternatively, set the environment variable:

```bash
export QT_QPA_PLATFORM=xcb
python -m anylabeling
```

See [issue #1145](https://github.com/CVHub520/X-AnyLabeling/issues/1145).
</details>

<details>
<summary>Q: Startup reports `Failed to execute script 'app' ... 'str' object does not support item assignment`.</summary>

Delete the user configuration file and restart the application. See [issue #996](https://github.com/CVHub520/X-AnyLabeling/issues/996).
</details>

<details>
<summary>Q: Startup reports `OPENSSL_Uplink(...): not OPENSSL_Applink`.</summary>

See [issue #941](https://github.com/CVHub520/X-AnyLabeling/issues/941).
</details>

<details>
<summary>Q: The application crashes with a `Qt5Core.dll` dependency error.</summary>

See [issue #907](https://github.com/CVHub520/X-AnyLabeling/issues/907).
</details>

<details>
<summary>Q: Startup prints `Gtk-WARNING: Could not load a pixbuf from icon theme`.</summary>

See [issue #893](https://github.com/CVHub520/X-AnyLabeling/issues/893).
</details>

<details>
<summary>Q: Startup reports `AttributeError: 'NoneType' object has no attribute 'items'`.</summary>

Delete `.xanylabelingrc` from your user directory and try again. See [issue #877](https://github.com/CVHub520/X-AnyLabeling/issues/877).
</details>

<details>
<summary>Q: Startup cannot find `cublasLt64_12.dll`.</summary>

ONNX Runtime may be incompatible with the installed CUDA version; see [issue #844](https://github.com/CVHub520/X-AnyLabeling/issues/844). If CUDA and cuDNN are not installed system-wide, see [issue #1014](https://github.com/CVHub520/X-AnyLabeling/issues/1014).
</details>

<details>
<summary>Q: Qt cannot load the `xcb` platform plugin even though it was found.</summary>

See [issues #541](https://github.com/CVHub520/X-AnyLabeling/issues/541) and [#496](https://github.com/CVHub520/X-AnyLabeling/issues/496).
</details>

<details>
<summary>Q: Qt cannot find the `wayland` platform plugin.</summary>

See [issue #761](https://github.com/CVHub520/X-AnyLabeling/issues/761).
</details>

<details>
<summary>Q: The GPU build exits unexpectedly.</summary>

See [issue #500](https://github.com/CVHub520/X-AnyLabeling/issues/500).
</details>

<details>
<summary>Q: `QStandardPaths` reports incorrect permissions for `/run/user/1000`.</summary>

Set the directory permissions to `0700`, for example by adding `chmod 0700 /run/user/1000/` to `.bashrc`, then activate the shell and restart the application.
</details>

<details>
<summary>Q: Startup reports `importlib.resources` has no attribute `files`.</summary>

Upgrade Python to 3.11 or later.
</details>

## Annotation and interface

<details>
<summary>Q: How can I draw small objects more quickly?</summary>

See [issue #1000](https://github.com/CVHub520/X-AnyLabeling/issues/1000).
</details>

<details>
<summary>Q: How can I annotate a hollow object for image segmentation?</summary>

See [issue #991](https://github.com/CVHub520/X-AnyLabeling/issues/991).
</details>

<details>
<summary>Q: How can I label and group keypoints for multiple objects?</summary>

See [issue #982](https://github.com/CVHub520/X-AnyLabeling/issues/982).
</details>

<details>
<summary>Q: How do I enable continuous drawing for rectangles and other boxes?</summary>

Edit `.xanylabelingrc` in your user directory and set `auto_highlight_shape` and `auto_switch_to_edit_mode` to `False`. See [issue #887](https://github.com/CVHub520/X-AnyLabeling/issues/887).
</details>

<details>
<summary>Q: The interface looks blurry on a high-resolution display.</summary>

See [issue #811](https://github.com/CVHub520/X-AnyLabeling/issues/811).
</details>

<details>
<summary>Q: The label manager does not open to assign a label after annotation.</summary>

Clear `Auto Use Last Label`. See [issue #805](https://github.com/CVHub520/X-AnyLabeling/issues/805).
</details>

<details>
<summary>Q: Why do copied annotations overlap? (Duplicate Polygons vs. Copy & Paste Objects)</summary>

- `Ctrl+D` quickly duplicates an object in the current image. The duplicate is offset by two pixels by default.
- `Ctrl+C` and `Ctrl+V` are intended mainly for copying objects between images. They preserve the original coordinates, so pasting into the same image overlaps the source object.

The system clipboard is disabled by default. In that mode, copied objects remain in the application and can be pasted after changing images. To copy annotation JSON to the system clipboard, enable `Edit > Use System Clipboard`; each copy then replaces the clipboard contents.
</details>

<details>
<summary>Q: The application cannot open a `.jpg` or `.png` image.</summary>

See [issue #823](https://github.com/CVHub520/X-AnyLabeling/issues/823).
</details>

<details>
<summary>Q: A very large image cannot be opened, or Qt reports an allocation limit.</summary>

Qt's `QImageReader` has a default allocation limit of about 256 MB. Change it as needed in `Settings` (`Ctrl+0`) > `General` > `Qt Image Allocation Limit`, in `.xanylabelingrc` with `qt_image_allocation_limit`, or with the `--qt-image-allocation-limit` launch option.

```yaml
qt_image_allocation_limit: 1024
```

```bash
xanylabeling --qt-image-allocation-limit 1024
```

`null` keeps the Qt default, `0` disables the limit, and a positive integer sets the limit in MB. Restart the application for the change to take effect. Avoid `0` on machines with limited memory.
</details>

## Models and inference

<details>
<summary>Q: How can I use a YOLO26 model with NMS to remove overlapping boxes during auto-labeling?</summary>

YOLO26 exports end-to-end (NMS-free) by default. Export from PT with `end2end=false`. Its post-processing then matches YOLO11, so set `type: yolo11` in the custom model configuration rather than `type: yolo26`. See [issue #1280](https://github.com/CVHub520/X-AnyLabeling/issues/1280).
</details>

<details>
<summary>Q: Why does Ultralytics (YOLO) training fail in the prebuilt EXE?</summary>

Training is not supported in the prebuilt EXE: the release does not bundle the Ultralytics training environment, and running dynamic Python training scripts from a frozen executable has additional process and resource constraints. Training works in a Python source environment; use that environment when training YOLO models. See [issue #1100](https://github.com/CVHub520/X-AnyLabeling/issues/1100).
</details>

<details>
<summary>Q: A virtual environment reports `Error loading tokenizer: No such file or directory`.</summary>

This can occur when `importlib.resources.files()` returns a `Traversable` rather than a filesystem path, such as with an editable `pip install -e .` install. The resource loading code has been fixed to read resource text directly. Reinstall or update X-AnyLabeling if the error persists.
</details>

<details>
<summary>Q: `predict_shapes` reports that `AutolabelingResult.emit()` received a `list`.</summary>

Check whether the image is damaged or uses an unsupported format.
</details>

<details>
<summary>Q: The application reports `YOLOE model will be not be available`.</summary>

See [issue #997](https://github.com/CVHub520/X-AnyLabeling/issues/997).
</details>

<details>
<summary>Q: YOLOE loading on Windows reports `WinError 1314` for `mobileclip_blt.pt`.</summary>

See [issue #992](https://github.com/CVHub520/X-AnyLabeling/issues/992).
</details>

<details>
<summary>Q: Loading a custom model reports `Invalid config file format`.</summary>

See [issue #986](https://github.com/CVHub520/X-AnyLabeling/issues/986).
</details>

<details>
<summary>Q: `predict_shapes` reports that local variable `p` is unassigned.</summary>

See [issue #983](https://github.com/CVHub520/X-AnyLabeling/issues/983).
</details>

<details>
<summary>Q: The model loader cannot download or initialize encoder data.</summary>

See [issue #961](https://github.com/CVHub520/X-AnyLabeling/issues/961).
</details>

<details>
<summary>Q: Where are downloaded models stored?</summary>

See [issue #943](https://github.com/CVHub520/X-AnyLabeling/issues/943).
</details>

<details>
<summary>Q: Predicted classes do not match the training classes.</summary>

Check the custom model YAML: use two-space indentation and do not include spaces in keys. See [issue #923](https://github.com/CVHub520/X-AnyLabeling/issues/923).
</details>

<details>
<summary>Q: How can Chatbot access online models such as Google Gemini?</summary>

Configure a proxy in the current shell or system settings, replacing `ip` and `port` with your proxy address and port:

```bash
export http_proxy=http://ip:port
export https_proxy=http://ip:port
```
</details>

<details>
<summary>Q: Chatbot reports that the `socksio` package is missing for a SOCKS proxy.</summary>

When a SOCKS proxy is set, install the SOCKS extra required by `httpx`:

```bash
pip install 'httpx[socks]'
```
</details>

<details>
<summary>Q: `predict_shapes` reports `'<=' not supported between instances of 'int' and 'str'`.</summary>

Check that the model YAML is valid. See [issue #902](https://github.com/CVHub520/X-AnyLabeling/issues/902).
</details>

<details>
<summary>Q: OpenCV `resize` reports `(-215:Assertion failed) inv_scale_x > 0`.</summary>

Check that the image width and height are specified in the model configuration, and check whether the model was exported with a dynamic batch. See [issues #885](https://github.com/CVHub520/X-AnyLabeling/issues/885) and [#784](https://github.com/CVHub520/X-AnyLabeling/issues/784).
</details>

<details>
<summary>Q: A YOLO-Pose model reports `'list' object has no attribute 'items'`.</summary>

Use the official configuration template. See [issue #880](https://github.com/CVHub520/X-AnyLabeling/issues/880).
</details>

<details>
<summary>Q: Prediction reports `list index out of range`.</summary>

Check that label names are not bare numbers; quote numeric labels in YAML. Also check that the `type` field is correct. See [issues #837](https://github.com/CVHub520/X-AnyLabeling/issues/837) and [#878](https://github.com/CVHub520/X-AnyLabeling/issues/878).
</details>

<details>
<summary>Q: Model loading reports `exceptions must derive from BaseException`.</summary>

Check that the configured model path exists and is valid. Also check that the ONNX `ir_version` is supported by the installed ONNX package. See [issues #868](https://github.com/CVHub520/X-AnyLabeling/issues/868), [#441](https://github.com/CVHub520/X-AnyLabeling/issues/441), and [#1120](https://github.com/CVHub520/X-AnyLabeling/issues/1120#issuecomment-3303864917).
</details>

<details>
<summary>Q: Prediction reports `'int' object is not subscriptable`.</summary>

For a custom model, check its preprocessing, inference, and post-processing logic. See [issue #828](https://github.com/CVHub520/X-AnyLabeling/issues/828).
</details>

<details>
<summary>Q: Installing SAM2 fails at `from sam2 import _C`.</summary>

See [issues #719](https://github.com/CVHub520/X-AnyLabeling/issues/719), [#842](https://github.com/CVHub520/X-AnyLabeling/issues/842), [#843](https://github.com/CVHub520/X-AnyLabeling/issues/843), [#864](https://github.com/CVHub520/X-AnyLabeling/issues/864), and [#865](https://github.com/CVHub520/X-AnyLabeling/issues/865).
</details>

<details>
<summary>Q: A downloaded model is deleted and downloaded again on every restart.</summary>

The application verifies an existing cached model before using it. An empty file or failed integrity check causes the cache entry to be removed and downloaded again. A download stops with an error after the retry limit; it does not loop indefinitely in one load attempt. Check your network, configure a proxy or change the model source (GitHub or ModelScope) in the [user guide](./user_guide.md#77-model-download-source). For a model load failure, use the [quick start guide](./get_started.md) to check CUDA, cuDNN, ONNX, and ONNX Runtime compatibility. You can also use [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server) for remote PyTorch inference. A non-ASCII character in the model path can also cause problems; see [issue #600](https://github.com/CVHub520/X-AnyLabeling/issues/600).
</details>

<details>
<summary>Q: How can inference be restricted to specific classes?</summary>

Only some models support class filtering. For YOLO models, configure `filter_classes`; see the [custom model guide](https://xanylabeling.com/en/docs/x-anylabeling/custom_model#load-supported-user-custom-models).
</details>

<details>
<summary>Q: In-painting reports `assert len(self.points) in [1, 2, 4]`.</summary>

See [issue #491](https://github.com/CVHub520/X-AnyLabeling/issues/491).
</details>

<details>
<summary>Q: A custom model fails to load.</summary>

Check that the configured classes match the model's supported classes, compare input and output dimensions with the corresponding built-in model using [Netron](https://netron.app/), verify every configuration field, and make sure built-in field names were not changed. See [issues #888](https://github.com/CVHub520/X-AnyLabeling/issues/888) and [#983](https://github.com/CVHub520/X-AnyLabeling/issues/983).
</details>

<details>
<summary>Q: The model finishes but produces no output.</summary>

See [issue #536](https://github.com/CVHub520/X-AnyLabeling/issues/536).
</details>

<details>
<summary>Q: Grounding DINO GPU inference reports an ONNX Runtime `Expand` broadcast error.</summary>

See [issue #389](https://github.com/CVHub520/X-AnyLabeling/issues/389).
</details>

<details>
<summary>Q: Model loading reports `Missing config: num_masks`.</summary>

See [issue #515](https://github.com/CVHub520/X-AnyLabeling/issues/515).
</details>

<details>
<summary>Q: The application reports `AttributeError: 'int' object has no attribute 'replace'`.</summary>

Check whether a label is defined as a bare number. Put numeric label names in single quotes (`'1'`).
</details>

<details>
<summary>Q: ONNX reports `Unsupported model IR version: 11, max supported IR version: 10`.</summary>

The model's IR version is newer than the installed ONNX tools support. Use a model exported with a compatible version. The source FAQ suggests changing the version explicitly; verify model compatibility before saving a modified model:

```python
import onnx

model = onnx.load("/path/to/your/onnx_model")
model.ir_version = 10
onnx.save(model, "/path/to/compatible_model.onnx")
```
</details>

<details>
<summary>Q: The model's `ir_version` is newer than the ONNX checker.</summary>

Upgrade ONNX:

```bash
pip install --upgrade onnx
```
</details>

<details>
<summary>Q: How can I use Segment Anything Video on macOS?</summary>

See [issue #865](https://github.com/CVHub520/X-AnyLabeling/issues/865).
</details>

<details>
<summary>Q: GPU loading fails because `libcudnn.so.9` cannot be found.</summary>

Install an ONNX Runtime version compatible with your CUDA and cuDNN setup, following the [quick start guide](./get_started.md). See [issue #834](https://github.com/CVHub520/X-AnyLabeling/issues/834).
</details>

<details>
<summary>Q: Windows reports `DLL load failed while importing onnx_cpp2py_export`.</summary>

The installed ONNX and ONNX Runtime versions may be incompatible. See [issue #886](https://github.com/CVHub520/X-AnyLabeling/issues/886).
</details>

<details>
<summary>Q: A model stays in the loading state in a compiled GPU build.</summary>

This is commonly caused by incompatible ONNX and `onnxruntime-gpu` versions. Check the [ONNX Runtime compatibility documentation](https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support) and install a compatible ONNX version.
</details>

## Files and exports

<details>
<summary>Q: YOLO-Pose export shows every keypoint ID as zero.</summary>

See [issue #1270](https://github.com/CVHub520/X-AnyLabeling/issues/1270).
</details>

<details>
<summary>Q: The exported mask image is empty.</summary>

Press `Ctrl+G` to check the current task's labels and annotation types; all annotations should be `polygon` objects. Then check that the export configuration is valid and its class names match the labels. Upgrade to the latest version or run from source for more detailed logs. See [issue #1153](https://github.com/CVHub520/X-AnyLabeling/issues/1153).
</details>

<details>
<summary>Q: A prebuilt version does not extract frames when opening a video.</summary>

See [issue #1136](https://github.com/CVHub520/X-AnyLabeling/issues/1136).
</details>

<details>
<summary>Q: Labelme reports `AssertionError` when importing X-AnyLabeling annotations.</summary>

See [issue #1007](https://github.com/CVHub520/X-AnyLabeling/issues/1007).
</details>

<details>
<summary>Q: Export reports `'<label>' is not in list`.</summary>

Press `Ctrl+G` to check the annotation classes. Make sure `classes.txt` contains the same names and that the selected export format matches the annotation type (for example, rectangles versus polygons or rotated boxes).
</details>

<details>
<summary>Q: Loading an image directory causes a segmentation fault.</summary>

See [issue #906](https://github.com/CVHub520/X-AnyLabeling/issues/906).
</details>

<details>
<summary>Q: Uploading a label file reports `cannot identify image file`.</summary>

Check whether image files and label files are stored in separate directories. See [issue #911](https://github.com/CVHub520/X-AnyLabeling/issues/911).
</details>

<details>
<summary>Q: Loading a label reports `a bytes-like object is required, not 'NoneType'`.</summary>

Check that the JSON `imagePath` value matches the image filename. See [issue #869](https://github.com/CVHub520/X-AnyLabeling/issues/869).
</details>

<details>
<summary>Q: An imported label file is empty.</summary>

Check that the annotation type matches the selected import type (for example, `rectangle` versus `Polygon`). Also check for nested subdirectories in the image folder; see [issue #839](https://github.com/CVHub520/X-AnyLabeling/issues/839).
</details>

<details>
<summary>Q: Importing or exporting YOLO keypoint labels causes a crash.</summary>

Check that the model YAML is valid. See [issue #898](https://github.com/CVHub520/X-AnyLabeling/issues/898).
</details>

<details>
<summary>Q: Importing labels reports `invalid literal for int() with base 10`.</summary>

See [issue #782](https://github.com/CVHub520/X-AnyLabeling/issues/782).
</details>

<details>
<summary>Q: Mask export reports `imageWidth`.</summary>

See [issue #477](https://github.com/CVHub520/X-AnyLabeling/issues/477).
</details>

<details>
<summary>Q: An operation reports `operands could not be broadcast together with shapes (0,) (1,2)`.</summary>

See [issue #492](https://github.com/CVHub520/X-AnyLabeling/issues/492).
</details>

<details>
<summary>Q: Keypoint export reports that `group_id` is missing or `None`.</summary>

Ensure every rectangle and keypoint has a group ID.
</details>

<details>
<summary>Q: An error reports `'NoneType' object has no attribute 'shape'`.</summary>

Check whether the file path contains non-ASCII characters.
</details>
