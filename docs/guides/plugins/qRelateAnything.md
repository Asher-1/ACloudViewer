# RelateAnything scene relations

Enable `AICore_ENABLED` and `PLUGIN_STANDARD_QRELATEANYTHING`, then open the
RelateAnything plugin. It detects objects with the existing YOLO inference task
and displays subject–predicate–object relations with scores and an annotated
preview. Choose among 24 relation models (12 ViT + 12 ConvNeXt) in F32/F16/Q8_0
precision.

**Try sample data** downloads and verifies the shared RelateAnything archive. One
click fills both the six-image list and the video input. Only `horse.jpg` is
checked initially; check more images or use **Check all** to process the whole
list. Click **Run** to download any missing models and begin inference. Use the
video tab for video files or cameras. Results can be
filtered, exported as JSON/CSV/text, or added as an annotated image to the DB.

**Try sample data** fills the images and the recommended video in one click;
the video tab has no separate sample button. Under **Options**, **Object
detector** lists YOLOE, YOLO-World (497 RA-4M categories), YOLO segmentation,
RF-DETR and YOLO detection models; the default is `yoloe-26s-seg-pf-f16`.
**Operating point** offers Official video reel (default) and Official image demo
presets, which set detection confidence, relation/spatial thresholds, maximum
objects, and the predicate vocabulary (65 or 35 curated phrases). **Relation
score threshold** and **Spatial relation threshold** hide relations the
calibrated model is less sure about; lower them to see more. At most 8 relations
are shown, 3 of them spatial (adjustable in the default preset).

Optional custom predicates are separated with `|`. The matching text student
and official tokenizer are downloaded through the same shared model cache.
The decomposition option exposes upstream spatial and semantic relations.
**Model arithmetic** defaults to **Automatic (recommended)**, matching upstream
precision selection. Choose **Precise** for strict F32 mode or **Optimized**
to force the optimized path.

For existing annotations, choose **Use supplied object boxes** in **Options**
and select a JSON file; this skips the detector. Optional binary masks and
detector scores follow the documented input format. Existing F32 vocabulary
vectors and their names can replace custom text, avoiding a student download.
Editing those input files at the same path reloads their contents on the next
run when their size or modification time changes.

Build, cache layout, model/data provenance and validation commands are documented
in [qRelateAnything README](../../../plugins/core/Standard/qRelateAnything/README.md)
and its [model card](../../../plugins/core/Standard/qRelateAnything/models/MODEL_CARD.md).
