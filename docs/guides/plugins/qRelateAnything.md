# RelateAnything scene relations

Enable `AICore_ENABLED` and `PLUGIN_STANDARD_QRELATEANYTHING`, then open the
RelateAnything plugin. It detects objects with the existing YOLO inference task
and displays subject–predicate–object relations with scores and an annotated
preview. Choose among all nine relation models in F32/F16/Q8_0 precision.

**Try sample data** downloads and verifies the shared RelateAnything archive. One
click fills both the six-image list and the video input. Select images or process
the whole list; use the video tab for video files or cameras. Results can be
filtered, exported as JSON/CSV/text, or added as an annotated image to the DB.

Optional custom predicates are separated with `|`. The matching text student
and official tokenizer are downloaded through the same shared model cache.
The decomposition option exposes upstream spatial and semantic relations.
Default arithmetic matches upstream; enable **Use precise model arithmetic**
for its strict F32 mode.

Build, cache layout, model/data provenance and validation commands are documented
in [qRelateAnything README](../../../plugins/core/Standard/qRelateAnything/README.md)
and its [model card](../../../plugins/core/Standard/qRelateAnything/models/MODEL_CARD.md).
