# OCR labelled input provenance

## Scanned receipt pages

- Source: [ICDAR-2019-SROIE contributor corpus](https://github.com/zzzDavid/ICDAR-2019-SROIE/tree/27be4271b251c256f695acbade9a801bffe85994/data), commit `27be4271b251c256f695acbade9a801bffe85994`.
- Selection: `data/img/000.jpg` through `049.jpg` and their original `data/box/000.csv` through `049.csv`, selected before inference. These are 50 complete receipt pages with 2,779 localisation/transcription annotations. This corrected contributor subset is not the official held-out competition test split.
- Integrity: the common `validation_manifest.json` input catalog pins all 100 files independently by immutable URL, byte count and SHA-256. The label-directory digest is `2d1c39b5fc6d44ce1565439f05adf7b23f1357542bf15ea0a80e11b5ae60ef4e`: SHA-256 over sorted CSV basename, NUL, and binary SHA-256 of each file. Image identities are verified separately.
- Storage: shared validation asset cache under `ocr_sroie_50/<source-commit>/{box,img}`. The common runner downloads and verifies this data; the probe owns neither downloading nor a second catalog. Images and transcriptions are not vendored into production packages.
- Evaluation: original quadrilaterals and text are retained, including commas and punctuation. The generic evaluator uses greedy one-to-one quadrilateral IoU matching at 0.5, micro-averaged Hmean, fixed-GT-region CER/WER, and complete-pipeline CER/WER. This protocol is not the competition's official DetEval score. Each native metric is compared with the unmodified, pinned X-AnyLabeling source on the same pages; source predictions never replace ground truth.

## Independent receipt gate for the published Paddle models

Receipts `050.jpg` through `099.jpg` and their original CSV labels are a
second, disjoint 50-page set from the same pinned contributor revision.
The first set (`000`–`049`) helped select the Q8_0 detector/English-recognizer
storage repairs; it therefore provides regression evidence. The second set
was fixed before evaluating those repaired files, and its outcomes must not
be used to tune that policy. This separation concerns our conversion work;
it does not establish that the source model never saw these pages in training,
and neither set is the official competition test split.

The manifest pins all 100 additional files by immutable URL, size and SHA-256.
The label-directory digest is
`8f085fedd59d4421352b4e1b37839f55515b74f75a89eae183fd810b03ecdc47`.
Files resolve under `ocr_sroie_holdout_50/<source-commit>/{box,img}` through
the common runner. The six `ppocr-heldout-*` scenarios cover Chinese and
English F32/F16/Q8_0 pipelines, retaining all detection and recognition gates.
Declaring these rows does not assert that inference, controlled performance,
or publication has passed.

## Other OCR inputs

The manifest also pins the PaddleOCR/PaddleX 50-image scene validation archive
and its 201 non-ignore word boxes, plus Chinese/Japanese XFUND validation pages
and labels. Scene word-level annotations and model line outputs have different
granularity. XFUND semantic-field regions are used only for fixed-region
recognition, including multiline fields; they do not certify text-line
detection or automatic KIE. Dataset-specific scores must retain these scopes.

## Derived recognizer regression crop

The PP-OCRv5 recognizer probability scenarios additionally consume the pinned
scene archive's `images/val_img_7.jpg`, using the source quadrilateral
`[1132,353,1195,353,1195,368,1132,368]`. Source perspective cropping yields
a 63 x 15 BGR image before recognizer normalization. This case was selected
after a page comparison exposed a confidence/CTC difference; it is a regression
input, not an independent held-out accuracy sample. The fixture exporter records
the page SHA, quadrilateral, crop shape and decoded-pixel SHA. All three v5
recognizer storage scenarios require five inputs, including this crop, while
the 50-page dataset scores remain separate. No generated crop is installed
with the application or privately downloaded by a probe.

## Experimental Tiny Q8 calibration

The v19 conversion experiment uses the first 256 non-`###` quadrilaterals in
`ocr_det_dataset_examples/train.txt`, in original file/annotation order. They
span 56 training images. The common PaddleX archive is pinned as
`a1b5ff0165e33effcd0ed5294130e6cf3212484d091d007b5e0b31b3056a9426`
(22,773,760 bytes); the original training-label SHA-256 is
`ddb39c7f4726e8f329cb0bef3c3ddc17f3b8b0e6a19b7c9fa2da4058517e526f`.
The conversion policy pins each selected image SHA and quadrilateral. The
converter verifies the image before applying the source perspective crop and
recognizer normalization, then observes the original ONNX graph on CPU.
Calibration inputs are external assets, never installed application resources.

This calibration split is separate from the 50 scene validation pages, four
word probability fixtures and 50 receipt pages. Kernel/damping choices were
compared using those probability fixtures, and validation-page failures guided
candidate revisions; those acceptance results are regression evidence, not an
untouched held-out estimate. The policy records the selected algorithm and
damping. Its source-only diagnostic is
`build_app/Testing/ocr-storage-reference/tiny-rec-training-activation-q8-diagnostic.json`.
Byte-identical conversion replay, backend tests and page scores remain
separate gates; calibration does not certify release accuracy or throughput.
