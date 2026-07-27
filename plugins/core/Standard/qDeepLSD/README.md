# qDeepLSD

DeepLSD wireframe field extraction for ACloudViewer — **native C++ GGML**:

```
Image → AICore DeepLSD GGML → distance field + angle → heatmap overlay in DB tree
```

## Build

```bash
cmake -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QDEEPLSD=ON \
  ..
make -j4 QDEEPLSD_PLUGIN
```

DeepLSD GGML sources live in `core/AICore/src/deeplsd/` (in-tree, no external repo path).

## Models

See [models/MODEL_CARD.md](models/MODEL_CARD.md). Default download:

[`deeplsd_wireframe-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DeepLSD/deeplsd_wireframe-f16.gguf)

F32 and Q8_0 variants are also listed in the model combo.

## Usage

1. **Plugins → DeepLSD Wireframe**
2. Select model (downloads on first Run if missing)
3. Pick image from disk or DB tree → **Run**
4. Overlay ccImage is added to the DB tree (optional checkbox)

## References

- [DeepLSD](https://github.com/cvg/DeepLSD) (upstream PyTorch)
- GGML parity: `DeepLSD/cpp/BENCHMARK.md`
