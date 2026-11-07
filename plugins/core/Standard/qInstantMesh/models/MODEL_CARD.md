# MODEL_CARD — InstantMesh GGUF (Asher-1/InstantMesh-GUFF)

Served by the AICore InstantMesh task
(`core/AICore/src/tasks/instantmesh/model_catalog.cpp` — single source of
truth for URL, size, SHA-256 and role). All files are Hugging Face LFS
objects; the oid column is the pinned content digest.

| File | Role | Size (bytes) | SHA-256 (LFS oid) |
|---|---|---|---|
| dino_f16.gguf | dino | 230,145,408 | ab09647993c9…10dd88f |
| dino_q8.gguf | dino | 123,424,128 | d36498038695…969b03 |
| dino_f32.gguf | dino | 459,021,696 | 388248c31e86…7cfe85 |
| lrm_transformer_f16.gguf | transformer | 527,786,304 | f49a3fa0b12e…d9fdea |
| lrm_transformer_q8.gguf | transformer | 283,992,384 | 3e0b458c43bd…05d1a3 |
| lrm_transformer_f32.gguf | transformer | 1,054,826,816 | 6a644ce6c4e9…c9b03f |
| synthesizer_f16.gguf | synthesizer | 412,384 | 712ac79db8c5…283521 |
| synthesizer_q8.gguf | synthesizer | 264,864 | e1419a06dbad…8ece691b |
| synthesizer_f32.gguf | synthesizer | 819,424 | b6396e6fd1f7…edfc39d |
| zero123pp_cond_f32.gguf | cond | 2,528,739,904 | 1c810da58151…94ec128 |
| zero123pp_unet_f16.gguf | unet | 1,732,768,320 | 983602d7bfd8…295222 |
| zero123pp_unet_f32.gguf | unet | 3,463,702,080 | f07135c64744…371d3e |
| zero123pp_vae_f16.gguf | vae | 167,459,584 | b91feec6ab80…306baa1 |
| zero123pp_vae_f32.gguf | vae | 334,635,424 | 6399f42fdf22…67218f4 |

Full digests: `core/AICore/include/aicore/asset_digests.h` (InstantMesh
section).

- **Licenses**: InstantMesh weights Apache-2.0 (TencentARC/InstantMesh);
  Zero123++ weights Apache-2.0 (stabilityai/zero123plus-v1.2); ggml
  conversion MIT (InstantMesh-GGML).
- **Recommended chain**: f16 (dino/lrm/synthesizer f16 + Zero123++ unet/vae
  f16 + cond f32) — ~5.2 GB total.
- **q8 chain**: dino/lrm/synthesizer q8 (Zero123++ has no q8 release and
  stays f16) — ~4.0 GB.
- **RMBG**: not in this card; the RMBG-2.0 GGUFs are owned by the in-tree
  RMBG task catalog (`aicore_rmbg_*`) and shared with qRMBG/qTrellis —
  identical LFS objects, one physical cache.
