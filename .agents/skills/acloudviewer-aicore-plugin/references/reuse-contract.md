# AICore Reuse and Ownership Contract

## The Reuse Test

Before adding a helper, name the two consumers and the invariant they share.
Promote code only when the consumers have the same input ownership, error
semantics, lifecycle, and compatibility requirements. A wrapper with one
caller is not an abstraction requirement.

| Responsibility | Owner | Plugin may do |
|---|---|---|
| Device discovery, leases, queues, cancellation, shutdown | AICore runtime/common | Request a lease and release it through the public API |
| Image storage, stride, channel order | `aicore_image_view` and task adapter | Borrow caller-owned decoded storage |
| Graph, preprocess, postprocess, model session/cache | `src/tasks/<task>` | Pass options and consume typed results |
| Model URL, digest, cache folder, role | task catalog/cache API + digest header | Read exported catalog/cache entries |
| Download transport/integrity/data root | AICore catalog + shared plugin services | Use `ecvModelDownloader` / `ecvAssetIntegrity`; do not implement transport |
| UI, source selection, DB entities, rendering/export | Qt plugin | Adapt typed results to application objects |
| Accuracy/stability/performance | probe + validation manifest | Add rows and consume report evidence |

Do not create a plugin model source of truth, private downloader implementation,
cache root, device lock, backend registry, JSON/file inference transport, or
private validation runner. A thin Qt projection of public catalog rows and
orchestration of `ecvModelDownloader` are allowed. Do not put model mathematics
into `src/common/` merely because two task files look similar.

## Static Boundary Check

Run this for every AICore-dependent plugin touched by a change:

```bash
python3 core/AICore/tests/check_plugin_aicore_boundaries.py \
  --root . --plugin qRMBG --strict
```

The check rejects private ggml/task headers, environment-controlled production
behavior, developer absolute paths, direct digest-table use, private Qt network
transport, and hard-coded model-release URLs. Catalog adapters that only copy
public AICore rows and shared integrity/download helpers are permitted. The
repository-wide strict form must pass before merge.

## C ABI and Data Flow

Public headers under `core/AICore/include/aicore/` expose opaque contexts,
explicit options, typed result accessors, release functions, `last_error`,
readiness, and common timing. Do not expose Qt, STL, OpenCV, exceptions, or
ggml types. Extensible structs validate `struct_size` and `abi_version`.

The hot path is:

```text
caller-owned decoded image -> borrowed image_view -> task session
  -> typed result -> plugin rendering/export
```

Use actual `bytesPerLine()` and channel format. JSON, encoded images, paths,
and temporary files are compatibility/export boundaries only. A context owns
its session/cache state; the plugin worker owns the context thread and carries
source identity plus generation through every result.

## Review Signals

- One public task contract is reused instead of a product-name task clone.
- One owner exists for every model, cache, device, timing, and error fact.
- Still and live paths share image/result ownership rules.
- Cancellation, stale-generation suppression, and teardown are tested.
- A typed result reaches render, metadata, and export consumers without a
  placeholder label or a second semantic mapping.
