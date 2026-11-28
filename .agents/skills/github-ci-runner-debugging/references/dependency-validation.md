# Dependency failure and validation protocols

## RealSense official firmware endpoint migration

Use this only after the exact current-attempt log identifies the mechanism. A DNS error at the connectivity probe alone proves neither permanent host retirement nor failure of every actual firmware payload.

1. Preserve full commit SHA, run ID/attempt, failed job ID, full job log, annotations, failing step and first substantive error context. If the job entered `Config and build Wheel` and the log shows repeated DNS code 6 at the probe, classify it as configuration-stage dependency preflight, not runner capacity, compiler, docs or unrelated model download failure.
2. Read the fixed-version upstream firmware recipe, not just latest upstream:
   - `https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.44.0/common/fw/CMakeLists.txt`
   - For migration corroboration, inspect official upstream sources such as `https://raw.githubusercontent.com/realsenseai/librealsense/master/CMake/connectivity_check.cmake` and their current provenance/redirects. Latest upstream may establish a host change but must not silently change the pinned recipe/version/hashes.
   Record the official evidence for `librealsense.intel.com` → `librealsense.realsenseai.com`. Do not treat quoted logs or an arbitrary mirror as authority for a new endpoint.
3. Read repository dependency rules, the existing patch style in `3rdparty/libpng/libpng.cmake`, and the actual owners:
   - `3rdparty/librealsense/librealsense.cmake` for ExternalProject and child arguments;
   - `3rdparty/librealsense/connectivity_check.cmake` for the probe entry;
   - `3rdparty/librealsense/check_connectivity.cmake` for the retry/fail-stop helper.
   Verify these paths in the current checkout. If an assumed `build_app/librealsense` directory does not exist, report it; locate the configured build or generate a controlled one rather than inventing child-cache evidence. Never edit disposable extracted sources as the durable fix.
4. Compare probe and **actual** firmware URL construction in the pinned upstream recipe. Where that recipe supports `REALSENSE_FIRMWARE_URL`, pass the shared official base via the owning ExternalProject's typed cache arguments:
   ```cmake
   CMAKE_CACHE_ARGS
       -DREALSENSE_FIRMWARE_URL:STRING=https://librealsense.realsenseai.com
   ```
   Preserve all other existing arguments. Have the probe use `${REALSENSE_FIRMWARE_URL}/Releases/connectivity_check`, so it and actual payloads share one base. Confirm the variable type/consumer against current recipe before applying. For another version, TODO: verify its actual supported configuration rather than assuming this contract.
5. Keep the dependency version, all four declared firmware payloads for the observed v2.44.0 recipe, their upstream expected hashes, firmware/TM2 and fatal download/hash behavior unchanged. Do not delete the probe without understanding its contract, disable required firmware, ignore DNS/download errors or substitute newly calculated hashes for upstream expectations. Keep the patch to the owners; inspect `git diff --check`, `git diff --stat` and the full relevant diff.
6. Save a local validation report with **each** declared payload's filename, exact URL, byte count, expected digest/algorithm, observed digest, result and sampling time. Enumerate from the actual pinned recipe; do not invent filenames/digests or rely on a count alone. Download the real bytes from the official new endpoint, not just HEAD/probe responses or cached unrelated files. All declared payloads must pass.
7. Configure through the real parent ExternalProject path. Inspect generated child configure arguments and actual child `CMakeCache.txt`; record the cache's resolved `REALSENSE_FIRMWARE_URL` value and confirm that both child downloads and the actual probe use it. A string in the parent source is not proof the child received it. Use the current repository configuration/target recipe; TODO if unavailable: establish that exact command before asserting a configure pass.
8. Exercise the real connectivity helper and a controlled wrong-hash download using the actual CMake download/validation mechanism. Preserve a negative-control transcript with a deliberately wrong digest, isolated output/cache and nonzero result; do not alter tracked expected hashes. Show that bad payloads remain rejected and failures stop configuration. Report any unexecuted test as pending.
9. Only after the required local report passes, commit/push/create the PR if authorized. Merge only with separate authorization and current scope/head/main review. Save PR receipt and resulting main full SHA. Query `actions/runs?head_sha=<new-full-main-sha>&per_page=100` with pagination; migrate the sole observer.
10. Report independent verdicts: minimal diff review; all real firmware hashes; generated child cache/configure; probe and wrong-hash control; exact target macOS build; downstream directory maintenance/lightweight publish/no-change closure. Local verification is not a real macOS build. A workflow starting is not success; old main success is not new main evidence.

## External model downloads and transient HTTP errors

1. Start from the exact full-SHA failed-job log. Identify whether failure occurred in transport, integrity checking, extraction or model computation. A 502 downloading a model is not evidence that inference failed.
2. Find and reuse the existing downloader owner. Inspect its timeout, retry count/delay, status classification, checksum contract and cancellation behavior before changing anything. Avoid adding a second retry loop that multiplies the existing budget.
3. Retry only explicitly transient conditions supported by the contract (for example the observed 502); preserve fixed expected hashes, permanent-error behavior and bounded budget. A 404 is not a reason for unlimited retries. Cancellation must stop promptly and must not be reclassified as transient failure.
4. When suitable, use a tiny local fault-injection HTTP service with the **real download function**, not a reimplementation. Record request counts, results, exit/errors, elapsed budget and final payload digest for:
   - temporary error(s) then valid payload: succeeds within existing budget;
   - persistent transient errors: fails once the budget is exhausted;
   - permanent 404: fails immediately without transient retries;
   - cancellation during request/backoff: stops without further retries;
   - corrupt payload: remains rejected by the existing checksum path.
   Bind only locally and use isolated temporary output/cache. Derive callable names and test commands from current source; TODO if unavailable: locate the real entry point before claiming coverage. Keep permanent test additions minimal and owner-scoped.
5. Preserve the original failure evidence after fixing. Run narrow local checks then relevant repository gate. On an authorized push, record the new full head SHA, reconcile its expected CI and move the single observer; do not overwrite the old attempt as if it had passed.
