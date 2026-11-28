# Unified C++ testing

This module owns the repository's first-party C++ test registration contract,
build closures, capability groups and CI execution entry points. Production
modules own their test sources and real link dependencies. A single binary is
not required: Qt event loops, GPU processes, ABI probes and ordinary algorithm
assertions have different isolation requirements.

## Architecture decision and research

The starting full GUI build had 277 CTest entries, 249 without labels. The
CVViewer GoogleTest executable was built but not registered with CTest. Several
CI scripts executed it directly and separately maintained plugin, foundation and
reconstruction lists. Adding another script-specific gate could therefore miss
a new test even if that test compiled successfully.

The following upstream implementation sources were inspected on 2026-11-24.
These are moving upstream branches, not pinned dependencies or copied code:

| Source | Observed mechanism | Applicable consequence |
| --- | --- | --- |
| [Open3D tests](https://github.com/isl-org/Open3D/blob/main/cpp/tests/CMakeLists.txt) | One algorithm runner, component directories add sources, target dependencies are explicit | Preserve the existing CVViewer source aggregation where dependency/lifetime requirements match |
| [LLVM unit tests](https://github.com/llvm/llvm-project/blob/main/llvm/unittests/CMakeLists.txt) and [regression entry](https://github.com/llvm/llvm-project/blob/main/llvm/test/CMakeLists.txt) | Root UnitTests aggregate and check targets build test dependencies before execution | Derive a repository-wide build closure from the same registration used to run tests |
| [Qt auto tests](https://github.com/qt/qtbase/blob/dev/tests/auto/CMakeLists.txt) | Component tests and platform/capability conditions, separate processes | Keep GUI/device constraints explicit instead of merging incompatible process lifetimes |
| [GoogleTest CMake](https://github.com/google/googletest/blob/main/googletest/CMakeLists.txt) | Test targets register with CTest and are selectable with native CTest filters | Reuse CTest rather than inventing another result protocol or framework |
| [CMake tests](https://github.com/Kitware/CMake/blob/master/Tests/CMakeLists.txt) | Root orchestration, isolated test homes, explicit build-and-test prerequisites | Test state must not depend on the maintainer's settings or silently reuse unavailable prerequisites |

The minimum information for one runnable test is its command, build targets,
owner, conditions and result. CMake already owns targets and CTest owns commands,
environment, timeouts, fixtures and exit status. The added contract only connects
those facts and derives groups; it does not parse generated compiler commands
or introduce a second runtime runner. First-party C++ unit assertions use
GoogleTest. Qt remains the production event/control implementation under test;
QtTest assertions, QObject slot-based test discovery and QTEST_MAIN have been
removed from first-party unit suites. qMPlane uses GoogleTest/gMock through the
same root dependency instead of its former Catch2/Trompeloeil copies. External
CLI and numeric-model probes remain integration/validation programs with their
original argument, output, prerequisite and accuracy contracts.

## Scope and infrastructure fixtures

This module owns execution/registration for enabled first-party C++ suites.
Test sources remain with production owners; other repository test languages
retain their own entry points. Vendored QtTest sources such as
`app/QSimpleUpdater/tests`, standalone compiler/loader/CUDA fixture projects,
Python/package tests and strict model validation have distinct scopes. They
are not evidence of GoogleTest unit coverage or implicitly part of headless CI.
Historical test sources disabled in owner CMake lists also remain disabled;
the catalog describes actual configured registrations, not all files named test.

`contracts/app/tests` is a synthetic application owner in the
standalone CMake contract project. Its path exercises app owner classification
and the raw-registration audit. `app_registration_probe` only proves
prerequisite compilation; it has no semantic unit assertions. The
`fixture-app-registration` command checks automatic CI/suite selection.
`without-app-registration` checks that removal needs no CI edits; `bypass`
must reject raw `add_test()`. The non-app owner fixture cannot establish
application path classification, so this small fixture remains necessary.

## Registration

Include `tests/Testing.cmake` from the repository root before component CMake
files. Register first-party tests using:

```cmake
cloudviewer_add_test(NAME example COMMAND example
    LABELS "unit")
```

The command retains normal named `add_test` semantics, including configuration,
working directory and list expansion. Direct target commands and references
such as `$<TARGET_FILE:plugin>` are captured as build dependencies. A script
that launches a native target indirectly must declare `BUILD_TARGETS target`.
An external asset is not a build dependency: its test must declare the relevant
`model`, `network` or `e2e` label and retain its own truthful prerequisite gate.

Registration is finalized at the end of the owning CMake directory, after its
labels and targets exist. Native entries receive `cpp` and a suite label.
Model/device processes share a CTest resource lock. Existing fixture,
skip-return-code, environment and timeout contracts remain in CTest.

Third-party subprojects and isolated compiler-contract fixture projects retain
their native testing systems. They are not implicitly imported as application
unit tests. Model accuracy/performance runners retain their task-specific
numeric and asset contracts; a generic C++ smoke selection cannot replace them.

## Build and run

The root `tests/CMakeLists.txt` is added after component registration. It exports
`<build>/tests/catalog-<config>.tsv`, containing owner, targets and final labels.
The same records determine both build prerequisites and test selection:

| Target | Selection |
| --- | --- |
| `cpp-tests` | Build all registered test prerequisites |
| `cpp-tests-ci` | Build the environment-independent CI selection, including script contracts |
| `check-cpp` | Build and run `test-ci` tests; absence is an error |
| `check-cpp-graphics` | Build and run explicitly labelled graphics tests |
| `check-cpp-model` | Build and run explicitly labelled model tests |
| `check-cpp-gpu` | Build and run explicitly labelled GPU tests |
| `check-cpp-all` | Build and run every registered command except infrastructure fixtures |

Groups with no configured tests do not produce a successful empty check target.
CI callers use the same script, which checks the CMake infrastructure first,
then builds and executes the C++ group without overlapping nested build budgets:

```bash
cmake -DBUILD_DIR=build_app -DCONFIG=Release -DJOBS=6 -P tests/Run.cmake
```

`JOBS` accepts integers from 1 through 12. The CI group excludes `model`, `gpu`,
`e2e`, `graphics`, `network`, `benchmark` and `slow` labels. Owners must identify
real external requirements; absence of a label is not evidence that a test is
hermetic. Tests execute serially by default;
CMake's outer build parallelism remains under the caller's budget. All targets
are configuration-aware, use native CTest status, and fail on an empty selection.

## Migration status

The root registration/catalog contract is active. Initial migration acceptance had
511 CTest commands: 480 reference native build targets and 340 belong to the
default C++ CI group. Comparing registration names against the original 277-entry
baseline found no missing entries. The added registrations include AICore
white-box coverage and qMPlane's previously disconnected suite.

The unified `Run.cmake` build and default CTest selection completed with zero
failures: 340 registered commands, 335 passed and 5 prerequisite skips (the YOLO
matrix and four InstantMesh upstream-reference contracts). Missing real model or
reference assets remain visible as skips; this is not full model-accuracy or
performance acceptance. The complete gate also passed with GoogleTest shuffle
seed 4242 after making the random-data and queued-worker fixtures independent
of execution order; those four affected suites passed three shuffled repeats.
All three shortcut registrations, including Xvfb graphics delivery, passed in a
focused run. qMPlane's suite passes with its inherited empty active-display
placeholder explicitly skipped inside GoogleTest. The CMake contract fixture
remains independently runnable:

Configuration checks passed with all test options disabled, AICore-only tests,
VTK-only tests, and system GoogleTest preference enabled (falling back to the
bundled framework when the system installation lacks gMock). The configured
application uses bundled GoogleTest 1.16.0. Isolated Ninja Multi-Config fixtures
also built and ran `check-cpp` in Debug and Release. These checks ran on Linux,
GCC 9.4 and Qt 5.15.2; they do not establish Windows, macOS or Qt 6 runtime
acceptance.

```bash
cmake -DSOURCE_DIR="$PWD/tests/contracts" \
  -DOUTPUT_DIR=/tmp/cloudviewer-test-contracts \
  -P tests/contracts/verify.cmake
```


## Adding a module without changing CI

Keep test sources beside their production owner, including `app/tests/` for
application widgets. Add that directory to its owner's CMake once under the
appropriate feature/test condition. Use the root GoogleTest dependency for C++
unit assertions and `cloudviewer_add_test()` for each executable command.
Declare `BUILD_TARGETS` when a script launches a native prerequisite indirectly.
There is no CI-maintained module or executable list to update.

The root catalog includes registered script contracts as well as native tests.
`cpp` and `cpp-*` identify native commands; `cloudviewer-test` and `test-*`
select the complete execution policy. This preserves AICore's Python-only
contract checks when CI invokes the common entry. Derived profile labels are
reserved: root registration clears and recomputes them, so a stale hand-written
`cpp-ci` or `test-ci` cannot override a `graphics` requirement.

Use `PROFILE=ci` (the default), `graphics`, `gpu`, `model`, or `all` through the
same entry. Comma-separated profiles select their union and execute each command
once. For example, a GPU-capable container uses:

```bash
cmake -DBUILD_DIR=build -DJOBS=4 -DPROFILE=ci,gpu -P tests/Run.cmake
```

The script rejects disabled tests, invalid profiles/budgets and empty requested
capabilities. It runs infrastructure contracts first, builds the derived
prerequisites, and executes CTest serially. Model tests retain their prerequisite
skip semantics; this is separate from the strict real-asset validation gate.
Whole-tree Python/wheel tests and package/CLI smoke checks retain their own
purposes and remain separate from these registered build-tree contracts.

Headless widget logic uses offscreen Qt. Any command that creates a real OpenGL
context must declare `graphics`; the default gate excludes it. Layout closure
and the VTK orientation-widget observer regression require graphics, while
ordinary shortcut dispatch remains in the default profile. On displayless Linux,
the entry reuses `xvfb-run` when installed because some Vulkan drivers need an X
connection during device discovery even without rendering. This does not add
graphics commands to the default selection. Explicit graphics runs require the
configured graphics prerequisites, including Xvfb/xcb for the Linux VTK suites.

Ubuntu, macOS and Windows app CI call the same root interface. Docker selects
capabilities rather than reconstruction test names; AICore runtime ABI/device
discovery is registered with the library target as its build prerequisite.
The macOS loader fixture is part of the root infrastructure gate and remains
Apple-only. Infrastructure fixtures verify late graphics labels, shared native
targets, script-only profiles, new app module discovery/removal, explicit script
build prerequisites, raw registration bypasses, and every public profile.

This guarantees automatic execution for enabled modules that register their
tests. It cannot discover a source directory that its owner never adds to CMake.
It also does not expand the hosted platform matrix: Windows CUDA/Debug and wheel
jobs that disable native testing do not become covered by changing the entry.

Follow-up local acceptance (2026-11-24): the catalog contains 512 registered
commands (482 native plus 30 script contracts), with one additional infrastructure
CTest fixture. With host display variables cleared and GoogleTest shuffle seed
4242, the root `ci` entry passed 364 commands and recorded five missing-asset
skips (369 selected, zero failures; 155.31 seconds). The separate root `graphics`
entry passed all five commands, including all three layout-close regressions
(12.13 seconds). The layout test initializes VTK's default surface format before
QApplication, matching the application's OpenGL context contract. The shell
helper contract passed all 22 checks. Workflow YAML parsing, shell syntax and
`git diff --check` passed; original 277 test registration names were retained.
These results are Linux evidence; Windows/macOS execution and real-model GPU
numeric validation were not rerun for this integration follow-up.


## Common execution and framework enforcement

`check-cpp*` targets are forwarding commands into [Run.cmake](Run.cmake).
They therefore share prerequisite builds, the bounded compilation budget,
serial CTest execution, infrastructure verification and display setup with CI.
The legacy AICore fast/contract/model and YOLO matrix commands also forward to
this script with their original label selection; their prerequisite closures
are derived in [AICoreTiers.cmake](AICoreTiers.cmake). They no longer maintain
separate `-j2`/`-j4`, timeouts or display policies. Forwarded commands default to
four compilation jobs; an outer `cmake --build --parallel` does not set the
inner script's budget. Call the script with `JOBS` when selecting that budget.

Native unit-test targets must resolve GoogleTest through their actual link
dependencies, including aliases and interface wrappers. Real process probes
may explicitly declare target property `CLOUDVIEWER_TEST_KIND=integration`.
This marker describes an external CLI/UI/model contract and must not be used
to bypass a unit suite's framework requirement. Linkage is a configure-time
boundary check: it does not prove that every assertion in a historical
handwritten `main()` has been converted. Some AICore validation consumers
still use the original `AICORE_CHECK`/stdout/exit-code probe protocol; those
are remaining assertion-migration work, not newly established GoogleTest
case coverage. Model validation remains separately strict about assets and
numeric reference evidence.

The infrastructure fixture reuses its dedicated build directories and never
recursively deletes a caller-provided output path. It rejects source-contained,
source/repository-ancestor and filesystem-root output paths. Repeated successful
runs verify reuse, while positive transitive-GoogleTest and negative non-GoogleTest
fixtures verify the framework check.

CVViewer is included through its owner's single `cvviewer` registration. Its
existing sources, CMake source lists, assertions, case names, test-data downloads
and execution behavior are preserved. Consolidation does not enable dormant
sources, replace external assets, add skip gates or repair production algorithms.
The migrated qDA3 model assertions are owned by
[AICore](../core/AICore/tests/depth/test_depth_capi_model_contract.cpp);
missing inputs skip truthfully and do not establish real-model acceptance.


## Historical acceptance before CVViewer source restoration (2026-11-24)

The counts below describe the earlier expanded-source run, not the current
registration-only CVViewer integration. All implementation and source-list
changes to CVViewer tests were subsequently reverted.

The historical catalog had 515
ordinary commands (485 native and 30 script contracts), plus the infrastructure
fixture. All 277 original registration names remain present. On Linux with host
display variables cleared and GoogleTest shuffle seed 4242, the common entry
with `PROFILE=ci,graphics` selected 375 commands: 370 passed, five missing-asset
commands skipped, and none failed (204.20 seconds). The selected commands include
370 default-profile registrations and all five graphics registrations. The
restored ordinary CVViewer suite passed within that run (35.28 seconds).

The expanded infrastructure fixture passed all four AICore compatibility groups,
every public forwarding target, transitive GoogleTest acceptance, non-GoogleTest
unit rejection, and rejection of source/repository/root output paths. Shell
helper contracts passed 22/22 checks. The migrated depth model suite and
InstantMesh VAE stage runner preserve exit 77 for missing prerequisites; explicitly
supplied invalid depth assets fail with exit 1 instead of becoming skips.

These historical results predate the scope correction and are not acceptance
of the restored CVViewer source set. All CVViewer implementation and source-list
ports have been reverted. The `cvviewer-slow` and `cvviewer-assets` registrations
were removed; the original suite is registered once with its existing optional
`LOW_MEM_USAGE` filter. Real-model accuracy, GPU/SYCL, Windows/macOS and Qt 6
behavior require their own runtime evidence. GoogleTest linkage alone is not
proof of assertion conversion.
