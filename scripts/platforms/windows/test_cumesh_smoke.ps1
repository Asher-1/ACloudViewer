# Windows CuMesh consumer smoke: configure/build/run
# core/AICore/tests/cumesh-cmake as a plain-CUDA standalone consumer (no Torch,
# no Python), then verify the built binaries are self-contained by running them
# with ONLY Windows system directories on PATH (Conda and CUDA must not mask
# omissions). This is the windows.yml CUDA lane's C++ contract test — it
# cannot live inside Run-CppUnitTests/tests.exe: its subject is
# standalone-ness of a consumer OUTSIDE the main build tree. Extracted here
# so the deployment contract is also exercisable locally:
#   powershell -ExecutionPolicy Bypass -File scripts\platforms\windows\test_cumesh_smoke.ps1
# Requires the MSVC environment (cl.exe / INCLUDE / LIB) and nvcc on PATH.

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$sourceDir = Join-Path $repoRoot 'core\AICore\tests\cumesh-cmake'
$buildBase = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
$smokeBuild = Join-Path $buildBase 'cumesh Windows build smoke'

cmake -S $sourceDir -B $smokeBuild `
    -G 'Ninja Multi-Config' `
    -DCMAKE_CUDA_ARCHITECTURES=75
if ($LASTEXITCODE -ne 0) { throw "CuMesh Windows configure failed: $LASTEXITCODE" }
cmake --build $smokeBuild --config Release --target aicore_cumesh_smoke --parallel 2
if ($LASTEXITCODE -ne 0) { throw "CuMesh Windows build failed: $LASTEXITCODE" }
$savedPath = $env:PATH
try {
    # Load only the DLLs staged with the binaries, plus Windows/MSVC
    # system runtimes; Conda and CUDA on PATH must not mask omissions.
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    & (Join-Path $smokeBuild 'bin\Release\aicore_cumesh_smoke.exe')
    if ($LASTEXITCODE -ne 0) { throw "CuMesh Windows runtime dependency smoke failed: $LASTEXITCODE" }
    & (Join-Path $smokeBuild 'bin\CloudViewer\aicore_cumesh_smoke.exe')
    if ($LASTEXITCODE -ne 0) { throw "CuMesh CloudViewer runtime deployment smoke failed: $LASTEXITCODE" }
} finally {
    $env:PATH = $savedPath
}
Write-Host 'CuMesh Windows smoke passed (configure, build, runtime dependency and deployment layouts)'
