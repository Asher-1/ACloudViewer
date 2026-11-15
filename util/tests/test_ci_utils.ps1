# Standalone unit test for util/ci_utils.ps1 build-diagnostics contracts.
# Extracted from the windows.yml "Check build diagnostics contracts" step:
# the assertions live here (reviewable, runnable locally), the workflow only
# invokes this script. Run locally with PowerShell 7.3+ (the native-command
# exception contract needs pwsh, same shell as the CI step):
#   pwsh -NoProfile -File util/tests/test_ci_utils.ps1
# Covers Invoke-CMakeStage output/exit-code/timing binding, ninja snapshot
# fan-out, Write-SccacheStats categories and performance-claim boundary,
# exception/exit-code preservation and native command semantics.

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repoRoot 'util\ci_utils.ps1')

# Fixture evidence directory: RUNNER_TEMP on CI, a per-process temp dir locally.
$fixtureBase = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
$fixture = Join-Path $fixtureBase 'diagnostics-contract'
$env:RUNNER_TEMP = Join-Path $fixture 'evidence'
$env:GITHUB_STEP_SUMMARY = Join-Path $fixture 'summary.md'
$fixtureBuild = Join-Path $fixture 'build'
New-Item -ItemType Directory -Force -Path $env:RUNNER_TEMP, (Join-Path $fixtureBuild 'child') | Out-Null
Push-Location $fixtureBuild
try {
    'root-CPU' | Set-Content '.ninja_log'
    'child-CPU' | Set-Content 'child/.ninja_log'
    $global:fixtureCode = 7
    $global:fixtureThrow = $false
    function global:cmake {
        $global:capturedArgs = @($args)
        $global:LASTEXITCODE = $global:fixtureCode
        if ($global:fixtureThrow) { throw 'fixture-command-error' }
        'fixture-output'
    }
    $output = Invoke-CMakeStage -Name 'fixture-CPU' -CMakeArgs @('--build', 'path with spaces')
    if ($LASTEXITCODE -ne 7 -or $output -ne 'fixture-output' -or
        $global:capturedArgs.Count -ne 2 -or $global:capturedArgs[1] -ne 'path with spaces') {
      throw 'CMake diagnostics wrapper changed output, arguments or exit code'
    }
    $timingPath = Join-Path $env:RUNNER_TEMP 'build-timings.jsonl'
    $timing = Get-Content $timingPath | ConvertFrom-Json
    if ($timing.exit_code -ne 7 -or $timing.seconds -lt 0 -or
        $timing.arguments[1] -ne 'path with spaces' -or $timing.build_directory -ne $fixtureBuild) {
      throw 'Stage timing is not bound to the command and build directory'
    }
    'root-CUDA' | Set-Content '.ninja_log'
    'child-CUDA' | Set-Content 'child/.ninja_log'
    $global:fixtureCode = 0
    $output = Invoke-CMakeStage -Name 'fixture-CUDA' -CMakeArgs @('--build', '.')
    if ($LASTEXITCODE -ne 0 -or $output -ne 'fixture-output') { throw 'Successful exit code was changed' }
    foreach ($pass in @('CPU', 'CUDA')) {
      $snapshot = Join-Path $env:RUNNER_TEMP "ninja-fixture-$pass"
      if ((Get-Content (Join-Path $snapshot '.ninja_log')) -ne "root-$pass" -or
          (Get-Content (Join-Path $snapshot 'child/.ninja_log')) -ne "child-$pass") {
        throw 'CPU/CUDA pass root or child graph snapshot is missing'
      }
    }
    function global:Invoke-SccacheFixture {
      $global:statsCalls++
      $global:LASTEXITCODE = $global:statsCode
      $global:statsFixture
    }
    $env:SCCACHE_PATH = 'Invoke-SccacheFixture'
    $env:CLOUDVIEWER_COMPILER_LAUNCHER = 'fixture-launcher'
    $global:statsCalls = 0
    $global:statsCode = 0
    $global:statsFixture = '{"stats":{"compile_requests":3,"cache_hits":{"counts":{"C/C++":2},"adv_counts":{"c++ [msvc]":2}},"cache_misses":{"counts":{"CUDA":3,"PTX":1},"adv_counts":{"cuda [nvcc]":3,"ptx [ptxas]":1}},"cache_errors":{"counts":{},"adv_counts":{}}}}'
    $global:LASTEXITCODE = 9
    $report = (Write-SccacheStats -Label 'fixture' 6>&1) -join "`n"
    if ($LASTEXITCODE -ne 9 -or $report -notmatch '\| C/C\+\+ \| 2 \| 0 \| 0 \|' -or
        $report -notmatch '\| CUDA \| 0 \| 3 \| 0 \|' -or $report -notmatch '\| PTX \| 0 \| 1 \| 0 \|' -or
        $report -notmatch '\| cuda \[nvcc\] \| 0 \| 3 \| 0 \|' -or
        $report -notmatch '\| c\+\+ \[msvc\] \| 2 \| 0 \| 0 \|' -or $report -notmatch '2 hits, 4 misses, 0 errors' -or
        (Get-Content (Join-Path $env:RUNNER_TEMP 'sccache-stats.json') -Raw).Trim() -ne $global:statsFixture -or
        (Get-Content $env:GITHUB_STEP_SUMMARY -Raw) -notmatch 'cannot be used as a source-file hit rate or an end-to-end speedup ratio') {
      throw 'sccache categories, raw response, performance claim boundary or exit-code preservation failed'
    }
    foreach ($empty in @('{"stats":{"compile_requests":0}}', '{"stats":{"compile_requests":3,"cache_hits":{"counts":{}},"cache_misses":{"counts":{}},"cache_errors":{"counts":{}}}}')) {
      $global:statsFixture = $empty
      $report = (Write-SccacheStats -Label 'empty' 6>&1) -join "`n"
      if ($LASTEXITCODE -ne 9 -or $report -match '::notice' -or $report -match 'skipped') { throw 'Empty stats were incorrectly reported' }
    }
    foreach ($invalid in @('{invalid', '{}', '{"stats":{"compile_requests":3}}')) {
      $global:statsFixture = $invalid
      $report = (Write-SccacheStats -Label 'invalid' 6>&1) -join "`n"
      if ($LASTEXITCODE -ne 9 -or $report -notmatch 'skipped' -or $report -match '::notice' -or
          (Get-Content (Join-Path $env:RUNNER_TEMP 'sccache-stats.json') -Raw).Trim() -ne $invalid) {
        throw 'Invalid stats were not preserved or changed the exit code'
      }
    }
    $global:statsCode = 5
    $report = (Write-SccacheStats -Label 'failed-stats' 6>&1) -join "`n"
    if ($LASTEXITCODE -ne 9 -or $report -notmatch 'skipped') { throw 'Stats command failure overwrote the upstream exit code' }
    $global:statsCode = 0
    $global:statsFixture = '{"stats":{"compile_requests":0}}'
    $global:fixtureCode = 17
    $global:fixtureThrow = $true
    $thrown = $false
    try { Invoke-CMakeStage -Name 'exception' -CMakeArgs @('--build', '.') }
    catch { if ($_ -notmatch 'fixture-command-error') { throw }; $thrown = $true }
    $lastTiming = Get-Content $timingPath | Select-Object -Last 1 | ConvertFrom-Json
    if (-not $thrown -or $LASTEXITCODE -ne 17 -or $null -ne $lastTiming.exit_code) {
      throw 'Diagnostics overwrote the original exception or exit code'
    }
    $global:fixtureThrow = $false
    $global:fixtureCode = 7
    $beforeCalls = $global:statsCalls
    $env:RUNNER_TEMP = Join-Path $fixtureBuild '.ninja_log'
    $output = Invoke-CMakeStage -Name 'failed-write' -CMakeArgs @('--build', '.')
    if ($LASTEXITCODE -ne 7 -or $output -ne 'fixture-output' -or $global:statsCalls -ne ($beforeCalls + 1)) {
      throw 'Diagnostics write failure affected the build or prevented cache stats collection'
    }
    $env:RUNNER_TEMP = ''
    $env:GITHUB_STEP_SUMMARY = ''
    $global:fixtureNative = (Get-Process -Id $PID).Path
    function global:cmake { & $global:fixtureNative -NoProfile -Command "Write-Output 'native-output'; exit 7" }
    $PSNativeCommandUseErrorActionPreference = $false
    $output = Invoke-CMakeStage -Name 'native-exit' -CMakeArgs @('--build', '.')
    if ($LASTEXITCODE -ne 7 -or $output -ne 'native-output') { throw 'Native process exit code or output was changed' }
    $PSNativeCommandUseErrorActionPreference = $true
    $thrown = $false
    try { Invoke-CMakeStage -Name 'native-exception' -CMakeArgs @('--build', '.') | Out-Null }
    catch [System.Management.Automation.NativeCommandExitException] { $thrown = $true }
    if (-not $thrown -or $LASTEXITCODE -ne 7) { throw 'Native exception exit code was overwritten' }
} finally { Pop-Location }
$global:LASTEXITCODE = 0
Write-Host 'Build diagnostics contracts passed (files, categories, exceptions, write failures, native exit codes)'
