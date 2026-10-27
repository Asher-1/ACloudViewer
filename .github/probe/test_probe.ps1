# 隔离工具测试：子 Ninja graph 覆盖、缺失对象及 native 编译失败必须失败。
$ErrorActionPreference = 'Stop'
$original = @{}
foreach ($name in 'RUNNER_TEMP', 'PROBE_SNAP', 'PROBE_TOOLS') { $original[$name] = [Environment]::GetEnvironmentVariable($name) }
$root = Join-Path ([IO.Path]::GetTempPath()) ('probe-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
try {
    $env:RUNNER_TEMP = $root
    $env:PROBE_SNAP = Join-Path $root 'snap'
    $env:PROBE_TOOLS = $PSScriptRoot
    Import-Module (Join-Path $PSScriptRoot 'probe.psm1') -Force
    $build = Join-Path $root 'build'
    $child = Join-Path $build 'external'
    New-Item -ItemType Directory -Path $child -Force | Out-Null
    $ninja = (Get-Command ninja).Source
    foreach ($directory in $build, $child) {
        Set-Content (Join-Path $directory 'CMakeCache.txt') "CMAKE_MAKE_PROGRAM:FILEPATH=$ninja"
        $output = if ($directory -eq $build) { 'root.cu.obj' } else { 'child.cu.obj' }
        $text = @"
rule cuda
  command = mock/sccache.exe mock/nvcc.exe -c source.cu -o `$out
build $output`: cuda
"@
        $graph = if ($directory -eq $build) { 'build-Release.ninja' } else { 'build.ninja' }
        Set-Content (Join-Path $directory $graph) $text
    }
    $objects = @('root.cu.obj', ('external' + [IO.Path]::DirectorySeparatorChar + 'child.cu.obj'))
    $commands = @(Get-VanillaCommands $build $objects)
    if ($commands.Count -ne 2 -or $commands[0].cwd -ne $build -or $commands[1].cwd -ne $child) { throw '未覆盖实际子构建工作目录' }
    $failed = $false
    try { Get-VanillaCommands $build @('absent.cu.obj') | Out-Null } catch { $failed = $true }
    if (-not $failed) { throw '缺失命令被放行' }
    $failed = $false
    try { Save-Snapshot $build @('absent.cu.obj') 'missing' | Out-Null } catch { $failed = $true }
    if (-not $failed) { throw '缺失对象被放行' }
    $pwsh = (Get-Process -Id $PID).Path
    $negative = @([pscustomobject]@{ obj = 'failed.cu.obj'; cwd = $build; cmd = ('"' + $pwsh + '" -NoProfile -Command "exit 7"') })
    $failed = $false
    try { Invoke-Vanilla 'negative' $build $negative | Out-Null } catch { $failed = $true }
    if (-not $failed) { throw 'native 编译失败被放行' }
    Write-Host 'PowerShell 全量命令/失败门禁测试通过'
    $global:LASTEXITCODE = 0
} finally {
    foreach ($name in $original.Keys) { [Environment]::SetEnvironmentVariable($name, $original[$name]) }
}
