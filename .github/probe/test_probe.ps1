# 隔离工具测试：子 Ninja graph 覆盖、缺失对象及 native 编译失败必须失败。
$ErrorActionPreference = 'Stop'
$original = @{}
foreach ($name in 'RUNNER_TEMP', 'PROBE_SNAP', 'PROBE_TOOLS', 'SCCACHE_PATH') { $original[$name] = [Environment]::GetEnvironmentVariable($name) }
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
    # 用真实子进程验证完整命令选择、原 cwd、记录及失败传播；统计为明确的 fixture。
    $env:SCCACHE_PATH = 'Test-Sccache'
    $fixtureStats = @{ stats = @{
        compile_requests = 2; cache_hits = @{ counts = @{ 'CUDA' = 2; 'CUDA (Device code)' = 2; PTX = 10; CUBIN = 10 } }
        cache_misses = @{ counts = @{} }; cache_errors = @{ counts = @{} }
        cache_timeouts = 0; cache_read_errors = 0; cache_write_errors = 0; compile_fails = 0; forced_recaches = 0
    } }
    $global:ProbeStatsFixture = $fixtureStats | ConvertTo-Json -Depth 8 -Compress
    function global:Test-Sccache {
        $global:LASTEXITCODE = 0
        if ($args -contains '--show-stats') {
            if ($global:ProbeFailStats) { $global:LASTEXITCODE = 7; return }
            $global:ProbeStatsFixture
        }
    }
    function Mock-CompileCommand([string]$Output, [int]$Timestamp) {
        $script = "`$b = [byte[]]::new(20); `$b[0] = 100; `$b[1] = 134; `$b[4] = $Timestamp; [IO.File]::WriteAllBytes('$Output', `$b)"
        '"' + $pwsh + '" -NoProfile -EncodedCommand ' + [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
    }
    $positive = @(
        [pscustomobject]@{ obj = $objects[0]; cwd = $build; cmd = (Mock-CompileCommand 'root.cu.obj' 1); original = (Mock-CompileCommand 'root.cu.obj' 2); launched = $true }
        [pscustomobject]@{ obj = $objects[1]; cwd = $child; cmd = (Mock-CompileCommand 'child.cu.obj' 1); original = (Mock-CompileCommand 'child.cu.obj' 2); launched = $true }
    )
    $bare = @(Invoke-Vanilla 'positive' $build $positive)
    Save-Snapshot $build $objects 'bare' | Out-Null
    $warm = @(Invoke-Cached 'positive' $build $positive)
    Save-Snapshot $build $objects 'W1' | Out-Null
    Assert-WarmStats 'positive' 2
    foreach ($pair in @(@{ rows = $bare; field = 'cmd' }, @{ rows = $warm; field = 'original' })) {
        if ($pair.rows.Count -ne 2) { throw '命令执行覆盖缺失' }
        foreach ($row in $pair.rows) {
            $command = $positive | Where-Object obj -eq $row.obj
            if ($row.executed -cne $command.($pair.field) -or $row.cwd -cne $command.cwd -or $row.rc -ne 0 -or $row.sec -le 0) {
                throw '实际执行命令/cwd/计时未绑定'
            }
        }
    }
    Invoke-Cached 'second' $build $positive | Out-Null
    foreach ($tag in 'positive', 'second') {
        $path = Join-Path $root "probe/phase-$tag.json"
        python -c 'import datetime,json,sys; p=json.load(open(sys.argv[1],encoding="utf-8-sig")); a,b=(datetime.datetime.fromisoformat(p[k].replace("Z","+00:00")) for k in ("started_utc","ended_utc")); assert b>a and p["objects"]==2 and p["wall_sec"]>0 and p["parallelism"]>0; assert abs((b-a).total_seconds()-p["wall_sec"])<=max(.05,p["wall_sec"]*.0001)' $path
        if ($LASTEXITCODE -ne 0) { throw '真实 phase 时间/数量/并行度无法由 Python 审计' }
    }
    Save-Snapshot $build $objects 'W2' | Out-Null
    Assert-SnapshotReplay 'W1' 'W2' $objects
    $failed = $false
    try { Assert-SnapshotReplay 'bare' 'W1' $objects } catch { $failed = $true }
    if (-not $failed) { throw '不同对象字节被放行' }
    $failed = $false
    try { Assert-SnapshotReplay 'W1' 'W2' @($objects[0]) } catch { $failed = $true }
    if (-not $failed) { throw '少检查一个对象被放行' }
    [IO.File]::WriteAllBytes((Join-Path $env:PROBE_SNAP 'W2/root.cu.obj'), [byte[]]::new(20))
    $failed = $false
    try { Assert-SnapshotReplay 'W1' 'W2' $objects } catch { $failed = $true }
    if (-not $failed) { throw '摘要之后对象被篡改仍放行' }
    $failed = $false
    try { Invoke-Vanilla 'duplicate' $build @($positive[0], $positive[0]) | Out-Null } catch { $failed = $true }
    if (-not $failed) { throw '重复命令被放行' }
    foreach ($kind in 'cached-failure', 'cached-missing', 'cached-launcher') {
        $bad = [pscustomobject]@{ obj = 'absent.cu.obj'; cwd = $build; cmd = $negative[0].cmd; original = $negative[0].cmd; launched = $true }
        if ($kind -eq 'cached-missing') { $bad.original = ('"' + $pwsh + '" -NoProfile -Command "exit 0"') }
        if ($kind -eq 'cached-launcher') { $bad.launched = $false }
        $failed = $false
        try { Invoke-Cached $kind $build @($bad) | Out-Null } catch { $failed = $true }
        if (-not $failed) { throw "$kind 被放行" }
    }
    # stats 子进程失败必须先保留完整快照和逐对象记录，再传播原始错误。
    $global:ProbeFailStats = $true
    $failed = $false
    try { Invoke-Cached 'stats-native-failure' $build $positive | Out-Null } catch { $failed = $true }
    $global:ProbeFailStats = $false
    if (-not $failed -or @(Get-ChildItem (Join-Path $env:PROBE_SNAP 'stats-native-failure') -Filter '*.obj').Count -ne 2 -or
        @(Get-ChildItem (Join-Path $root 'probe/phase-records/stats-native-failure') -Filter '*.json').Count -ne 2) {
        throw 'stats 失败丢失快照/已完成对象记录或被放行'
    }
    # 一条真实子进程生成对象后失败、另一条没有对象；不能把部分实物判为通过。
    $partial = @(
        [pscustomobject]@{ obj = 'partial.cu.obj'; cwd = $build; cmd = ((Mock-CompileCommand 'partial.cu.obj' 3) -replace ' -EncodedCommand .*$', '') + ' -EncodedCommand ' +
            [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes("[IO.File]::WriteAllBytes('partial.cu.obj', [byte[]]::new(20)); exit 7")) },
        [pscustomobject]@{ obj = 'absent-partial.cu.obj'; cwd = $build; cmd = $negative[0].cmd }
    )
    $failed = $false
    try { Invoke-Vanilla 'partial' $build $partial | Out-Null } catch { $failed = $true }
    $failurePath = Get-ChildItem (Join-Path $env:PROBE_SNAP 'failures') -Directory -Filter 'partial-*' | Get-ChildItem -Filter 'failure.json' | Select-Object -Last 1
    $failureRecord = Get-Content $failurePath.FullName -Raw | ConvertFrom-Json
    if (-not $failed -or -not $failureRecord.partial -or $failureRecord.expected -ne 2 -or
        @($failureRecord.copied).Count -ne 1 -or @($failureRecord.missing).Count -ne 1 -or
        @($failureRecord.errors).Count -ne 0 -or -not (Test-Path (Join-Path $failurePath.DirectoryName 'partial.cu.obj'))) {
        throw '部分编译失败未完整留证或被放行'
    }
    # 真正终止正在等待的独立执行器进程树，再由另一步从活动标记恢复已生成对象。
    $stalled = [pscustomobject]@{ obj = 'interrupted.cu.obj'; cwd = $build; cmd = '"' + $pwsh + '" -NoProfile -EncodedCommand ' +
        [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes("[IO.File]::WriteAllBytes('interrupted.cu.obj', [byte[]]::new(20)); Start-Sleep -Seconds 120")) }
    $inner = 'Import-Module ' + "'" + (Join-Path $PSScriptRoot 'probe.psm1') + "' -Force; " +
        '$c = ConvertFrom-Json @' + "'`n" + ($stalled | ConvertTo-Json -Compress) + "`n'@; Invoke-Vanilla 'interrupted' '" + $build + "' @(`$c)"
    $psi = [Diagnostics.ProcessStartInfo]::new($pwsh, '-NoProfile -EncodedCommand ' + [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner)))
    $psi.UseShellExecute = $false
    $process = [Diagnostics.Process]::Start($psi)
    $wait = [Diagnostics.Stopwatch]::StartNew()
    try {
        while (-not (Test-Path (Join-Path $build 'interrupted.cu.obj')) -and $wait.Elapsed.TotalSeconds -lt 20 -and -not $process.WaitForExit(50)) {}
        if (-not (Test-Path (Join-Path $build 'interrupted.cu.obj'))) { throw '中断留证测试未真正启动生成对象的子进程' }
        $process.Kill($true); $process.WaitForExit()
    } finally {
        if (-not $process.HasExited) { $process.Kill($true); $process.WaitForExit() }
        $process.Dispose()
    }
    $active = Get-Content (Join-Path $root 'probe/active-phase.json') -Raw | ConvertFrom-Json
    if ($active.tag -ne 'interrupted' -or $active.status -ne 'running') { throw '超时活动标记丢失/错误' }
    Save-PhaseFailure $active.build @($active.objects) $active.tag '独立步骤模拟超时恢复'
    $recovered = Get-ChildItem (Join-Path $env:PROBE_SNAP 'failures') -Directory -Filter 'interrupted-*' | Get-ChildItem -Filter 'failure.json' | Select-Object -Last 1
    $record = Get-Content $recovered.FullName -Raw | ConvertFrom-Json
    if (-not $record.partial -or @($record.copied).Count -ne 1 -or @($record.missing).Count -ne 0 -or
        (Get-FileHash (Join-Path $recovered.DirectoryName 'interrupted.cu.obj') -Algorithm SHA256).Hash.ToLower() -cne $record.copied[0].sha256) {
        throw '独立超时恢复没有保存真实原物及 SHA'
    }
    foreach ($kind in 'empty', 'missing-hit', 'cuda-miss', 'cache-error', 'read-error', 'missing-field', 'missing-counters') {
        $badStats = $global:ProbeStatsFixture | ConvertFrom-Json
        switch ($kind) {
            'empty' { $badStats.stats.compile_requests = 0 }
            'missing-hit' { $badStats.stats.cache_hits.counts.PTX = 9 }
            'cuda-miss' { $badStats.stats.cache_misses.counts | Add-Member CUDA 1 }
            'cache-error' { $badStats.stats.cache_errors.counts | Add-Member CUDA 1 }
            'read-error' { $badStats.stats.cache_read_errors = 1 }
            'missing-field' { $badStats.stats.PSObject.Properties.Remove('compile_fails') }
            'missing-counters' { $badStats.stats.cache_errors.PSObject.Properties.Remove('counts') }
        }
        $badStats | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $root 'probe/stats-bad.json')
        $failed = $false
        try { Assert-WarmStats 'bad' 2 } catch { $failed = $true }
        if (-not $failed) { throw "错误统计 $kind 被放行" }
    }
    Write-Host 'PowerShell 全量命令、热回放字节绑定及失败门禁测试通过'
    $global:LASTEXITCODE = 0
} finally {
    foreach ($name in $original.Keys) { [Environment]::SetEnvironmentVariable($name, $original[$name]) }
}
