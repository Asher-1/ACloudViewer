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
    # ZIP 校验使用真实压缩文件；错误输入不得生成解包目录。
    function Write-ZipFixture([string]$Path, [string[]]$Names) {
        $archive = [IO.Compression.ZipFile]::Open($Path, [IO.Compression.ZipArchiveMode]::Create)
        try {
            foreach ($name in $Names) {
                $entry = $archive.CreateEntry($name, [IO.Compression.CompressionLevel]::NoCompression)
                $stream = $entry.Open()
                try { $stream.WriteByte(42) } finally { $stream.Dispose() }
            }
        } finally { $archive.Dispose() }
    }
    $zip = Join-Path $root 'valid.zip'
    Write-ZipFixture $zip @('input.json')
    $zipSize = (Get-Item $zip).Length; $zipSha = (Get-FileHash $zip).Hash.ToLower()
    $unpacked = Join-Path $root 'valid-unpacked'
    Expand-VerifiedProbeArchive $zip $zipSize $zipSha $unpacked
    if ([IO.File]::ReadAllBytes((Join-Path $unpacked 'input.json'))[0] -ne 42) { throw 'ZIP 正例未解包原字节' }
    foreach ($kind in 'size', 'sha', 'existing', 'crc', 'traversal', 'case-collision') {
        $path = $zip; $size = $zipSize; $digest = $zipSha; $destination = Join-Path $root "bad-zip-$kind"
        if ($kind -eq 'size') { $size++ }
        if ($kind -eq 'sha') { $digest = '0' * 64 }
        if ($kind -eq 'existing') { $destination = $unpacked }
        if ($kind -eq 'crc') {
            $path = Join-Path $root 'crc.zip'
            $data = [IO.File]::ReadAllBytes($zip)
            $offset = 30 + [BitConverter]::ToUInt16($data, 26) + [BitConverter]::ToUInt16($data, 28)
            $data[$offset] = $data[$offset] -bxor 1
            [IO.File]::WriteAllBytes($path, $data)
            $digest = (Get-FileHash $path).Hash.ToLower()
        }
        if ($kind -in @('traversal', 'case-collision')) {
            $path = Join-Path $root "$kind.zip"
            $names = if ($kind -eq 'traversal') { @('../outside') } else { @('same', 'SAME') }
            Write-ZipFixture $path $names
            $size = (Get-Item $path).Length; $digest = (Get-FileHash $path).Hash.ToLower()
        }
        $failed = $false
        try { Expand-VerifiedProbeArchive $path $size $digest $destination } catch { $failed = $true }
        if (-not $failed -or ($kind -ne 'existing' -and (Test-Path $destination))) { throw "ZIP 负例被放行/提前解包: $kind" }
    }
    # 独立补测重放使用完整 58 项 metadata fixture；不伪装成真实静态门禁。
    $env:PROBE_SNAP = Join-Path $root 'benefit-snap'
    Import-Module (Join-Path $PSScriptRoot 'probe.psm1') -Force
    $fixtureBytes = [byte[]]::new(64); $fixtureBytes[0] = 0x64; $fixtureBytes[1] = 0x86; $fixtureBytes[2] = 1
    [Array]::Copy([Text.Encoding]::ASCII.GetBytes('.fatbin'), 0, $fixtureBytes, 20, 7)
    $fixtureBytes[36] = 4; $fixtureBytes[40] = 60
    $fixtureSha = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($fixtureBytes)).ToLower()
    $source = Join-Path $root 'benefit-source'
    $gate = Join-Path $root 'benefit-gate'
    New-Item -ItemType Directory -Path (Join-Path $source 'strict'), $gate -Force | Out-Null
    $manifest = @(foreach ($i in 0..57) {
        $obj = if ($i -lt 3) { "librealsense\\$i.cu.obj" } else { "$i.cu.obj" }
        $cmd = "nvcc.exe -c $i.cu -o $obj"
        [ordered]@{ obj = $obj; cmd = $cmd; original = "sccache.exe $cmd"; launched = $true;
            cwd = if ($i -lt 3) { 'D:\\build\\ext_librealsense-build' } else { 'D:\\build' } }
    })
    function Write-BenefitFixture([string]$Name, $Value) {
        ConvertTo-Json -InputObject $Value -Depth 20 | Set-Content (Join-Path $source $Name)
    }
    Write-BenefitFixture 'commands.json' $manifest
    Write-BenefitFixture 'benefit.json' ([ordered]@{ schema = 'nvcc-benefit/1'; lane = 'wheel'; expected = 58;
        head = '47136622fac901e999ebded6df55875a23fbf482'; source_ref = 'cc5a64d2bc0c609165693641fcbd379d4792e76b';
        pool_run = '40358505262'; pool_name = 'probe-pool-wheel'; runner = 'fixture'; image = 'fixture';
        sequence = @('A','W1','B','W2'); manifest_sha = (Get-FileHash (Join-Path $source 'commands.json')).Hash.ToLower() })
    Write-BenefitFixture 'strict/gate.json' @{ equivalent = $false }
    $originalSha = (Get-FileHash (Join-Path $source 'strict/gate.json')).Hash
    $pairs = @{}
    foreach ($label in 'S1-vs-A', 'A-vs-B', 'S1-vs-B') {
        $rows = @{}
        foreach ($c in $manifest) { $rows[($c.obj -replace '[\\/:]', '__')] = @{ equivalent = $true; raw_sha_a = $fixtureSha; raw_sha_b = $fixtureSha } }
        $pairs[$label] = @{ equivalent = $true; fatbin = 58; errors = @{}; objects = $rows }
    }
    $candidate = @{ schema = 'nvcc-equivalence/5'; controls = 'fully-decoded-physical-lengths+proven-minimal-zero-load-padding';
        ptx_extraction = 'exact-or-windows-lf-expansion'; lane = 'wheel'; expected = 58; equivalent = $true;
        manifest_sha = (Get-FileHash (Join-Path $source 'commands.json')).Hash.ToLower(); pairs = $pairs }
    $candidate | ConvertTo-Json -Depth 20 | Set-Content (Join-Path $gate 'gate.json')
    $order = 0
    foreach ($tag in 'A', 'W1', 'B', 'W2') {
        $mode = if ($tag.StartsWith('W')) { 'cached' } else { 'vanilla' }
        $wall = if ($mode -eq 'cached') { 2 } else { 20 }
        Write-BenefitFixture "phase-$tag.json" @{ tag = $tag; mode = $mode; objects = 58; parallelism = 8; wall_sec = $wall;
            started_utc = "2026-10-28T18:0${order}:00.1234567Z"; ended_utc = ('2026-10-28T18:0{0}:{1:00}.1234567Z' -f $order,$wall) }
        Write-BenefitFixture "$mode-$tag.json" @(foreach ($c in $manifest) {
            @{ obj = $c.obj; cwd = $c.cwd; rc = 0; sec = 0.1; executed = if ($mode -eq 'cached') { $c.original } else { $c.cmd } }
        })
        if ($mode -eq 'cached') {
            $stats = $global:ProbeStatsFixture | ConvertFrom-Json
            $stats | Add-Member version '0.19.0'
            $stats | Add-Member max_cache_size 20GB
            $stats | Add-Member cache_size 1GB
            $stats.stats.compile_requests = 58
            $stats.stats.cache_hits.counts.CUDA = 58; $stats.stats.cache_hits.counts.'CUDA (Device code)' = 58
            $stats.stats.cache_hits.counts.PTX = 290; $stats.stats.cache_hits.counts.CUBIN = 290
            Write-BenefitFixture "stats-$tag.json" $stats
        }
        $order++
    }
    # 最小 COFF 仅用于摘要/字节回放，静态门禁由上面的明确 fixture 隔离。
    foreach ($tag in 'S1', 'A', 'B', 'W1', 'W2') {
        $directory = Join-Path $env:PROBE_SNAP $tag
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        foreach ($c in $manifest) {
            [IO.File]::WriteAllBytes((Join-Path $directory ($c.obj -replace '[\\/:]', '__')), $fixtureBytes)
        }
        # 前面测试留下的无关对象移至其它 fixture 目录，不能混进 58 项正例。
        foreach ($file in Get-ChildItem $directory -Filter '*.obj') {
            if ($file.Name -notin @($manifest.obj | ForEach-Object { $_ -replace '[\\/:]', '__' })) {
                Move-Item $file.FullName (Join-Path $root ($tag+'-'+$file.Name))
            }
        }
    }
    $recovery = Join-Path $env:PROBE_SNAP 'failures/W2-fixture'
    New-Item -ItemType Directory -Path $recovery -Force | Out-Null
    foreach ($c in $manifest) { Copy-Item (Join-Path (Join-Path $env:PROBE_SNAP 'W2') ($c.obj -replace '[\\/:]', '__')) $recovery }
    @{ schema = 'nvcc-phase-failure/1'; partial = $true; tag = 'W2'; expected = 58; missing = @(); errors = @();
        copied = @(foreach ($c in $manifest) { @{ obj = $c.obj; size = 64; sha256 = $fixtureSha } })
    } | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $recovery 'failure.json')
    Assert-BenefitReplay $source $gate
    $result = Get-Content (Join-Path $root 'probe/benefit-replay.json') -Raw | ConvertFrom-Json
    if ($result.original_equivalent -cne $false -or $result.speedup.'A/W1' -ne 10 -or $result.raw_objects -ne 290 -or
        (Get-FileHash (Join-Path $source 'strict/gate.json')).Hash -cne $originalSha -or
        @($result.recovery_evidence).Count -ne 1 -or $result.recovery_evidence[0].objects -ne 58 -or
        ($IsWindows -and $result.verification_kind -cne 'windows-original-artifact-replay') -or
        (-not $IsWindows -and $result.verification_kind -cne 'non-windows-original-artifact-replay')) { throw '独立重放覆盖了原失败或计时/平台归因错误' }
    foreach ($case in @(
        @{ path = 'benefit.json'; mutate = { param($x) $x.head = 'wrong' } },
        @{ path = 'benefit.json'; mutate = { param($x) $x.sequence = @('A','B','W1','W2') } },
        @{ path = 'benefit.json'; mutate = { param($x) $x.manifest_sha = 'wrong' } },
        @{ path = 'strict/gate.json'; mutate = { param($x) $x.equivalent = $true } },
        @{ path = 'phase-W1.json'; mutate = { param($x) $x.parallelism = 4 } },
        @{ path = 'phase-W1.json'; mutate = { param($x) $x.wall_sec = 0 } },
        @{ path = 'phase-W1.json'; mutate = { param($x) $x.started_utc = '2026-10-28T18:00:00Z' } },
        @{ path = 'phase-W1.json'; mutate = { param($x) $x.ended_utc = '2026-10-28T18:01:30Z' } },
        @{ path = 'phase-W1.json'; mutate = { param($x) $x.started_utc = '2026-10-28T18:01:00+08:00' } },
        @{ path = 'phase-W1.json'; mutate = { param($x) $x.started_utc = '2026-10-28T18:01:00' } },
        @{ path = 'phase-W1.json'; mutate = { param($x) $x.started_utc = $null } },
        @{ path = 'phase-W1.json'; mutate = { param($x) $x.started_utc = '2026-13-28T18:01:00Z' } },
        @{ path = 'cached-W1.json'; mutate = { param($x) $x[0].rc = 7 } },
        @{ path = 'cached-W1.json'; mutate = { param($x) $x[0].cwd = 'wrong' } },
        @{ path = 'cached-W1.json'; mutate = { param($x) $x[0].executed = 'wrong' } },
        @{ path = 'cached-W1.json'; mutate = { param($x) foreach ($p in $x) { $p.sec = 1 } } },
        @{ path = 'stats-W1.json'; mutate = { param($x) $x.stats.cache_hits.counts.PTX = 289 } },
        @{ path = 'stats-W1.json'; mutate = { param($x) $x.version = 'wrong' } },
        @{ path = 'stats-W1.json'; mutate = { param($x) $x.max_cache_size = 10GB } },
        @{ path = 'stats-W1.json'; mutate = { param($x) $x.cache_size = 0 } }
    )) {
        $path = Join-Path $source $case.path
        $saved = Get-Content $path -Raw
        $value = $saved | ConvertFrom-Json
        & $case.mutate $value
        Write-BenefitFixture $case.path $value
        $failed = $false
        try { Assert-BenefitReplay $source $gate } catch { $failed = $true }
        Set-Content $path $saved -NoNewline
        if (-not $failed) { throw "补测重放负例被放行: $($case.path)" }
    }
    $saved = Get-Content (Join-Path $gate 'gate.json') -Raw
    $bad = $saved | ConvertFrom-Json
    foreach ($kind in 'fatbin', 'raw-sha') {
        $bad = $saved | ConvertFrom-Json
        if ($kind -eq 'fatbin') { $bad.pairs.'A-vs-B'.fatbin = 56 }
        else { $bad.pairs.'A-vs-B'.objects.PSObject.Properties[$manifest[0].obj -replace '[\\/:]', '__'].Value.raw_sha_b = 'wrong' }
        $bad | ConvertTo-Json -Depth 20 | Set-Content (Join-Path $gate 'gate.json')
        $failed = $false
        try { Assert-BenefitReplay $source $gate } catch { $failed = $true }
        if (-not $failed) { throw "候选负例被放行: $kind" }
        Set-Content (Join-Path $gate 'gate.json') $saved -NoNewline
    }
    $copiedPath = Join-Path $recovery ($manifest[0].obj -replace '[\\/:]', '__')
    [IO.File]::WriteAllBytes($copiedPath, [byte[]]::new(64))
    $failed = $false
    try { Assert-BenefitReplay $source $gate } catch { $failed = $true }
    if (-not $failed) { throw '原失败恢复副本被篡改后仍放行' }
    [IO.File]::WriteAllBytes($copiedPath, $fixtureBytes)
    [IO.File]::WriteAllBytes((Join-Path (Join-Path $env:PROBE_SNAP 'W2') ($manifest[0].obj -replace '[\\/:]', '__')), [byte[]]::new(20))
    $failed = $false
    try { Assert-BenefitReplay $source $gate } catch { $failed = $true }
    if (-not $failed) { throw '补测热回放原对象被篡改后仍放行' }
    Write-Host 'PowerShell 全量命令、热回放字节绑定、补测来源和失败门禁测试通过'
    $global:LASTEXITCODE = 0
} finally {
    foreach ($name in $original.Keys) { [Environment]::SetEnvironmentVariable($name, $original[$name]) }
}
