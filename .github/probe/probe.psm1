# Probe-only helpers (branch ci/probe-nvcc-sccache, never merged).
# Imported by the probe workflow's measurement steps; every function writes
# its raw evidence under $env:RUNNER_TEMP\probe so the artifact is the record.

$script:Out = Join-Path $env:RUNNER_TEMP 'probe'
New-Item -ItemType Directory -Force -Path $script:Out | Out-Null
$script:Snap = if ($env:PROBE_SNAP) { $env:PROBE_SNAP } else { 'C:\probe-snap' }
$script:Coff = Join-Path $env:PROBE_TOOLS 'coff.py'

function Write-Probe([string]$Line) {
    Write-Host "[probe] $Line"
    Add-Content -Path (Join-Path $script:Out 'summary.md') -Value $Line
}

# Live-daemon stats; hit/miss counts are per sccache sub-stage (C/C++, CUDA,
# CUDA device code, PTX, CUBIN), not per object.
function Get-ScStats([string]$Tag) {
    $raw = & $env:SCCACHE_PATH --show-stats --stats-format=json
    if ($LASTEXITCODE -ne 0) { throw "live stats $Tag failed" }
    $raw | Out-File (Join-Path $script:Out "stats-$Tag.json") -Encoding utf8
    $s = ($raw | ConvertFrom-Json).stats
    $sum = { param($c) $t = 0L; if ($c) { foreach ($p in $c.counts.PSObject.Properties) { $t += [long]$p.Value } }; $t }
    [pscustomobject][ordered]@{
        tag = $Tag; requests = $s.compile_requests; non_cacheable = $s.non_cacheable_compilations
        hits = (& $sum $s.cache_hits); misses = (& $sum $s.cache_misses); errors = (& $sum $s.cache_errors)
        hits_by = ($s.cache_hits.counts | ConvertTo-Json -Compress)
        misses_by = ($s.cache_misses.counts | ConvertTo-Json -Compress)
    }
}

# Pool size, and the part the production trim would keep (mtime >= T0:
# entries written or hit by this job).
function Get-PoolSize([long]$T0 = 0) {
    $f = @(Get-ChildItem $env:SCCACHE_DIR -Recurse -File)
    $k = @($f | Where-Object { $_.LastWriteTimeUtc.Ticks -ge $T0 })
    "{0} files {1:N3} GiB; kept by trim {2} files {3:N3} GiB" -f $f.Count, (($f | Measure-Object Length -Sum).Sum / 1GB),
        $k.Count, (($k | Measure-Object Length -Sum).Sum / 1GB)
}

function Get-Ninja([string]$Build) {
    (Select-String -Path (Join-Path $Build 'CMakeCache.txt') -Pattern '^CMAKE_MAKE_PROGRAM:[A-Z]+=(.*)$').Matches[0].Groups[1].Value
}

# Every *.cu.obj on disk (ExternalProject sub-builds included) and the
# subset registered in the root Ninja graph (re-buildable by name).
function Get-CudaObjects([string]$Build) {
    $ninja = Get-Ninja $Build
    Push-Location $Build
    $all = @(Get-ChildItem . -Recurse -Filter '*.cu.obj' | ForEach-Object { $_.FullName.Substring($Build.Length + 1) })
    $root = @($all | Where-Object { & $ninja -f build-Release.ninja -t query $_ *> $null; $LASTEXITCODE -eq 0 })
    Pop-Location
    Write-Probe "CUDA objects: $($root.Count) in the root Ninja graph, $($all.Count) *.cu.obj on disk"
    @{ Root = $root; All = $all }
}

# Per-edge durations of every Ninja graph (root + sub-builds), for the
# offline nvcc share / critical-path analysis.
function Copy-NinjaLogs([string]$Build) {
    $dst = Join-Path $script:Out 'ninja_logs'
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Get-ChildItem $Build -Recurse -Force -Filter '.ninja_log' | ForEach-Object {
        $rel = $_.FullName.Substring($Build.Length + 1) -replace '[\\/:]', '__'
        Copy-Item $_.FullName (Join-Path $dst ("ninja" + $rel)) -Force
    }
}

# The smallest objects of each snapshot, byte-for-byte, for offline diffing.
function Save-RawSamples([string[]]$Snaps, [int]$Count = 3) {
    $pick = Get-ChildItem (Join-Path $script:Snap $Snaps[0]) -Filter '*.obj' | Sort-Object Length | Select-Object -First $Count
    foreach ($s in $Snaps) {
        $dst = Join-Path (Join-Path $script:Out 'raw') $s
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        foreach ($p in $pick) { Copy-Item (Join-Path (Join-Path $script:Snap $s) $p.Name) $dst -ErrorAction SilentlyContinue }
    }
}

function Save-Snapshot([string]$Build, [string[]]$Objs, [string]$Name) {
    $dst = Join-Path $script:Snap $Name
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    foreach ($o in $Objs) {
        $src = Join-Path $Build $o
        if (-not (Test-Path $src)) { throw "snapshot $Name missing object $o" }
        Copy-Item $src (Join-Path $dst ($o -replace '[\\/:]', '__')) -Force
    }
    if (@(Get-ChildItem $dst -Filter '*.obj').Count -ne $Objs.Count) { throw "snapshot $Name count/collision" }
    python $script:Coff digest (Join-Path $script:Out "digest-$Name.json") $dst
    if ($LASTEXITCODE -ne 0) { throw "snapshot digest $Name failed" }
    $dst
}

function Save-PhaseFailure([string]$Build, [string[]]$Objs, [string]$Tag, [string]$Reason) {
    # 正常异常和独立超时恢复共同使用；部分实物永远不作为通过快照。
    $dst = Join-Path (Join-Path $script:Snap 'failures') ($Tag + '-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    $copied = [Collections.Generic.List[object]]::new()
    $missing = [Collections.Generic.List[string]]::new()
    $errors = [Collections.Generic.List[object]]::new()
    foreach ($obj in $Objs) {
        $src = Join-Path $Build $obj
        if (-not (Test-Path $src -PathType Leaf)) { $missing.Add($obj); continue }
        try {
            $path = Join-Path $dst ($obj -replace '[\\/:]', '__')
            Copy-Item $src $path -Force -ErrorAction Stop
            $copied.Add([pscustomobject]@{ obj = $obj; size = (Get-Item $path).Length; sha256 = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower() })
        } catch { $errors.Add([pscustomobject]@{ obj = $obj; error = $_.Exception.Message }) }
    }
    [ordered]@{ schema = 'nvcc-phase-failure/1'; partial = $true; tag = $Tag; reason = $Reason;
        expected = @($Objs).Count; copied = $copied; missing = $missing; errors = $errors
    } | ConvertTo-Json -Depth 6 | Out-File (Join-Path $dst 'failure.json') -Encoding utf8
    Write-Probe "FAILURE EVIDENCE $Tag copied $($copied.Count)/$(@($Objs).Count), missing $($missing.Count), copy errors $($errors.Count); partial only"
}

function Compare-Snapshot([string]$A, [string]$B, [string]$DigestDirA = $script:Out) {
    $label = "$A-vs-$B"
    $r = python $script:Coff compare (Join-Path $DigestDirA "digest-$A.json") (Join-Path $script:Out "digest-$B.json") `
        (Join-Path $script:Snap $A) (Join-Path $script:Snap $B) $label (Join-Path $script:Out "cmp-$label.json")
    if ($LASTEXITCODE -ne 0) { throw "snapshot compare $label failed" }
    Write-Probe "bytes $r"
}

function Assert-SnapshotReplay([string]$A, [string]$B, [string[]]$Objs, [switch]$RequireFatbin) {
    # 全对象字节绑定；逐文件哈希，内存不随快照总大小增长。
    # 小型对象索引排序 O(n log n)，用于唯一性、覆盖核对和确定的差异顺序。
    Compare-Snapshot $A $B
    $expected = @($Objs | ForEach-Object { $_ -replace '[\\/:]', '__' } | Sort-Object)
    $report = Get-Content (Join-Path $script:Out "cmp-$A-vs-$B.json") -Raw | ConvertFrom-Json
    if ($expected.Count -eq 0 -or @($expected | Sort-Object -Unique).Count -ne $expected.Count -or
        @($report.missing.only_in_a).Count -ne 0 -or @($report.missing.only_in_b).Count -ne 0 -or
        @($report.objects.PSObject.Properties).Count -ne $expected.Count -or
        @($report.objects.PSObject.Properties | Where-Object { $_.Value -cne 'identical' }).Count -ne 0 -or
        @(Compare-Object $expected @($report.objects.PSObject.Properties.Name | Sort-Object)).Count -ne 0) {
        throw "snapshot replay $A/$B missing/different objects"
    }
    if ($RequireFatbin -and ($report.fatbin_objects -ne $expected.Count -or @($report.fatbin_differs).Count -ne 0)) {
        throw "snapshot replay $A/$B incomplete fatbin coverage"
    }
    foreach ($label in $A, $B) {
        $directory = Join-Path $script:Snap $label
        $digest = Get-Content (Join-Path $script:Out "digest-$label.json") -Raw | ConvertFrom-Json
        $actual = @(Get-ChildItem $directory -Filter '*.obj' -File | ForEach-Object Name | Sort-Object)
        if (@(Compare-Object $expected $actual).Count -ne 0 -or
            @(Compare-Object $expected @($digest.PSObject.Properties.Name | Sort-Object)).Count -ne 0) {
            throw "snapshot replay $label inventory mismatch"
        }
        foreach ($name in $expected) {
            $path = Join-Path $directory $name
            $record = $digest.PSObject.Properties[$name].Value
            if ((Get-Item $path).Length -ne $record.size -or
                (Get-FileHash $path -Algorithm SHA256).Hash.ToLower() -cne $record.full) {
                throw "snapshot replay $label raw digest mismatch: $name"
            }
        }
    }
    Write-Probe "REPLAY BYTES PASS $A/$B $($expected.Count)/$($expected.Count)"
}

function Assert-WarmStats([string]$Tag, [int]$Count) {
    $s = (Get-Content (Join-Path $script:Out "stats-$Tag.json") -Raw | ConvertFrom-Json).stats
    foreach ($key in 'cache_hits', 'cache_misses', 'cache_errors') {
        if ($null -eq $s.$key.counts) { throw "warm $Tag missing counters: $key" }
    }
    $want = @{ 'CUDA' = $Count; 'CUDA (Device code)' = $Count; 'PTX' = 5 * $Count; 'CUBIN' = 5 * $Count }
    if ($Count -le 0 -or $s.compile_requests -lt $Count) { throw "warm $Tag empty/incomplete daemon" }
    foreach ($key in $want.Keys) {
        if ($s.cache_hits.counts.$key -ne $want[$key] -or $s.cache_misses.counts.$key -gt 0) {
            throw "warm $Tag incomplete CUDA hits/misses: $key"
        }
    }
    foreach ($property in $s.cache_errors.counts.PSObject.Properties) {
        if ($property.Value -ne 0) { throw "warm $Tag cache error: $($property.Name)" }
    }
    foreach ($key in 'cache_timeouts', 'cache_read_errors', 'cache_write_errors', 'compile_fails', 'forced_recaches') {
        if ($s.$key -ne 0) { throw "warm $Tag error/recache: $key" }
    }
    Write-Probe "CUDA HITS PASS $Tag $Count objects; C/C++ counts remain separately reported"
}

function Expand-VerifiedProbeArchive([string]$Archive, [long]$Size, [string]$Sha256, [string]$Destination) {
    # 冻结 ZIP 长度和 SHA 绑定原件；CRC、路径校验完成后才解压，不覆盖已有证据。
    if ($Size -le 0 -or $Sha256 -cnotmatch '^[0-9a-f]{64}$' -or (Get-Item $Archive).Length -ne $Size -or
        (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLower() -cne $Sha256 -or (Test-Path $Destination)) {
        throw '原始 ZIP 大小/SHA/独立目标不符'
    }
    # ZIP 按成员流式校验/解压；数据缓冲约 1 MiB，索引仅保存成员路径。
    python -I -c @'
import stat, sys, zipfile
from pathlib import Path, PureWindowsPath
root = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(sys.argv[1]) as archive:
    members = archive.infolist()
    names = [str(PureWindowsPath(m.filename)).casefold() for m in members]
    if not members or len(names) != len(set(names)):
        raise ValueError('empty or duplicate/case-colliding ZIP members')
    for member in members:
        path = PureWindowsPath(member.filename)
        if (path.drive or path.root or '..' in path.parts or
                stat.S_ISLNK(member.external_attr >> 16) or
                not (root / member.filename).resolve().is_relative_to(root)):
            raise ValueError('unsafe ZIP member path')
    if archive.testzip() is not None:
        raise ValueError('ZIP CRC mismatch')
    root.mkdir(parents=True, exist_ok=False)
    archive.extractall(root)
'@ $Archive $Destination
    if ($LASTEXITCODE -ne 0) { throw '原始 ZIP CRC/路径/解包失败' }
}

function Assert-BenefitReplay([string]$Source, [string]$Gate) {
    # 输入是已校验 ZIP 的独立解包目录；原失败门禁只读，候选另存。
    $sourceRoot = [IO.Path]::GetFullPath($Source).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if ([IO.Path]::GetFullPath($Gate).StartsWith($sourceRoot, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFullPath($script:Out).StartsWith($sourceRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw '候选门禁/摘要不可写入原始证据目录'
    }
    $benefit = Get-Content (Join-Path $Source 'benefit.json') -Raw | ConvertFrom-Json
    $originalPath = Join-Path $Source 'strict/gate.json'
    $originalGate = Get-Content $originalPath -Raw | ConvertFrom-Json
    $originalSha = (Get-FileHash $originalPath -Algorithm SHA256).Hash.ToLower()
    $gatePath = Join-Path $Gate 'gate.json'
    $candidate = Get-Content $gatePath -Raw | ConvertFrom-Json
    $manifestPath = Join-Path $Source 'commands.json'
    $commands = @(Get-Content $manifestPath -Raw | ConvertFrom-Json)
    $manifestSha = (Get-FileHash $manifestPath -Algorithm SHA256).Hash.ToLower()
    if ($benefit.schema -cne 'nvcc-benefit/1' -or $benefit.head -cne '47136622fac901e999ebded6df55875a23fbf482' -or
        $benefit.source_ref -cne 'cc5a64d2bc0c609165693641fcbd379d4792e76b' -or $benefit.pool_run -cne '40358505262' -or
        $benefit.pool_name -cne 'probe-pool-wheel' -or $benefit.lane -cne 'wheel' -or $benefit.expected -ne 58 -or
        ($benefit.sequence -join ',') -cne 'A,W1,B,W2' -or $benefit.manifest_sha -cne $manifestSha -or
        -not $benefit.runner -or -not $benefit.image -or $originalGate.equivalent -cne $false) {
        throw '本轮补测来源或原失败门禁身份不符'
    }
    $expected = @($commands.obj | Sort-Object)
    $children = @($commands | Where-Object { $_.obj.StartsWith('librealsense') })
    if ($commands.Count -ne 58 -or @($expected | Sort-Object -Unique).Count -ne 58 -or $children.Count -ne 3 -or
        @($children | Where-Object { -not $_.cwd.EndsWith('ext_librealsense-build') }).Count -ne 0 -or
        @($commands | Where-Object { $_.launched -cne $true -or -not $_.cmd.Contains('nvcc.exe') -or
            -not $_.original.EndsWith($_.cmd) -or -not $_.original.Contains('sccache') }).Count -ne 0) {
        throw '本轮 58 命令或 librealsense 三个子图覆盖不符'
    }
    if ($candidate.schema -cne 'nvcc-equivalence/5' -or
        $candidate.controls -cne 'fully-decoded-physical-lengths+proven-minimal-zero-load-padding' -or
        $candidate.ptx_extraction -cne 'exact-or-windows-lf-expansion' -or $candidate.lane -cne 'wheel' -or
        $candidate.expected -ne 58 -or $candidate.manifest_sha -cne $manifestSha -or
        $candidate.equivalent -cne $true -or $candidate.fatal -or
        (($candidate.pairs.PSObject.Properties.Name | Sort-Object) -join ',') -cne 'A-vs-B,S1-vs-A,S1-vs-B') {
        throw '本轮候选全量门禁未成功或身份不符'
    }
    $flat = @($expected | ForEach-Object { $_ -replace '[\\/:]', '__' } | Sort-Object)
    foreach ($pair in $candidate.pairs.PSObject.Properties.Value) {
        if ($pair.equivalent -cne $true -or $pair.fatbin -ne 58 -or @($pair.errors.PSObject.Properties).Count -ne 0 -or
            @(Compare-Object -CaseSensitive $flat @($pair.objects.PSObject.Properties.Name | Sort-Object)).Count -ne 0 -or
            @($pair.objects.PSObject.Properties.Value | Where-Object { $_.equivalent -cne $true -or $_.error }).Count -ne 0) {
            throw '候选三组对照存在失败或缺项'
        }
    }
    $byObject = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    foreach ($command in $commands) { $byObject.Add($command.obj, $command) }
    $phases = @{}; $parallelism = $null; $previousEnd = $null
    foreach ($tag in 'A', 'W1', 'B', 'W2') {
        $mode = if ($tag.StartsWith('W')) { 'cached' } else { 'vanilla' }
        $phaseRaw = Get-Content (Join-Path $Source "phase-$tag.json") -Raw
        $phase = $phaseRaw | ConvertFrom-Json
        $records = @(Get-Content (Join-Path $Source "$mode-$tag.json") -Raw | ConvertFrom-Json)
        $wall = [double]$phase.wall_sec
        if ($phase.tag -cne $tag -or $phase.mode -cne $mode -or $phase.objects -ne 58 -or $records.Count -ne 58 -or
            ($phase.parallelism -isnot [long] -and $phase.parallelism -isnot [int]) -or $phase.parallelism -le 0 -or
            [double]::IsNaN($wall) -or [double]::IsInfinity($wall) -or $wall -le 0 -or
            @(Compare-Object -CaseSensitive $expected @($records.obj | Sort-Object)).Count -ne 0) {
            throw "补测 phase $tag 数量/计时/模式不符"
        }
        if ($null -eq $parallelism) { $parallelism = $phase.parallelism }
        if ($phase.parallelism -ne $parallelism) { throw '补测 phase 并行度不同' }
        # ConvertFrom-Json 自动生成 DateTime；再转字符串会丢失 Kind/时区。
        # 从原 JSON 取时间字符串，不能把非 UTC 输入悄悄转换成 UTC。
        $document = [System.Text.Json.JsonDocument]::Parse($phaseRaw)
        try {
            $startText = $document.RootElement.GetProperty('started_utc').GetString()
            $endText = $document.RootElement.GetProperty('ended_utc').GetString()
        } finally { $document.Dispose() }
        $utcPattern = '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,7})?(?:Z|\+00:00)$'
        if ($startText -cnotmatch $utcPattern -or $endText -cnotmatch $utcPattern) { throw "补测 phase $tag 原时间不是显式 UTC" }
        $started = [DateTimeOffset]::Parse($startText, [Globalization.CultureInfo]::InvariantCulture)
        $ended = [DateTimeOffset]::Parse($endText, [Globalization.CultureInfo]::InvariantCulture)
        if ($started.Offset -ne [TimeSpan]::Zero -or $ended.Offset -ne [TimeSpan]::Zero -or $ended -le $started -or
            ($null -ne $previousEnd -and $started -lt $previousEnd) -or
            [math]::Abs(($ended-$started).TotalSeconds-$wall) -gt [math]::Max(0.05, $wall*0.0001)) {
            throw "补测 phase $tag UTC/Stopwatch 或顺序不符"
        }
        $previousEnd = $ended; $sum = 0.0
        foreach ($record in $records) {
            if (-not $byObject.ContainsKey($record.obj)) { throw "补测 phase $tag 对象来源未知" }
            $command = $byObject[$record.obj]
            $line = if ($mode -eq 'cached') { $command.original } else { $command.cmd }
            $seconds = [double]$record.sec
            if ($record.rc -ne 0 -or $record.cwd -cne $command.cwd -or $record.executed -cne $line -or
                [double]::IsNaN($seconds) -or [double]::IsInfinity($seconds) -or $seconds -le 0 -or $seconds -gt $wall) {
                throw "补测 phase $tag 实际命令/cwd/退出码/计时不符"
            }
            $sum += $seconds
        }
        if ($sum -gt [math]::Min(58, $parallelism)*$wall+58e-6) { throw "补测 phase $tag 逐对象耗时超出并行上界" }
        if ($mode -eq 'cached') {
            $sample = Get-Content (Join-Path $Source "stats-$tag.json") -Raw | ConvertFrom-Json
            if ($sample.version -cne '0.19.0' -or $sample.max_cache_size -ne 20GB -or
                ($sample.cache_size -isnot [long] -and $sample.cache_size -isnot [int]) -or
                $sample.cache_size -le 0 -or $sample.cache_size -gt $sample.max_cache_size) {
                throw "补测热轮 $tag sccache 版本/容量不符"
            }
            Copy-Item (Join-Path $Source "stats-$tag.json") (Join-Path $script:Out "stats-$tag.json")
            Assert-WarmStats $tag 58
        }
        $phases[$tag] = $phase
    }
    foreach ($tag in 'S1', 'A', 'B', 'W1', 'W2') {
        python $script:Coff digest (Join-Path $script:Out "digest-$tag.json") (Join-Path $script:Snap $tag)
        if ($LASTEXITCODE -ne 0) { throw "独立派生摘要 $tag 失败" }
    }
    foreach ($pair in @(@('S1','W1'), @('S1','W2'), @('W1','W2'))) { Assert-SnapshotReplay $pair[0] $pair[1] $expected -RequireFatbin }
    $digests = @{}
    foreach ($tag in 'S1', 'A', 'B', 'W1', 'W2') {
        $digests[$tag] = Get-Content (Join-Path $script:Out "digest-$tag.json") -Raw | ConvertFrom-Json
        $actual = @(Get-ChildItem (Join-Path $script:Snap $tag) -File -Filter '*.obj' | ForEach-Object Name)
        if (@(Compare-Object -CaseSensitive $flat $actual).Count -ne 0) { throw "快照 $tag 对象覆盖不符" }
    }
    foreach ($label in 'S1-vs-A', 'A-vs-B', 'S1-vs-B') {
        $left, $right = $label -split '-vs-'
        foreach ($name in $flat) {
            $row = $candidate.pairs.$label.objects.PSObject.Properties[$name].Value
            if ($row.raw_sha_a -cne $digests[$left].PSObject.Properties[$name].Value.full -or
                $row.raw_sha_b -cne $digests[$right].PSObject.Properties[$name].Value.full) {
                throw "候选门禁与原快照 $label/$name 未绑定"
            }
        }
    }
    $allowed = @('S1', 'A', 'B', 'W1', 'W2', 'failures')
    if (@(Get-ChildItem $script:Snap | Where-Object { $_.Name -notin $allowed -or -not $_.PSIsContainer }).Count -ne 0) {
        throw '原快照根目录含未知项'
    }
    $recovery = @()
    $failureRoot = Join-Path $script:Snap 'failures'
    if (Test-Path $failureRoot) {
        $directories = @(Get-ChildItem $failureRoot)
        if ($directories.Count -eq 0) { throw '原失败留证目录为空' }
        foreach ($directory in $directories) {
            if (-not $directory.PSIsContainer) { throw '原失败留证含未知文件' }
            $recordPath = Join-Path $directory.FullName 'failure.json'
            $record = Get-Content $recordPath -Raw | ConvertFrom-Json
            if ($record.schema -cne 'nvcc-phase-failure/1' -or $record.partial -cne $true -or $record.expected -ne 58 -or
                $record.tag -cnotin @('S1','A','B','W1','W2') -or @($record.copied).Count -ne 58 -or
                @($record.missing).Count -ne 0 -or @($record.errors).Count -ne 0 -or
                @(Compare-Object -CaseSensitive $expected @($record.copied.obj)).Count -ne 0 -or
                @(Compare-Object -CaseSensitive @($flat + 'failure.json') @(Get-ChildItem $directory.FullName | ForEach-Object Name)).Count -ne 0) {
                throw '原失败恢复实物不完整或身份不符'
            }
            foreach ($item in $record.copied) {
                $name = $item.obj -replace '[\\/:]', '__'
                $path = Join-Path $directory.FullName $name
                $identity = $digests[$record.tag].PSObject.Properties[$name].Value
                if ((Get-Item $path).Length -ne $item.size -or $item.size -ne $identity.size -or
                    (Get-FileHash $path -Algorithm SHA256).Hash.ToLower() -cne $item.sha256 -or $item.sha256 -cne $identity.full) {
                    throw '原失败恢复对象与完整快照/记录不符'
                }
            }
            $recovery += @{ tag = $record.tag; objects = 58; record_sha = (Get-FileHash $recordPath).Hash.ToLower() }
        }
    }
    if ((Get-FileHash $originalPath -Algorithm SHA256).Hash.ToLower() -cne $originalSha) { throw '原失败门禁被修改' }
    $ratios = @{ 'A/W1' = $phases.A.wall_sec/$phases.W1.wall_sec; 'B/W2' = $phases.B.wall_sec/$phases.W2.wall_sec }
    if ($ratios['A/W1'] -le 1 -or $ratios['B/W2'] -le 1) { throw '两组同机收益未均为正' }
    [ordered]@{ schema = 'nvcc-benefit-replay/1'; source_run = 40503002099;
        verification_kind = if ($IsWindows) { 'windows-original-artifact-replay' } else { 'non-windows-original-artifact-replay' };
        runtime_os = [Runtime.InteropServices.RuntimeInformation]::OSDescription; recovery_evidence = $recovery;
        source_head = $benefit.head; manifest_sha = $manifestSha; original_equivalent = $originalGate.equivalent;
        original_gate_sha = $originalSha;
        candidate_gate_sha = (Get-FileHash $gatePath -Algorithm SHA256).Hash.ToLower();
        raw_objects = 290; speedup = $ratios; equivalent = $true;
        scope = '静态全量原件重放与原同机计时核验；不是 GPU 算法功能或新的编译测量'
    } | ConvertTo-Json -Depth 6 | Out-File (Join-Path $script:Out 'benefit-replay.json') -Encoding utf8
}

function Remove-Objects([string]$Build, [string[]]$Objs) {
    foreach ($o in $Objs) { Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Build $o) }
}

# nvcc without the launcher: the exact Ninja command lines with sccache
# stripped. Lines exceed cmd.exe's 8191 chars, so start nvcc directly.
function Get-VanillaCommands([string]$Build, [string[]]$Objs) {
    $cmds = @(foreach ($o in $Objs) {
        $found = $null
        $directory = [IO.DirectoryInfo](Split-Path (Join-Path $Build $o))
        while ($directory -and $directory.FullName.StartsWith($Build, [StringComparison]::OrdinalIgnoreCase)) {
            $cwd = $directory.FullName
            if (Test-Path (Join-Path $cwd 'CMakeCache.txt')) {
                $ninja = Get-Ninja $cwd
                $target = [IO.Path]::GetRelativePath($cwd, (Join-Path $Build $o)) -replace '\\', '/'
                foreach ($graph in 'build-Release.ninja', 'build.ninja') {
                    if (-not (Test-Path (Join-Path $cwd $graph))) { continue }
                    Push-Location $cwd
                    try {
                        & $ninja -f $graph -t query $target *> $null
                        if ($LASTEXITCODE -ne 0) { continue }
                        $lines = @(& $ninja -f $graph -t commands -s $target)
                        if ($LASTEXITCODE -ne 0 -or $lines.Count -eq 0) { throw "no command for $o" }
                        $c = $lines[-1]
                        $v = $c -replace '^"?[^"\s]*sccache(\.exe)?"?\s+', ''
                        if ($v -eq $c) { throw "launcher missing on $o" }
                        if ($v -notmatch '^"?[^"\r\n]*nvcc\.exe"?\s+' -or $v.Contains(' && ')) { throw "not a direct nvcc command: $o" }
                        $found = [pscustomobject]@{ obj = $o; cmd = $v; original = $c; cwd = $cwd; graph = $graph; target = $target; launched = $true }
                        break
                    } finally { Pop-Location }
                }
            }
            if ($found) { break }
            $directory = $directory.Parent
        }
        if (-not $found) { throw "no owning Ninja graph for $o" }
        $found
    })
    $cmds | ConvertTo-Json -Depth 5 | Out-File (Join-Path $script:Out 'commands.json') -Encoding utf8
    Write-Probe "launcher present on $($cmds.Count)/$($Objs.Count) CUDA command lines (all owning graphs)"
    $cmds
}

function Invoke-CompilePhase([string]$Tag, [string]$Build, $Cmds, [switch]$Cached) {
    $mode = if ($Cached) { 'cached' } else { 'vanilla' }
    # O(n log n) 索引排序只用于重复/覆盖核对，不包含在编译 phase 计时中。
    if (@($Cmds).Count -eq 0 -or @($Cmds.obj | Sort-Object -Unique).Count -ne @($Cmds).Count) {
        throw "$mode $Tag empty/duplicate commands"
    }
    foreach ($command in $Cmds) {
        $line = if ($Cached) { $command.original } else { $command.cmd }
        if ([string]::IsNullOrWhiteSpace($line) -or -not (Test-Path $command.cwd -PathType Container) -or
            ($Cached -and -not $command.launched)) { throw "$mode $Tag invalid command/cwd" }
    }
    $active = [ordered]@{ tag = $Tag; mode = $mode; build = $Build; objects = @($Cmds.obj); status = 'running' }
    $active | ConvertTo-Json -Depth 5 | Out-File (Join-Path $script:Out 'active-phase.json') -Encoding utf8
    try {
        Remove-Objects $Build $Cmds.obj
        if ($Cached) {
            & $env:SCCACHE_PATH --zero-stats | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "warm $Tag reset stats failed" }
        }
        $records = Join-Path (Join-Path $script:Out 'phase-records') $Tag
        New-Item -ItemType Directory -Force -Path $records | Out-Null
        $np = [Environment]::ProcessorCount
        # 六位小数保留微秒并兼容审计端 Python 3.10 的 ISO 时间解析。
        $started = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ss.ffffffZ')
        $t = [Diagnostics.Stopwatch]::StartNew()
        $per = @($Cmds | ForEach-Object -ThrottleLimit $np -Parallel {
            $c = if ($using:Cached) { $_.original } else { $_.cmd }
            $rc = -1; $errorText = $null
            $w = [Diagnostics.Stopwatch]::StartNew()
            try {
                if ($c -match '^"([^"]+)"\s+(.*)$' -or $c -match '^(\S+)\s+(.*)$') { $exe = $Matches[1]; $a = $Matches[2] }
                else { throw "invalid direct command: $($_.obj)" }
                $psi = [Diagnostics.ProcessStartInfo]::new($exe, $a)
                $psi.UseShellExecute = $false
                $psi.WorkingDirectory = $_.cwd
                $p = [Diagnostics.Process]::Start($psi); $p.WaitForExit(); $rc = $p.ExitCode
                $p.Dispose()
            } catch { $errorText = $_.Exception.Message }
            $w.Stop()
            $row = [pscustomobject]@{ obj = $_.obj; sec = $w.Elapsed.TotalSeconds; rc = $rc; cwd = $_.cwd; executed = $c; error = $errorText }
            # 每个已完成对象立即写独立记录；中断时不用依赖主 pipeline 缓冲刷新。
            $path = Join-Path $using:records (($_.obj -replace '[\\/:]', '__') + '.json')
            $row | ConvertTo-Json -Depth 5 | Out-File $path -Encoding utf8
            $row
        })
        $t.Stop()
        $ended = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ss.ffffffZ')
        ConvertTo-Json -InputObject $per -Depth 5 | Out-File (Join-Path $script:Out "$mode-$Tag.json") -Encoding utf8
        $phase = [pscustomobject]@{ tag = $Tag; mode = $mode; objects = $per.Count; wall_sec = $t.Elapsed.TotalSeconds; parallelism = $np; started_utc = $started; ended_utc = $ended }
        $phase | ConvertTo-Json | Out-File (Join-Path $script:Out "phase-$Tag.json") -Encoding utf8
        Write-Probe ("| {0} | {1} | {2:N1} | {3} | process-wall-sum {4:N1} s; mode {5}; parallelism {6} |" -f $Tag, $per.Count, $phase.wall_sec,
            @($per | Where-Object rc -ne 0).Count, ($per | Measure-Object sec -Sum).Sum, $mode, $np)
        if ($per.Count -ne @($Cmds).Count -or @($per | Where-Object { $_.rc -ne 0 -or $_.sec -le 0 }).Count -ne 0 -or
            @(Compare-Object @($Cmds.obj | Sort-Object) @($per.obj | Sort-Object)).Count -ne 0) { throw "$mode $Tag failed/incomplete" }
        foreach ($command in $Cmds) { if (-not (Test-Path (Join-Path $Build $command.obj))) { throw "$mode $Tag missing $($command.obj)" } }
        # 在任何 stats 查询/判定前保存成功生成的原物，计时已结束。
        Save-Snapshot $Build $Cmds.obj $Tag | Out-Null
        if ($Cached) {
            $stats = Get-ScStats $Tag
            Write-Probe "warm stats: $($stats | ConvertTo-Json -Compress)"
        }
        $active.status = 'complete'
        $active | ConvertTo-Json -Depth 5 | Out-File (Join-Path $script:Out 'active-phase.json') -Encoding utf8
        $per
    } catch {
        $failure = $_
        try { Save-PhaseFailure $Build $Cmds.obj $Tag $failure.Exception.Message } catch { Write-Warning "failure evidence $Tag could not be saved: $($_.Exception.Message)" }
        throw $failure
    }
}

function Invoke-Vanilla([string]$Tag, [string]$Build, $Cmds) {
    Invoke-CompilePhase $Tag $Build $Cmds
}

function Invoke-Cached([string]$Tag, [string]$Build, $Cmds) {
    Invoke-CompilePhase $Tag $Build $Cmds -Cached
}

function Invoke-NinjaPhase([string]$Tag, [string]$Build, [string[]]$Objs) {
    $ninja = Get-Ninja $Build
    Remove-Objects $Build $Objs
    & $env:SCCACHE_PATH --zero-stats | Out-Null
    Push-Location $Build
    $t = [Diagnostics.Stopwatch]::StartNew()
    & $ninja -f build-Release.ninja -j ([Environment]::ProcessorCount) @Objs | Out-File (Join-Path $script:Out "ninja-$Tag.log") -Encoding utf8
    $rc = $LASTEXITCODE
    Pop-Location
    $s = Get-ScStats $Tag
    Write-Probe ("| {0} | {1} | {2:N1} | {3} | requests {4}, hits {5}, misses {6}, errors {7}; hits {8}; misses {9} |" -f $Tag, @($Objs).Count,
        $t.Elapsed.TotalSeconds, $rc, $s.requests, $s.hits, $s.misses, $s.errors, $s.hits_by, $s.misses_by)
}

# cuobjdump is not in the CI sub-package list; fetch the matching redist
# archive AFTER all compiles so the toolkit used for the builds is untouched.
function Install-Cuobjdump {
    $c = Get-Command cuobjdump -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $dst = 'C:\probe-tools'
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    $base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
    $j = Invoke-RestMethod "${base}redistrib_$($env:CUDA_VERSION).json"
    foreach ($pkg in 'cuda_cuobjdump', 'cuda_nvdisasm') {
        $row = $j.$pkg.'windows-x86_64'
        $rel = $row.relative_path
        Invoke-WebRequest "$base$rel" -OutFile "$dst\$pkg.zip"
        if ((Get-FileHash "$dst\$pkg.zip" -Algorithm SHA256).Hash.ToLower() -ne $row.sha256) { throw "CUDA redist digest mismatch: $pkg" }
        Expand-Archive "$dst\$pkg.zip" $dst -Force
    }
    $env:PATH = ((Get-ChildItem $dst -Recurse -Directory -Filter bin).FullName -join ';') + ";$env:PATH"
    (Get-ChildItem $dst -Recurse -Filter cuobjdump.exe | Select-Object -First 1).FullName
}

# Disassemble only where the fatbin bytes differ (Compare-Snapshot A B must
# have run): identical fatbin bytes already prove identical device code, and
# a full-arch SASS listing of every object would not fit the job.
function Compare-DeviceCode([string]$Tool, [string]$A, [string]$B, [int]$Limit = 6) {
    $label = "$A-vs-$B"
    $cmp = Join-Path $script:Out "cmp-$label.json"
    $j = Get-Content $cmp -Raw | ConvertFrom-Json
    if ($j.fatbin_objects -gt 0 -and @($j.fatbin_differs).Count -eq 0) {
        Write-Probe "device code [$label]: fatbin bytes identical in all $($j.fatbin_objects) objects, no disassembly needed"
        return
    }
    foreach ($n in $A, $B) {
        $d = Join-Path $script:Snap $n
        Get-ChildItem $d -Filter '*.txt' -ErrorAction SilentlyContinue | Remove-Item -Force
        python $script:Coff dump $Tool $d $cmp $Limit
    }
    $r = python $script:Coff textcmp (Join-Path $script:Snap $A) (Join-Path $script:Snap $B) $label (Join-Path $script:Out "sass-$label.json")
    Write-Probe "device code $r"
}

Export-ModuleMember -Function *
