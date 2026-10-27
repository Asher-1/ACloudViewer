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

function Assert-SnapshotReplay([string]$A, [string]$B, [string[]]$Objs) {
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
