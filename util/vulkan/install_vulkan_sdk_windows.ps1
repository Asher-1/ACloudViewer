[CmdletBinding()]
param(
    [string]$Version = $(if ($env:VULKAN_SDK_VERSION) { $env:VULKAN_SDK_VERSION } else { "1.4.350.0" }),
    [string]$InstallRoot,
    [switch]$Force,
    [switch]$SkipProfile
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$syncScript = Join-Path $scriptDir "sync_vulkan_env_from_sdk.ps1"

function Test-VulkanSdkTree {
    param([string]$Root)

    if (-not $Root) { return $false }
    foreach ($sdkRoot in @($Root, (Join-Path $Root "x86_64"))) {
        foreach ($includeDir in @("Include", "include")) {
            $header = Join-Path $sdkRoot "$includeDir\vulkan\vulkan_core.h"
            if (-not (Test-Path $header)) { continue }
            foreach ($binDir in @("Bin", "bin")) {
                if (Test-Path (Join-Path $sdkRoot "$binDir\glslc.exe")) {
                    return $true
                }
            }
        }
    }
    return $false
}

if ($env:VULKAN_SDK -and (Test-VulkanSdkTree $env:VULKAN_SDK)) {
    Write-Host "Using preinstalled Vulkan SDK at $($env:VULKAN_SDK)"
    if ($SkipProfile) {
        & $syncScript -SkipProfile
    } else {
        & $syncScript
    }
    return
}

if (-not $InstallRoot) {
    $InstallRoot = Join-Path ${env:ProgramData} "ACloudViewer\VulkanSDK\$Version"
}

if ((Test-Path $InstallRoot) -and -not (Test-VulkanSdkTree $InstallRoot)) {
    Write-Host "Removing incomplete Vulkan SDK tree at $InstallRoot (common when SDK >= 1.4.313 was 7z-extracted)."
    Remove-Item $InstallRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
}

if ((Test-VulkanSdkTree $InstallRoot) -and -not $Force) {
    $env:VULKAN_SDK = $InstallRoot
    Write-Host "Reusing existing Vulkan SDK at $InstallRoot"
    if ($SkipProfile) {
        & $syncScript -SkipProfile
    } else {
        & $syncScript
    }
    return
}

$defaultSha256 = "855b27ba05d2d8119c5114c5d4ff870ca38f2c632b11e1bb9923b9b7e6ecfe7b"
$sha256 = if ($Version -eq "1.4.350.0") {
    $defaultSha256
} elseif ($env:VULKAN_SDK_SHA256) {
    $env:VULKAN_SDK_SHA256
} else {
    throw "Set VULKAN_SDK_SHA256 when overriding VULKAN_SDK_VERSION."
}

$workDir = Join-Path ([System.IO.Path]::GetTempPath()) "acloudviewer-vulkan-$Version"
$installer = Join-Path $workDir "vulkan_sdk.exe"
Remove-Item $workDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

# The LunarG CDN intermittently resets large-file transfers (seen in CI as
# "An existing connection was forcibly closed by the remote host"). A single
# Invoke-WebRequest has no resilience to that transient I/O fault, so retry
# with backoff. The SHA256 gate below makes partial/aborted retries safe: any
# corrupt re-fetch fails the hash check and never reaches the installer.
# On Windows PowerShell 5.1 progress rendering and response buffering stall
# large downloads, so silence progress and use basic parsing.
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$uri = "https://sdk.lunarg.com/sdk/download/$Version/windows/vulkan_sdk.exe"
Write-Host "Downloading Vulkan SDK $Version from $uri"
$maxAttempts = 5
for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    try {
        Remove-Item $installer -Force -ErrorAction SilentlyContinue
        Invoke-WebRequest -Uri $uri -OutFile $installer -UseBasicParsing -TimeoutSec 300
        break
    } catch {
        if ($attempt -eq $maxAttempts) {
            throw "Failed to download Vulkan SDK after $maxAttempts attempts: $($_.Exception.Message)"
        }
        $backoff = $attempt * 10
        Write-Host "Download attempt $attempt failed ($($_.Exception.Message)); retrying in ${backoff}s."
        Start-Sleep -Seconds $backoff
    }
}
$actualSha256 = (Get-FileHash -Algorithm SHA256 $installer).Hash.ToLowerInvariant()
if ($actualSha256 -ne $sha256.ToLowerInvariant()) {
    throw "Vulkan SDK SHA256 mismatch: got $actualSha256, expected $sha256"
}

$arguments = @(
    "--root", $InstallRoot,
    "--accept-licenses",
    "--default-answer",
    "--confirm-command",
    "install",
    "copy_only=1"
)
$process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru -NoNewWindow
if ($process.ExitCode -ne 0) {
    throw "Vulkan SDK installer failed with exit code $($process.ExitCode)"
}
# The installer process may still hold handles on its own files (observed:
# Temp\acloudviewer-vulkan-*\vulkan_sdk.exe) for a short while after
# reporting success, so a cleanup Remove-Item can hit a sharing violation
# (run 39366721072: the SDK was installed, only this cleanup failed, and
# the job failure cancelled the other three matrix jobs). The work dir is
# transient — the runner reclaims it — so best-effort cleanup, never a job
# failure.
try {
    Remove-Item $workDir -Recurse -Force -ErrorAction Stop
} catch {
    Write-Host "Warning: could not remove installer work dir '$workDir': $($_.Exception.Message)"
}

$env:VULKAN_SDK = $InstallRoot
if ($SkipProfile -or $env:GITHUB_ACTIONS) {
    & $syncScript -SkipProfile
} else {
    & $syncScript
}
