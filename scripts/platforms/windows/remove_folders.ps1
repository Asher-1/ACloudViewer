param (
    [Parameter(Mandatory=$false)]
    [string]$FolderPath,

    [Parameter(Mandatory=$false)]
    [switch]$y
)

# If no parameter is provided, prompt for input
if (-not $FolderPath) {
    Write-Host "$FolderPath has been already deleted."
    return $true
}

# Validate path
if (-not (Test-Path $FolderPath)) {
    Write-Warning "Specified path does not exist: $FolderPath"
    return $true
}

Write-Host "Target folder to delete: $FolderPath"
if (-not $y) {
    $confirmation = Read-Host "Confirm deletion? (Y/N)"
    if ($confirmation -ne 'Y') {
        Write-Host "Operation cancelled"
        return $false
    }
}

# Define processes to terminate
$processesToKill = @(
    # VS2019 related processes
    "devenv", "MSBuild", "VBCSCompiler", "ServiceHub", 
    "PerfWatson2", "CodeHelper", "vshost", "VsDebugConsole",
    "vcpkgsrv", "VSIXInstaller", "VSIXConfigurationUpdater",
    "Microsoft.ServiceHub*", "ServiceHub*",
    # Other potential processes
    "cmake", "ninja", "cl", "link", "rc", "git"
)

# Terminate processes
Write-Host "Terminating related processes..."
foreach ($process in $processesToKill) {
    Get-Process -Name $process -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "Terminating process: $($_.ProcessName) (PID: $($_.Id))"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}

# Check for file locks using handle.exe (if available)
$handlePath = "handle.exe"
if (Get-Command $handlePath -ErrorAction SilentlyContinue) {
    Write-Host "Checking for file locks..."
    $handleOutput = & $handlePath -a -u $FolderPath
    if ($handleOutput) {
        Write-Host "File locks detected, attempting to release..."
        # Add code here to process handle output if needed
    }
}

# Clean temporary files
Write-Host "Cleaning temporary files..."
Remove-Item -Path $env:TEMP\* -Force -Recurse -ErrorAction SilentlyContinue

# Attempt to delete the folder
Write-Host "Attempting to delete folder..."
try {
    # First, try to remove read-only attributes
    Get-ChildItem -Path $FolderPath -Recurse -Force | 
        ForEach-Object { $_.Attributes = 'Normal' }
    
    # Try using Remove-Item
    Remove-Item -Path $FolderPath -Force -Recurse -ErrorAction Stop
    Write-Host "Folder successfully deleted."
    return $true
}
catch {
    Write-Host "Standard deletion failed, trying alternative methods..."
    
    try {
        # Create an empty temporary folder
        $emptyFolder = Join-Path $env:TEMP "empty"
        New-Item -ItemType Directory -Path $emptyFolder -Force -ErrorAction SilentlyContinue

        # Use robocopy to force deletion
        Write-Host "Attempting deletion using robocopy..."
        $robocopyArgs = @("/MIR", "/PURGE", $emptyFolder, $FolderPath)
        Start-Process "robocopy" -ArgumentList $robocopyArgs -NoNewWindow -Wait

        # Clean up temporary folder
        Remove-Item -Path $emptyFolder -Force -Recurse -ErrorAction SilentlyContinue
        
        # Final attempt to delete target folder
        [System.IO.Directory]::Delete($FolderPath, $true)
        
        if (-not (Test-Path $FolderPath)) {
            Write-Host "Folder successfully deleted."
            return $true
        }
        else {
            Write-Warning "Deletion failed. Please try the following:"
            Write-Warning "1. Restart your computer and try again"
            Write-Warning "2. Run the script with administrator privileges"
            Write-Warning "3. Use a dedicated file unlocker tool (e.g., LockHunter)"
            return $false
        }
    }
    catch {
        Write-Warning "Deletion failed. Error message: $_"
        return $false
    }
}