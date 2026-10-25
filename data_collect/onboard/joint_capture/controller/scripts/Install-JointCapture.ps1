[CmdletBinding()]
param(
    [string]$TargetRoot = 'D:\OneDriveData\Desktop',
    [string]$SourceRoot,
    [string]$PythonPath,
    [string]$ManifestRoot = 'D:\JointCaptureData\manifests'
)

$ErrorActionPreference = 'Stop'
if (-not $SourceRoot) { $SourceRoot = Split-Path -Parent $PSScriptRoot }
$source = (Resolve-Path -LiteralPath $SourceRoot).Path
$target = [IO.Path]::GetFullPath($TargetRoot)
$manifest = [IO.Path]::GetFullPath($ManifestRoot)
if (-not (Test-Path -LiteralPath (Join-Path $source 'src\jointctl\__main__.py') -PathType Leaf)) {
    throw "SourceRoot is not a JointCapture repository: $source"
}
if (Test-Path -LiteralPath $target -PathType Leaf) {
    throw "TargetRoot is a file: $target"
}
New-Item -ItemType Directory -Force -Path $target | Out-Null

if ($PythonPath) {
    $python = (Resolve-Path -LiteralPath $PythonPath).Path
} else {
    $pythonCommand = Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1
    $python = [IO.Path]::GetFullPath($pythonCommand.Source)
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable was not found: $python"
}

$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$backup = Join-Path $target "JointCaptureBackup-$stamp"
$suffix = 0
while (Test-Path -LiteralPath $backup) {
    $suffix++
    $backup = Join-Path $target "JointCaptureBackup-$stamp-$suffix"
}
$stage = Join-Path $target ".JointCapture.install-$stamp"
$application = Join-Path $target 'JointCapture'
if (Test-Path -LiteralPath $stage) {
    throw "Staging path already exists: $stage"
}

New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -LiteralPath (Join-Path $source 'src') -Destination $stage -Recurse
Copy-Item -LiteralPath (Join-Path $source 'config') -Destination $stage -Recurse
Copy-Item -LiteralPath (Join-Path $source 'scripts') -Destination $stage -Recurse
Copy-Item -LiteralPath (Join-Path $source 'pyproject.toml') -Destination $stage
if (Test-Path -LiteralPath (Join-Path $source 'README.md')) {
    Copy-Item -LiteralPath (Join-Path $source 'README.md') -Destination $stage
}
$pythonFile = Join-Path $stage 'python_path.txt'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($pythonFile, $python + [Environment]::NewLine, $utf8NoBom)
[IO.File]::WriteAllText((Join-Path $stage 'manifest_root.txt'), $manifest + [Environment]::NewLine, $utf8NoBom)

$legacyNames = @('JointStart.bat', 'JointStop.bat', 'JointStatus.bat', 'JointRecover.bat', 'JointConsole.bat',
                 'joint_start.ps1', 'joint_stop.ps1', 'Joint_episode_counter.txt')
$hasBackup = (Test-Path -LiteralPath $application)
foreach ($name in $legacyNames) {
    if (Test-Path -LiteralPath (Join-Path $target $name)) { $hasBackup = $true }
}
if ($hasBackup) {
    New-Item -ItemType Directory -Path $backup | Out-Null
    if (Test-Path -LiteralPath $application) {
        Move-Item -LiteralPath $application -Destination (Join-Path $backup 'JointCapture')
    }
    foreach ($name in $legacyNames) {
        $existing = Join-Path $target $name
        if (Test-Path -LiteralPath $existing) {
            Copy-Item -LiteralPath $existing -Destination (Join-Path $backup $name)
        }
    }
}

Move-Item -LiteralPath $stage -Destination $application
foreach ($name in @('JointStart.bat', 'JointStop.bat', 'JointStatus.bat', 'JointRecover.bat', 'JointConsole.bat')) {
    Copy-Item -LiteralPath (Join-Path $source "scripts\$name") -Destination (Join-Path $target $name) -Force
}

[pscustomobject]@{
    Application = $application
    Python = $python
    ManifestRoot = $manifest
    Backup = $(if ($hasBackup) { $backup } else { $null })
} | ConvertTo-Json
