param([string]$GpuHost = "16.7", [string]$UnitreeHost = "192.168.0.147", [string]$P450Host = "192.168.1.11")
$ErrorActionPreference = "Stop"
foreach ($robot in @($UnitreeHost, $P450Host)) {
    if (-not (Test-NetConnection $robot -Port 22 -InformationLevel Quiet)) {
        throw "Robot SSH port unreachable from Windows: ${robot}:22. Connect the robot network first."
    }
}
Write-Host "Keep this window open. No shell prompt while SSH forwarding is active is normal."
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 `
    -R "127.0.0.1:22271:${UnitreeHost}:22" `
    -R "127.0.0.1:22272:${P450Host}:22" $GpuHost
if ($LASTEXITCODE -ne 0) { throw "SSH tunnel exited with code $LASTEXITCODE" }
