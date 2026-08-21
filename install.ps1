$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Join-Path $RepositoryRoot "code"
$Installer = Join-Path $ProjectRoot "scripts/setup_env.ps1"
$Config = Join-Path $ProjectRoot "configs/setup.yaml"

if (-not (Test-Path $Installer -PathType Leaf)) {
    throw "Installer not found: $Installer"
}

Push-Location $ProjectRoot
try {
    & $Installer --config $Config @args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
