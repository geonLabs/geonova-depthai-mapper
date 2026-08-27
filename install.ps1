$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Join-Path $RepositoryRoot "code"
$Installer = Join-Path $ProjectRoot "scripts/setup_env.ps1"
$Config = Join-Path $ProjectRoot "configs/setup.yaml"
$VirtualEnvironment = Join-Path $RepositoryRoot ".venv"
$HasExplicitVenv = $false

for ($index = 0; $index -lt $args.Length; $index++) {
    $argument = $args[$index]
    if ($argument -eq "--venv" -or $argument -like "--venv=*") {
        $HasExplicitVenv = $true
    }
}

if (-not (Test-Path $Installer -PathType Leaf)) {
    throw "Installer not found: $Installer"
}
if (Test-Path $VirtualEnvironment) {
    $VirtualEnvironmentItem = Get-Item -Force $VirtualEnvironment
    if (($VirtualEnvironmentItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing symbolic-link virtualenv: $VirtualEnvironment"
    }
}

Push-Location $ProjectRoot
try {
    if ($HasExplicitVenv) {
        & $Installer --config $Config @args
    }
    else {
        & $Installer --config $Config @args --venv $VirtualEnvironment
    }
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
