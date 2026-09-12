$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
python (Join-Path $RepositoryRoot "tools/install_component.py") --component capture @args
exit $LASTEXITCODE
