$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 (Join-Path $Root "setup_env.py") @args
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python (Join-Path $Root "setup_env.py") @args
}
else {
    throw "Python 3.8 or newer is required to bootstrap the environment."
}

exit $LASTEXITCODE
