param([string]$BuildDirectory = 'build-local-active')

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousPath = $env:PATH
try {
    # Use the local compiler installation when available; otherwise use PATH.
    if (Test-Path -LiteralPath 'C:\mingw\mingw64\bin\cmake.exe') {
        $env:PATH = 'C:\mingw\mingw64\bin;' + $env:PATH
    }
    $cmake = (Get-Command cmake -ErrorAction Stop).Source
    if (-not [System.IO.Path]::IsPathRooted($BuildDirectory)) {
        $BuildDirectory = Join-Path $repoRoot $BuildDirectory
    }
    if (-not (Test-Path -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt'))) {
        & $cmake -S $repoRoot -B $BuildDirectory -G Ninja `
            -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF -DKVMEM_ENABLE_NVME=OFF
        if ($LASTEXITCODE -ne 0) { throw 'Demo configuration failed.' }
    }
    & $cmake --build $BuildDirectory --target historical_kv_gc_demo -j 2
    if ($LASTEXITCODE -ne 0) { throw 'Demo build failed.' }
    & (Join-Path $BuildDirectory 'bin\historical_kv_gc_demo.exe')
    if ($LASTEXITCODE -ne 0) { throw 'Demo checks failed.' }
} finally {
    $env:PATH = $previousPath
}
