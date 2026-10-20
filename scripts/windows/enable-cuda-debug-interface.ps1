#Requires -RunAsAdministrator
<#
Enables NVIDIA's documented CUDA debugger interface on the explicitly named
Windows host. Does not change the display timeout, driver, or running jobs.
Run manually on that host after its administrator approves the change.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory = $true)]
    [string] $ExpectedComputerName
)
$ErrorActionPreference = 'Stop'
if ($env:COMPUTERNAME -ine $ExpectedComputerName) {
    throw "Wrong host: expected $ExpectedComputerName, found $env:COMPUTERNAME."
}
$registryKey = 'HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger'
$providerKey = 'HKLM:\SOFTWARE\NVIDIA Corporation\GPUDebugger'
$keyExists = Test-Path -LiteralPath $providerKey -ErrorAction Stop
$key = if ($keyExists) { Get-Item -LiteralPath $providerKey -ErrorAction Stop } else { $null }
$valueExists = $key -and ($key.GetValueNames() -contains 'EnableInterface')
$previousValue = if ($valueExists) { $key.GetValue('EnableInterface') } else { $null }
$previousKind = if ($valueExists) { $key.GetValueKind('EnableInterface').ToString() } else { $null }
if ($valueExists -and $previousValue -eq 1 -and $previousKind -eq 'DWord') {
    Write-Output 'NVIDIA CUDA debugger interface is already enabled.'
    return
}
if (-not $PSCmdlet.ShouldProcess("$env:COMPUTERNAME\$registryKey", 'Set EnableInterface (DWORD) to 1')) {
    return
}
$backupPath = Join-Path $env:TEMP ("kvmem-cuda-debug-interface-before-{0}.json" -f [guid]::NewGuid())
$before = @{
    computerName = $env:COMPUTERNAME
    registryKey = $registryKey
    valueName = 'EnableInterface'
    keyExisted = $keyExists
    valueExisted = [bool]$valueExists
    value = $previousValue
    valueKind = $previousKind
}
$before | ConvertTo-Json | Set-Content -LiteralPath $backupPath -Encoding UTF8
& reg.exe add $registryKey /v EnableInterface /t REG_DWORD /d 1 /f
if ($LASTEXITCODE -ne 0) { throw "Registry update failed. Previous state saved in $backupPath" }
$verified = Get-ItemProperty -LiteralPath $providerKey -Name EnableInterface
if ($verified.EnableInterface -ne 1) { throw "Registry verification failed. Previous state: $backupPath" }
Write-Output "Enabled NVIDIA CUDA debugger interface on $env:COMPUTERNAME. Previous state: $backupPath"
