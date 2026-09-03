# Installs the Titan least-privilege scheduled-task broker.
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$ToolsDir = $PSScriptRoot
$RunnerPath = Join-Path $ToolsDir 'titan-privileged-runner.ps1'
$BrokerRoot = Join-Path $ProjectRoot 'local-artifacts\privileged-runner'
$QueueDir = Join-Path $BrokerRoot 'queue'
$RunningDir = Join-Path $BrokerRoot 'running'
$ResultsDir = Join-Path $BrokerRoot 'results'
$TaskName = 'Titan-PrivilegedRunner'
$identity = '{0}\{1}' -f $env:USERDOMAIN, $env:USERNAME

$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this installer from an elevated PowerShell token.'
}
if (-not (Test-Path -LiteralPath $RunnerPath -PathType Leaf)) { throw "Runner not found: $RunnerPath" }

foreach ($directory in @($BrokerRoot, $QueueDir, $RunningDir, $ResultsDir)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

function Set-LockedAcl {
    param([Parameter(Mandatory)] [string] $Path, [Parameter(Mandatory)] [ValidateSet('directory','file')] [string] $Kind)
    $inheritance = if ($Kind -eq 'directory') { '(OI)(CI)' } else { '' }
    $rules = if ($Kind -eq 'directory') {
        @("${identity}:$($inheritance)M", "SYSTEM:$($inheritance)F", "Administrators:$($inheritance)F")
    } else {
        @('SYSTEM:F', 'Administrators:F')
    }
    & icacls.exe $Path '/inheritance:r' '/grant:r' $rules | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "icacls failed for $Path ($LASTEXITCODE)" }
}

function Set-OwnerAcl {
    param([Parameter(Mandatory)] [string] $Path, [ValidateSet('M','RX')] [string] $Rights = 'M')
    & icacls.exe $Path '/inheritance:r' '/grant:r' "${identity}:(OI)(CI)$Rights" 'SYSTEM:(OI)(CI)F' 'Administrators:(OI)(CI)F' | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "icacls failed for $Path ($LASTEXITCODE)" }
}

# The broker root and queue lifecycle directories are not writable by ordinary users.
Set-LockedAcl -Path $BrokerRoot -Kind directory
Set-OwnerAcl -Path $QueueDir
Set-OwnerAcl -Path $RunningDir
Set-OwnerAcl -Path $ResultsDir -Rights RX
Set-LockedAcl -Path $RunnerPath -Kind file
Set-LockedAcl -Path $PSCommandPath -Kind file

$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$RunnerPath`""
$taskPrincipal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$task = New-ScheduledTask -Action $action -Principal $taskPrincipal -Settings $settings -Description 'Runs only the fixed Titan GPU benchmark allowlist.'
Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null

# Best effort: protect the task definition from ordinary file writes. Windows Task Scheduler
# does not expose a portable schtasks flag for a Run-only ACE; the exact limitation is documented.
$taskFile = Join-Path "$env:SystemRoot\System32\Tasks" $TaskName
$taskAclWarning = $null
try {
    if (Test-Path -LiteralPath $taskFile) {
        & icacls.exe $taskFile '/inheritance:r' '/grant:r' 'SYSTEM:F' 'Administrators:F' "${identity}:RX" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "icacls failed ($LASTEXITCODE)" }
    } else {
        throw "task definition was not found at $taskFile"
    }
} catch {
    $taskAclWarning = $_.Exception.Message
    Write-Warning "Could not apply task-definition ACL: $taskAclWarning"
}

Write-Output "Installed $TaskName for $identity"
Write-Output "Queue:   $QueueDir"
Write-Output "Results: $ResultsDir"
if ($null -ne $taskAclWarning) {
    Write-Warning 'Task ACL limitation: the user may retain Task Scheduler modify rights; use a separate administrator-owned task/security descriptor if Run-only access is mandatory.'
}
