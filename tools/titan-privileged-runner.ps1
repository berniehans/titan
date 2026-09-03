# Titan privileged benchmark broker. This file is intended to be immutable after installation.
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$BrokerRoot = Join-Path $ProjectRoot 'local-artifacts\privileged-runner'
$QueueDir = Join-Path $BrokerRoot 'queue'
$RunningDir = Join-Path $BrokerRoot 'running'
$ResultsDir = Join-Path $BrokerRoot 'results'
$TaskName = 'Titan-PrivilegedRunner'

function Write-ImmutableJson {
    param(
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [object] $Value
    )

    $json = $Value | ConvertTo-Json -Depth 8
    $temp = Join-Path ([IO.Path]::GetDirectoryName($Path)) ('.{0}.{1}.tmp' -f [IO.Path]::GetFileName($Path), [guid]::NewGuid().ToString('N'))
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json + [Environment]::NewLine)
        $stream = [IO.File]::Open($temp, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try { $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Dispose() }
        [IO.File]::Move($temp, $Path)
    } finally {
        if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue }
    }
}

function Get-FixedOperation {
    param([Parameter(Mandatory)] [string] $Operation)

    # Arguments are deliberately constants. Do not add job-controlled arguments here.
    switch ($Operation) {
        'gpu-parity-suite' {
            return [pscustomobject]@{
                executable = 'cargo.exe'
                arguments = @('test', '--workspace', '--', '--ignored', '--test-threads=1')
                working_directory = Join-Path $ProjectRoot 'engine'
            }
        }
        'dense-decode-benchmark' {
            return [pscustomobject]@{
                executable = 'cargo.exe'
                arguments = @('test', '--release', '-p', 'engine-server', '--test', 'multi_model_comparison_bench', '--', '--ignored', '--nocapture')
                working_directory = Join-Path $ProjectRoot 'engine'
            }
        }
        'ncu-titan-3b' {
            return [pscustomobject]@{
                executable = 'C:/Program Files/NVIDIA Corporation/Nsight Compute 2025.4.1/target/windows-desktop-win7-x64/ncu.exe'
                arguments = @('--target-processes', 'all', '--set', 'basic', '--kernel-name-base', 'function', '--kernel-name', 'regex:^gemm_q4k_fused_gate_up_swiglu_mma_kernel$', '--launch-count', '1', '--export', '..\local-artifacts/privileged-runner/titan-3b-gate-up', '--force', 'cargo.exe', 'test', '--release', '-p', 'engine-server', '--test', 'multi_model_comparison_bench', '--', '--ignored', '--nocapture')
                working_directory = Join-Path $ProjectRoot 'engine'
            }
        }
        default { throw "unrecognized operation: $Operation" }
    }
}

function New-Rejection {
    param([string] $Reason, [string] $InputFile, [string] $JobId)
    return [ordered]@{
        schema_version = 1
        status = 'rejected'
        job_id = $JobId
        input_file = $InputFile
        error = $Reason
        completed_utc = [DateTime]::UtcNow.ToString('o')
    }
}

$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "$TaskName must run with an elevated token"
}

foreach ($directory in @($QueueDir, $RunningDir, $ResultsDir)) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) { throw "missing broker directory: $directory" }
}

$files = @(Get-ChildItem -LiteralPath $QueueDir -Filter '*.json' -File | Sort-Object Name)
foreach ($file in $files) {
    $claimed = Join-Path $RunningDir $file.Name
    try {
        # Same-volume rename is atomic. A competing runner loses this move and skips the job.
        Move-Item -LiteralPath $file.FullName -Destination $claimed -ErrorAction Stop
    } catch {
        continue
    }

    $jobId = [IO.Path]::GetFileNameWithoutExtension($file.Name)
    $safeJobId = $jobId -match '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'
    $resultName = if ($safeJobId) { "$jobId.json" } else { "rejected-$([guid]::NewGuid().ToString('N')).json" }
    $resultPath = Join-Path $ResultsDir $resultName
    $result = $null

    try {
        $raw = Get-Content -LiteralPath $claimed -Raw -Encoding UTF8
        $job = $raw | ConvertFrom-Json
        if ($null -eq $job -or $job -is [array]) { throw 'job must be a JSON object' }
        $properties = @($job.PSObject.Properties.Name | Sort-Object)
        if (($properties -join ',') -ne 'job_id,operation') { throw 'job must contain exactly job_id and operation' }
        if ($job.job_id -isnot [string] -or $job.operation -isnot [string]) { throw 'job_id and operation must be strings' }
        if ($job.job_id -ne $jobId) { throw 'JSON job_id must match the queue filename' }
        if ($jobId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') { throw 'invalid job_id' }
        $fixed = Get-FixedOperation -Operation $job.operation
        if (-not (Test-Path -LiteralPath $fixed.working_directory -PathType Container)) { throw "missing fixed working directory" }

        $command = Get-Command $fixed.executable -CommandType Application -ErrorAction Stop
        $psi = [Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $command.Source
        $psi.WorkingDirectory = $fixed.working_directory
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
$psi.Environment['TITAN_CUDA_DLL_DIR'] = 'C:/Users/niber/AppData/Local/hermes/workspace/titan/local-artifacts/nvrtc-cu12/runtime'
$psi.Environment['TITAN_BENCHMARK_MODEL_FILTER'] = 'Llama 3.2 3B'
$psi.Environment['TITAN_BENCHMARK_SKIP_LLAMA'] = '1'
$psi.Environment['TITAN_BENCHMARK_REPETITIONS'] = '1'
$psi.Environment['TITAN_BENCHMARK_DISPATCH_TELEMETRY'] = '1'
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        # The only arguments assigned here are the constants returned by Get-FixedOperation.
        $psi.Arguments = (($fixed.arguments | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' ')
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $psi
        if (-not $process.Start()) { throw 'failed to start fixed command' }
        # Drain both pipes concurrently so a verbose benchmark cannot deadlock the broker.
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $stdout = $stdoutTask.Result
        $stderr = $stderrTask.Result
        $result = [ordered]@{
            schema_version = 1
            status = if ($process.ExitCode -eq 0) { 'completed' } else { 'failed' }
            job_id = $jobId
            operation = $job.operation
            exit_code = $process.ExitCode
            stdout = $stdout
            stderr = $stderr
            completed_utc = [DateTime]::UtcNow.ToString('o')
        }
        $process.Dispose()
    } catch {
        $result = New-Rejection -Reason $_.Exception.Message -InputFile $file.Name -JobId $jobId
    }

    if (-not (Test-Path -LiteralPath $resultPath)) {
        Write-ImmutableJson -Path $resultPath -Value $result
    }
    Remove-Item -LiteralPath $claimed -Force -ErrorAction SilentlyContinue
}
