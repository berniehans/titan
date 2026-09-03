# Titan Privileged Runner

`Titan-PrivilegedRunner` is a Windows-only, allowlist-based broker for the GPU operations that require elevation. It is not a general command runner.

## Install

Open an elevated PowerShell window and run:

```powershell
& .\tools\install-titan-privileged-runner.ps1
```

The installer creates `local-artifacts/privileged-runner/{queue,running,results}`, removes inherited ACLs, grants the current user Modify access only to those job/result directories, and protects both scripts with SYSTEM/Administrators-only write access. It registers the task for the current interactive user with `RunLevel Highest` and a fixed PowerShell action.

The task definition receives a best-effort ACL (`SYSTEM` and `Administrators` Full Control; the current user Read/Execute). Windows Task Scheduler does not provide a portable `schtasks` option for a Run-only task ACE, and the task is registered for the current user. If Windows rejects the task-file ACL, the installer prints a warning; the exact limitation is that the current user may retain Task Scheduler modify rights. A separate administrator-owned task/security descriptor is required for strict Run-only enforcement.

## Non-elevated client

A client must write a JSON file atomically into the queue. The filename and `job_id` must match, and the object must contain exactly these two string properties:

```powershell
$queue = 'C:\path\to\titan\local-artifacts\privileged-runner\queue'
$id = 'job-001'
$tmp = Join-Path $queue ".$id.tmp"
$dst = Join-Path $queue "$id.json"
Set-Content -LiteralPath $tmp -Value '{"job_id":"job-001","operation":"gpu-parity-suite"}' -Encoding UTF8
Move-Item -LiteralPath $tmp -Destination $dst
schtasks.exe /Run /TN Titan-PrivilegedRunner
```

Allowed operations are exactly:

- `gpu-parity-suite` — fixed `cargo test --workspace -- --ignored --test-threads=1`, run from the Cargo workspace at `engine/`
- `dense-decode-benchmark` — fixed release `multi_model_comparison_bench` test, run from `engine/`
- `ncu-titan-3b` — fixed Nsight Compute invocation around that benchmark, run from `engine/` and exporting to the project-root `local-artifacts/privileged-runner/` directory

Read `results/<job-id>.json` for `status`, `exit_code`, `stdout`, and `stderr`. Malformed, extra-property, filename-mismatched, and unrecognized jobs are rejected. Results are created once and never overwritten by the runner.

The broker never accepts a command, executable, path, working-directory, argument, or environment override from a job. All commands and directories are constants in the protected runner script.
