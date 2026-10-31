$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$launcher = Join-Path $env:LOCALAPPDATA "Programs\Python\Launcher\py.exe"
if (Test-Path $launcher) {
    $pythonCommand = $launcher
    $pythonPrefix = @("-3")
} else {
    $launcherCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($launcherCommand) {
        $pythonCommand = $launcherCommand.Source
        $pythonPrefix = @("-3")
    } else {
        $python = Get-Command python3 -ErrorAction SilentlyContinue
        if (-not $python) {
            $python = Get-Command python -ErrorAction Stop
        }
        $pythonCommand = $python.Source
        $pythonPrefix = @()
    }
}

function Invoke-ProjectPython {
    param([string[]]$PythonArguments)
    & $pythonCommand @pythonPrefix @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

Invoke-ProjectPython -PythonArguments @("scripts/download_data.py")
Invoke-ProjectPython -PythonArguments @("scripts/prepare_data.py")
Invoke-ProjectPython -PythonArguments @("scripts/analyze_data.py")
Invoke-ProjectPython -PythonArguments @("scripts/train_baselines.py", "--model", "o-only")
Invoke-ProjectPython -PythonArguments @("scripts/train_baselines.py", "--model", "most-frequent-word")
Invoke-ProjectPython -PythonArguments @("scripts/train_baselines.py", "--model", "crf", "--max-train-contracts", "10", "--bootstrap-iterations", "200")
Invoke-ProjectPython -PythonArguments @("scripts/compare_models.py", "results/most-frequent-word-seed42_predictions.jsonl", "results/o-only-seed42_predictions.jsonl")
Invoke-ProjectPython -PythonArguments @("scripts/audit_predictions.py", "results/most-frequent-word-seed42_predictions.jsonl")
Invoke-ProjectPython -PythonArguments @("scripts/summarize_experiments.py")
Invoke-ProjectPython -PythonArguments @("scripts/build_presentation.py")
Invoke-ProjectPython -PythonArguments @("-m", "unittest", "discover", "-s", "tests")
