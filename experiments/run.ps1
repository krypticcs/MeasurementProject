# Train + test one experiment and keep its logs and results.csv
# usage (from project root):
#   .\experiments\run.ps1 -Folder cnn_on_points -Name measw20 -TrainArgs "--model basic --meas_w 20 --select pe" -TestArgs "--model basic"
# saves <folder>/model_saves/<Name>_best.pth, experiments/logs/<Name>_<folder>_{train,test}.log, experiments/results/<Name>_<folder>.csv
param(
    [Parameter(Mandatory = $true)][string]$Folder,
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$TrainArgs = "",
    [string]$TestArgs = "",
    [string]$MData = "20260427.csv",
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force experiments\logs, experiments\results | Out-Null

#seg_heatmaps and dual_heatmaps read ./data/box_images directly and have no --idata flag
$idata = if ($Folder -in @("seg_heatmaps", "dual_heatmaps", "two_stage")) { "" } else { "--idata box_images" }
$tag = "${Name}_${Folder}"

if (-not $SkipTrain) {
    cmd /c "$py -m $Folder.train --mdata $MData $idata --seed 42 --run $Name $TrainArgs > experiments\logs\${tag}_train.log 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "train failed, see experiments\logs\${tag}_train.log" }
}

cmd /c "$py -m $Folder.test --mdata $MData $idata --ds test --makecsv --path ${Name}_best.pth $TestArgs > experiments\logs\${tag}_test.log 2>&1"
if ($LASTEXITCODE -ne 0) { throw "test failed, see experiments\logs\${tag}_test.log" }
Copy-Item "$Folder\test_results\results.csv" "experiments\results\${tag}.csv" -Force
Write-Output "done: $tag"
