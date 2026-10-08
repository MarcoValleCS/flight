# Pipeline completa FlightPREV (eseguire dalla cartella del progetto)
# Uso:  powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"

Write-Host "=== 1/6 dataset train (03-05 aprile) ===" -ForegroundColor Cyan
python -m src.build_dataset --files Italy_24-04-03.csv Italy_24-04-04.csv Italy_24-04-05.csv --split train

Write-Host "=== 2/6 dataset val (06 aprile) ===" -ForegroundColor Cyan
python -m src.build_dataset --files Italy_24-04-06.csv --split val

Write-Host "=== 3/6 dataset test (07 aprile) ===" -ForegroundColor Cyan
python -m src.build_dataset --files Italy_24-04-07.csv --split test

Write-Host "=== 4/6 training ===" -ForegroundColor Cyan
python -m src.train --run runs/transformer_v1

Write-Host "=== 5/6 valutazione ===" -ForegroundColor Cyan
python -m src.evaluate --run runs/transformer_v1/best.pt --split test

Write-Host "=== 6/6 demo di predizione ===" -ForegroundColor Cyan
python -m src.predict --run runs/transformer_v1/best.pt --n 6

Write-Host "Fatto. Report in .\reports\" -ForegroundColor Green
