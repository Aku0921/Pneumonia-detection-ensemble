# Evaluate all 2-model ensemble combinations

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Evaluating All 2-Model Ensembles" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python -m src.ensemble `
    --models_dir models `
    --data_dir data/chest_xray `
    --img_size 224 `
    --batch_size 32 `
    --output_dir models/ensemble_results_2model

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Ensemble evaluation failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Ensemble evaluation complete!" -ForegroundColor Green
Write-Host "Results saved to: models/ensemble_results_2model/" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
