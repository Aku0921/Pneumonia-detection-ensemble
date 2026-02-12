# Training script for all 5 backbones
# Run this from the project root directory

$backbones = @(
    "efficientnetb0",
    "mobilenetv2",
    "resnet50v2",
    "densenet121",
    "vgg16"
)

$top_epochs = 10
$fine_tune_epochs = 5

foreach ($backbone in $backbones) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Training $backbone" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    python -m src.train `
        --backbone $backbone `
        --top_epochs $top_epochs `
        --fine_tune_epochs $fine_tune_epochs `
        --batch_size 32
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Training failed for $backbone" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "$backbone training completed successfully!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "All models trained successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
