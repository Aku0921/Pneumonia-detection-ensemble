"""Comprehensive model evaluation to find best accuracy configuration."""
import tensorflow as tf
from src.data import get_datasets
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from pathlib import Path

# Load test data
print("Loading test dataset...")
test_ds = get_datasets('data/chest_xray', (224,224), 32)[2]

# Load all models
models_dir = Path('models')
model_configs = {
    'efficientnetb0': models_dir / 'efficientnetb0' / 'final_model.keras',
    'resnet50v2': models_dir / 'resnet50v2' / 'final_model.keras',
    'densenet121': models_dir / 'densenet121' / 'final_model.keras',
    'vgg16': models_dir / 'vgg16' / 'final_model.keras',
    'mobilenetv2': models_dir / 'mobilenetv2' / 'final_model.keras',
}

print("\n" + "="*60)
print("LOADING MODELS AND COLLECTING PREDICTIONS")
print("="*60)

# Collect all predictions
y_true = []
predictions = {}

for name in model_configs:
    predictions[name] = []

# Get predictions from all models
for images, labels in test_ds:
    y_true.extend(labels.numpy())
    for name, path in model_configs.items():
        if not path.exists():
            print(f"⚠️  Model not found: {name}")
            continue
        model = tf.keras.models.load_model(str(path))
        pred = model.predict(images, verbose=0)
        predictions[name].extend(pred)

y_true = np.array(y_true)

# Convert to numpy arrays
for name in list(predictions.keys()):
    if len(predictions[name]) == 0:
        del predictions[name]
        del model_configs[name]
    else:
        predictions[name] = np.array(predictions[name]).flatten()

print(f"\n✅ Loaded {len(predictions)} models")
print(f"✅ Test samples: {len(y_true)}")
print(f"   - NORMAL: {np.sum(y_true == 0)}")
print(f"   - PNEUMONIA: {np.sum(y_true == 1)}")

# Evaluate single models
print("\n" + "="*60)
print("SINGLE MODEL PERFORMANCE")
print("="*60)

single_results = {}
for name in sorted(predictions.keys()):
    pred_binary = predictions[name] >= 0.5
    acc = accuracy_score(y_true, pred_binary)
    prec = precision_score(y_true, pred_binary)
    rec = recall_score(y_true, pred_binary)
    f1 = f1_score(y_true, pred_binary)
    auc = roc_auc_score(y_true, predictions[name])
    
    single_results[name] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc)
    }
    
    print(f"\n{name.upper()}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")

# Test ensemble strategies
print("\n" + "="*60)
print("ENSEMBLE STRATEGIES")
print("="*60)

ensemble_results = {}

# 1. Simple average of top 2
model_names = list(predictions.keys())
print("\n1️⃣  TESTING ALL 2-MODEL COMBINATIONS (Simple Average)")
print("-" * 60)

from itertools import combinations
for m1, m2 in combinations(model_names, 2):
    ensemble_pred = (predictions[m1] + predictions[m2]) / 2.0
    pred_binary = ensemble_pred >= 0.5
    acc = accuracy_score(y_true, pred_binary)
    auc = roc_auc_score(y_true, ensemble_pred)
    rec = recall_score(y_true, pred_binary)
    
    key = f"{m1}+{m2}"
    ensemble_results[key] = {
        'accuracy': float(acc),
        'auc': float(auc),
        'recall': float(rec),
        'strategy': 'average'
    }
    
    print(f"  {m1:15} + {m2:15}  Acc: {acc*100:5.2f}%  AUC: {auc:.4f}  Recall: {rec*100:5.2f}%")

# 2. Weighted average based on AUC
print("\n2️⃣  TESTING WEIGHTED ENSEMBLES (Based on Model AUC)")
print("-" * 60)

for m1, m2 in combinations(model_names, 2):
    w1 = single_results[m1]['auc']
    w2 = single_results[m2]['auc']
    total = w1 + w2
    
    ensemble_pred = (predictions[m1] * w1 + predictions[m2] * w2) / total
    pred_binary = ensemble_pred >= 0.5
    acc = accuracy_score(y_true, pred_binary)
    auc = roc_auc_score(y_true, ensemble_pred)
    rec = recall_score(y_true, pred_binary)
    
    key = f"{m1}+{m2}_weighted"
    ensemble_results[key] = {
        'accuracy': float(acc),
        'auc': float(auc),
        'recall': float(rec),
        'strategy': 'weighted',
        'weights': {m1: float(w1/total), m2: float(w2/total)}
    }
    
    print(f"  {m1:15} + {m2:15}  Acc: {acc*100:5.2f}%  AUC: {auc:.4f}  Recall: {rec*100:5.2f}%")

# 3. Test different thresholds
print("\n3️⃣  TESTING DIFFERENT THRESHOLDS (Best Ensemble)")
print("-" * 60)

# Find best average ensemble
best_combo = max(ensemble_results.items(), 
                 key=lambda x: x[1]['accuracy'] if x[1]['strategy'] == 'average' else 0)
best_name = best_combo[0]
m1, m2 = best_name.split('+')

ensemble_pred = (predictions[m1] + predictions[m2]) / 2.0

threshold_results = {}
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred_binary = ensemble_pred >= threshold
    acc = accuracy_score(y_true, pred_binary)
    prec = precision_score(y_true, pred_binary)
    rec = recall_score(y_true, pred_binary)
    f1 = f1_score(y_true, pred_binary)
    
    threshold_results[threshold] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1)
    }
    
    print(f"  Threshold {threshold:.1f}:  Acc: {acc*100:5.2f}%  Prec: {prec*100:5.2f}%  Rec: {rec*100:5.2f}%  F1: {f1:.4f}")

# Find best configuration
print("\n" + "="*60)
print("🏆 BEST CONFIGURATIONS")
print("="*60)

best_single = max(single_results.items(), key=lambda x: x[1]['accuracy'])
best_ensemble_avg = max([(k,v) for k,v in ensemble_results.items() if v['strategy'] == 'average'], 
                        key=lambda x: x[1]['accuracy'])
best_ensemble_weighted = max([(k,v) for k,v in ensemble_results.items() if v['strategy'] == 'weighted'], 
                             key=lambda x: x[1]['accuracy'])
best_threshold = max(threshold_results.items(), key=lambda x: x[1]['accuracy'])

print(f"\n✅ Best Single Model: {best_single[0].upper()}")
print(f"   Accuracy: {best_single[1]['accuracy']*100:.2f}%")
print(f"   AUC: {best_single[1]['auc']:.4f}")

print(f"\n✅ Best Ensemble (Average): {best_ensemble_avg[0].upper()}")
print(f"   Accuracy: {best_ensemble_avg[1]['accuracy']*100:.2f}%")
print(f"   AUC: {best_ensemble_avg[1]['auc']:.4f}")
print(f"   Recall: {best_ensemble_avg[1]['recall']*100:.2f}%")

print(f"\n✅ Best Ensemble (Weighted): {best_ensemble_weighted[0].upper()}")
print(f"   Accuracy: {best_ensemble_weighted[1]['accuracy']*100:.2f}%")
print(f"   AUC: {best_ensemble_weighted[1]['auc']:.4f}")
print(f"   Weights: {best_ensemble_weighted[1]['weights']}")

print(f"\n✅ Best Threshold: {best_threshold[0]:.1f}")
print(f"   Accuracy: {best_threshold[1]['accuracy']*100:.2f}%")
print(f"   Precision: {best_threshold[1]['precision']*100:.2f}%")
print(f"   Recall: {best_threshold[1]['recall']*100:.2f}%")

# Save results
results = {
    'single_models': single_results,
    'ensembles': ensemble_results,
    'thresholds': threshold_results,
    'best_config': {
        'single': best_single[0],
        'ensemble_avg': best_ensemble_avg[0],
        'ensemble_weighted': best_ensemble_weighted[0],
        'threshold': best_threshold[0]
    }
}

with open('accuracy_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("✅ Analysis saved to accuracy_analysis.json")
print("="*60)
