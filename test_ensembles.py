import tensorflow as tf
from src.data import get_datasets
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

test_ds = get_datasets('data/chest_xray', (224,224), 32)[2]

# Test all 5 single models
models = {
    'efficientnetb0': 'models/efficientnetb0/final_model.keras',
    'resnet50v2': 'models/resnet50v2/final_model.keras',
    'densenet121': 'models/densenet121/final_model.keras',
    'vgg16': 'models/vgg16/final_model.keras',
    'mobilenetv2': 'models/mobilenetv2/final_model.keras',
}

y_true = []
predictions = {}

for name in models:
    predictions[name] = []

# Load all predictions
for images, labels in test_ds:
    y_true.extend(labels.numpy())
    for name, path in models.items():
        model = tf.keras.models.load_model(path)
        pred = model.predict(images, verbose=0)
        predictions[name].extend(pred)

y_true = np.array(y_true)

# Convert to numpy arrays and flatten
for name in predictions:
    predictions[name] = np.array(predictions[name]).flatten()

# Test best combinations
combinations = [
    ('efficientnetb0', 'densenet121'),
    ('resnet50v2', 'densenet121'),
    ('efficientnetb0', 'resnet50v2'),
    ('efficientnetb0', 'vgg16'),
    ('densenet121', 'vgg16'),
]

results = {}
print("\n=== SINGLE MODELS ===")
for name in models:
    pred = predictions[name] >= 0.5
    acc = accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, predictions[name])
    results[name] = {'accuracy': float(acc), 'auc': float(auc)}
    print(f"{name:15} Acc: {acc:.4f}  AUC: {auc:.4f}")

print("\n=== BEST ENSEMBLES (Average) ===")
for m1, m2 in combinations:
    ensemble_pred = (predictions[m1] + predictions[m2]) / 2.0
    pred = ensemble_pred >= 0.5
    acc = accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, ensemble_pred)
    key = f"{m1}__{m2}"
    results[key] = {'accuracy': float(acc), 'auc': float(auc)}
    print(f"{m1} + {m2:12} Acc: {acc:.4f}  AUC: {auc:.4f}")

# Save results
with open('ensemble_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Best single model:", max(results, key=lambda x: results[x]['auc'] if '__' not in x else -1))
print("✅ Best ensemble:", max(results, key=lambda x: results[x]['auc'] if '__' in x else -1))
