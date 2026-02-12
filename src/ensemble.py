import os
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras

# MUST be called before loading any models with Lambda layers
keras.config.enable_unsafe_deserialization()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def load_model_safe(model_path):
    """Load model with unsafe deserialization enabled."""
    return keras.models.load_model(model_path)


def get_predictions_on_ds(model, ds):
    """Get true labels and predictions for a dataset."""
    y_true = []
    y_scores = []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_scores.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_scores)


def voting_ensemble(predictions_list):
    """Voting ensemble: majority vote on binary predictions (0.5 threshold)."""
    y_preds = [(p >= 0.5).astype(int) for p in predictions_list]
    votes = np.sum(y_preds, axis=0)
    # Majority vote: > len/2 means class 1
    return (votes > len(predictions_list) / 2).astype(float)


def averaging_ensemble(predictions_list):
    """Averaging ensemble: average probabilities."""
    return np.mean(predictions_list, axis=0)


def evaluate_ensemble_combination(models, model_names, test_ds):
    """Evaluate a single ensemble combination using voting and averaging."""
    # Get predictions from all models
    all_y_true = None
    predictions = []
    
    for model in models:
        y_true, y_scores = get_predictions_on_ds(model, test_ds)
        if all_y_true is None:
            all_y_true = y_true
        predictions.append(y_scores)
    
    # Voting ensemble
    y_pred_voting = voting_ensemble(predictions)
    
    # Averaging ensemble
    y_pred_averaging = (averaging_ensemble(predictions) >= 0.5).astype(int)
    y_score_averaging = averaging_ensemble(predictions)
    
    # Calculate metrics for averaging ensemble
    accuracy = accuracy_score(all_y_true, y_pred_averaging)
    precision = precision_score(all_y_true, y_pred_averaging, zero_division=0)
    recall = recall_score(all_y_true, y_pred_averaging, zero_division=0)
    f1 = f1_score(all_y_true, y_pred_averaging, zero_division=0)
    
    try:
        fpr, tpr, _ = roc_curve(all_y_true, y_score_averaging)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = 0.0
    
    return {
        'y_true': all_y_true,
        'y_pred': y_pred_averaging,
        'y_score': y_score_averaging,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'model_names': model_names
    }


def evaluate_all_2model_ensembles(models_dir, data_dir, img_size=(224, 224), batch_size=16):
    """Evaluate all 2-model ensemble combinations."""
    from .data import get_datasets
    
    # Define the 5 backbones
    backbones = ['efficientnetb0', 'mobilenetv2', 'resnet50v2', 'densenet121', 'vgg16']
    model_paths = {
        bb: os.path.join(models_dir, bb, 'final_model.keras')
        for bb in backbones
    }
    
    # Load all models
    print("Loading all 5 models...")
    loaded_models = {}
    for bb, path in model_paths.items():
        if os.path.exists(path):
            try:
                loaded_models[bb] = load_model_safe(path)
                print(f"✓ Loaded {bb}")
            except Exception as e:
                print(f"✗ Failed to load {bb}: {str(e)}")
        else:
            print(f"✗ Model not found: {path}")
    
    if len(loaded_models) < 5:
        raise ValueError(f"Expected 5 models, but only found {len(loaded_models)}")
    
    # Load test dataset
    print(f"\nLoading test dataset from {data_dir}...")
    _, _, test_ds = get_datasets(data_dir, img_size=img_size, batch_size=batch_size)
    
    # Generate all 2-model combinations (C(5,2) = 10)
    backbone_list = list(loaded_models.keys())
    combinations_2model = list(combinations(backbone_list, 2))
    
    print(f"\nEvaluating {len(combinations_2model)} 2-model combinations...\n")
    
    results = []
    for idx, (bb1, bb2) in enumerate(combinations_2model, 1):
        print(f"[{idx}/10] Evaluating {bb1} + {bb2}...")
        
        models = [loaded_models[bb1], loaded_models[bb2]]
        model_names = [bb1, bb2]
        
        combo_result = evaluate_ensemble_combination(models, model_names, test_ds)
        combo_result['combination'] = f"{bb1} + {bb2}"
        combo_result['index'] = idx
        
        results.append(combo_result)
        
        print(f"    Accuracy: {combo_result['accuracy']:.4f} | AUC: {combo_result['auc']:.4f}\n")
    
    # Sort by AUC (descending)
    results = sorted(results, key=lambda x: x['auc'], reverse=True)
    
    return results


def save_ensemble_results(results, output_dir):
    """Save ensemble results and top combination visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary to JSON
    summary = []
    for i, r in enumerate(results, 1):
        summary.append({
            'rank': i,
            'combination': r['combination'],
            'accuracy': round(r['accuracy'], 4),
            'precision': round(r['precision'], 4),
            'recall': round(r['recall'], 4),
            'f1': round(r['f1'], 4),
            'auc': round(r['auc'], 4)
        })
    
    summary_path = os.path.join(output_dir, 'ensemble_2model_results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {summary_path}")
    
    # Visualize top 3 combinations
    for rank in range(min(3, len(results))):
        r = results[rank]
        combo_name = r['combination'].replace(' + ', '_')
        combo_dir = os.path.join(output_dir, f"ensemble_{rank+1}_{combo_name}")
        os.makedirs(combo_dir, exist_ok=True)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_score'])
        roc_auc = r['auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Rank #{rank+1}: {r["combination"]} - ROC Curve', fontsize=14)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(combo_dir, 'roc_curve.png'), dpi=150)
        plt.close()
        
        # Confusion matrix
        cm = confusion_matrix(r['y_true'], r['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NORMAL', 'PNEUMONIA'])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        fig.suptitle(f'Rank #{rank+1}: {r["combination"]} - Confusion Matrix', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(combo_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        # Save metrics for this combination
        metrics = {
            'combination': r['combination'],
            'rank': rank + 1,
            'accuracy': round(r['accuracy'], 4),
            'precision': round(r['precision'], 4),
            'recall': round(r['recall'], 4),
            'f1': round(r['f1'], 4),
            'auc': round(r['auc'], 4)
        }
        with open(os.path.join(combo_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return summary_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
    parser.add_argument('--data_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chest_xray')))
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.models_dir, 'ensemble_results_2model')
    
    print("=" * 60)
    print("Evaluating All 2-Model Ensemble Combinations")
    print("=" * 60)
    
    results = evaluate_all_2model_ensembles(
        args.models_dir,
        args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 60)
    print("FINAL RANKINGS - 2-Model Ensembles (Top 10)")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['combination']:40s} | AUC: {r['auc']:.4f} | Acc: {r['accuracy']:.4f}")
    
    summary_path = save_ensemble_results(results, args.output_dir)
    print("\n✓ Ensemble evaluation complete!")
