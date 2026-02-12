import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras


def evaluate_model(model_path, data_dir, img_size=(224,224), batch_size=16, out_dir='models'):
    # load model
    model = keras.models.load_model(model_path)

    # reuse data loader from src.data
    from .data import get_datasets
    train_ds, val_ds, test_ds = get_datasets(data_dir, img_size=img_size, batch_size=batch_size)

    # get predictions and labels from test set
    y_true = []
    y_scores = []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_scores.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    os.makedirs(out_dir, exist_ok=True)
    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(out_dir, 'roc.png'))
    plt.close()

    # Confusion matrix at threshold 0.5
    y_pred = (y_scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'))

    # Save metrics
    metrics = {'roc_auc': float(roc_auc)}
    with open(os.path.join(out_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(metrics, f)

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join('..','models','final_model.keras'))
    parser.add_argument('--data_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chest_xray')))
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--out_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
    args = parser.parse_args()

    m = evaluate_model(args.model, args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, out_dir=args.out_dir)
    print('Eval done:', m)
