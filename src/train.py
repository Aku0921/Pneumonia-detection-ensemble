import os
import json
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .data import get_datasets
from .models import create_model


def build_model(backbone='efficientnetb0', input_shape=(224, 224, 3), dropout=0.2):
    # delegate to models.create_model which returns (model, base)
    model, base = create_model(backbone_name=backbone, input_shape=input_shape, dropout=dropout, base_trainable=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model, base


def get_class_weights(train_ds):
    """Calculate class weights to handle imbalance."""
    # Count samples per class
    class_counts = {0: 0, 1: 0}
    for _, labels in train_ds:
        labels_flat = tf.reshape(labels, [-1])
        for label_val in labels_flat.numpy():
            class_counts[int(label_val)] += 1
    
    total = sum(class_counts.values())
    if total == 0:
        return {0: 1.0, 1: 1.0}
    weights = {0: total / (2 * max(class_counts[0], 1)), 1: total / (2 * max(class_counts[1], 1))}
    print(f"Class weights: {weights}")
    return weights


def unfreeze_and_compile(model, base, lr=1e-5, unfreeze_at=None):
    if unfreeze_at is None:
        base.trainable = True
    else:
        for layer in base.layers[:unfreeze_at]:
            layer.trainable = False
        for layer in base.layers[unfreeze_at:]:
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chest_xray')))
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--backbone', type=str, default='efficientnetb0', help='backbone to use: efficientnetb0|mobilenetv2|resnet50v2|densenet121|vgg16')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--top_epochs', type=int, default=10, help='Train head for N epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Then unfreeze and fine-tune for M epochs')
    parser.add_argument('--unfreeze_at', type=int, default=None, help='Layer index to start unfreezing (None = all)')
    parser.add_argument('--output_dir', default=None, help='Output directory for model (auto-created if None)')
    args = parser.parse_args(argv)

    # Auto-create output directory with backbone name if not specified
    if args.output_dir is None:
        args.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', args.backbone))
    
    train_ds, val_ds, test_ds = get_datasets(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size)

    print(f"Training {args.backbone} backbone...")
    print(f"Output directory: {args.output_dir}")
    
    model, base = build_model(backbone=args.backbone, input_shape=(args.img_size, args.img_size, 3))

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'best_model.keras')

    # Calculate class weights
    class_weights = get_class_weights(train_ds)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_auc', save_best_only=True, mode='max'),
        keras.callbacks.EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.TensorBoard(log_dir=os.path.join(args.output_dir, 'logs'))
    ]

    # Train head
    print(f'Training head for {args.top_epochs} epochs...')
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.top_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Fine-tune
    history2 = None
    if args.fine_tune_epochs > 0:
        print(f'Unfreezing base and compiling at lower LR...')
        model = unfreeze_and_compile(model, base, lr=1e-5, unfreeze_at=args.unfreeze_at)
        print(f'Fine-tuning for {args.fine_tune_epochs} epochs...')
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.top_epochs + args.fine_tune_epochs,
            initial_epoch=args.top_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

    # Save final model in native Keras format
    final_model_path = os.path.join(args.output_dir, 'final_model.keras')
    model.save(final_model_path)
    print(f'Final model saved to: {final_model_path}')

    # Save history
    history = {'head': history1.history, 'fine_tune': history2.history if history2 is not None else {}}
    history_path = os.path.join(args.output_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f'Training history saved to: {history_path}')

    # Evaluate on test set
    print('Evaluating on test set...')
    res = model.evaluate(test_ds, verbose=0)
    print(f'Test Loss: {res[0]:.4f}')
    print(f'Test Accuracy: {res[1]:.4f}')
    print(f'Test AUC: {res[2]:.4f}')
    
    # Save test metrics
    metrics = {
        'backbone': args.backbone,
        'test_loss': float(res[0]),
        'test_accuracy': float(res[1]),
        'test_auc': float(res[2])
    }
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Test metrics saved to: {metrics_path}')


if __name__ == '__main__':
    main()
