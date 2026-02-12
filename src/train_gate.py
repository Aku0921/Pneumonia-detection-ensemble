"""Train a lightweight X-ray gate classifier (X-ray vs non-X-ray)."""
import os
import json
import glob
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


VALID_EXTS = {".jpg", ".jpeg", ".png", ".gif"}


def collect_paths(root_dir, patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    filtered = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in VALID_EXTS:
            filtered.append(p)
    return filtered


def load_image(path, img_size):
    data = tf.io.read_file(path)
    img = tf.image.decode_image(data, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, img_size)
    return img


def build_dataset(paths, labels, img_size, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_image(p, img_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    x = base(x, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output_dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "xray_gate")))
    args = parser.parse_args(argv)

    xray_root = os.path.join(args.data_dir, "chest_xray")
    non_xray_root = os.path.join(args.data_dir, "non_xray")

    xray_paths = collect_paths(xray_root, ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.gif"])
    non_xray_paths = collect_paths(non_xray_root, ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.gif"])

    if not xray_paths:
        raise RuntimeError("No X-ray images found in data/chest_xray")
    if not non_xray_paths:
        raise RuntimeError("No non-X-ray images found in data/non_xray")

    labels = [1] * len(xray_paths) + [0] * len(non_xray_paths)
    paths = xray_paths + non_xray_paths

    combined = list(zip(paths, labels))
    random.shuffle(combined)
    paths, labels = zip(*combined)
    paths = list(paths)
    labels = list(labels)

    split_idx = int(0.8 * len(paths))
    train_paths, val_paths = paths[:split_idx], paths[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    img_size = (args.img_size, args.img_size)
    train_ds = build_dataset(train_paths, train_labels, img_size, args.batch_size, shuffle=True)
    val_ds = build_dataset(val_paths, val_labels, img_size, args.batch_size, shuffle=False)

    model = build_model((args.img_size, args.img_size, 3))
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "best_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_auc", save_best_only=True, mode="max"),
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=3, mode="max", restore_best_weights=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = os.path.join(args.output_dir, "final_model.keras")
    model.save(final_model_path)

    metrics = model.evaluate(val_ds, verbose=0)
    metrics_out = {
        "val_loss": float(metrics[0]),
        "val_accuracy": float(metrics[1]),
        "val_auc": float(metrics[2]),
        "xray_count": len(xray_paths),
        "non_xray_count": len(non_xray_paths),
    }

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("Gate model saved to:", final_model_path)
    print("Validation metrics:", metrics_out)


if __name__ == "__main__":
    main()
