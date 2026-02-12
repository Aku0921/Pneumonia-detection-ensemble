import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os


def get_datasets(base_dir, img_size=(224,224), batch_size=32):
    """Return (train_ds, val_ds, test_ds) created from directory layout:
    base_dir/train, base_dir/val, base_dir/test with subfolders per class.
    Includes data augmentation for training set.
    """
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    train_ds = image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )
    
    # Add data augmentation to training set
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.1),
    ])
    
    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
