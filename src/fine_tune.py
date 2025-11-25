import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input


def preprocess_arrays(X):
    # Convert BGR->RGB if needed (data_preprocessing used OpenCV)
    Xp = X[..., ::-1]
    Xp = Xp.astype('float32')
    Xp = preprocess_input(Xp)
    return Xp


def fine_tune(model_path='mediscan_model.h5', out_path='mediscan_finetuned.h5', epochs=3, batch_size=16):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    print('Loading model...')
    model = load_model(model_path)

    # Freeze most layers, unfreeze last N layers for fine-tuning
    N_unfreeze = 10
    for layer in model.layers[:-N_unfreeze]:
        layer.trainable = False
    for layer in model.layers[-N_unfreeze:]:
        layer.trainable = True

    # Compile with a low learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('Loading data arrays...')
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    print('Preprocessing arrays for VGG16...')
    X_train_p = preprocess_arrays(X_train)
    X_test_p = preprocess_arrays(X_test)

    # Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow(X_train_p, y_train, batch_size=batch_size)

    # Callbacks
    checkpoint = ModelCheckpoint(out_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)

    print('Starting fine-tuning...')
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(X_train_p) // batch_size),
        epochs=epochs,
        validation_data=(X_test_p, y_test),
        callbacks=[checkpoint, reduce_lr]
    )

    print('Fine-tuning complete. Best weights saved to', out_path)


if __name__ == '__main__':
    # Short run for quick improvement - adjust epochs as needed
    fine_tune(epochs=3, batch_size=16)
