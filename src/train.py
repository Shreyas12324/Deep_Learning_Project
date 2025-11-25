import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    # Load the preprocessed data
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    # Define model parameters
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 4

    # Create the model
    model = create_model(INPUT_SHAPE, NUM_CLASSES)

    # Define callbacks
    checkpoint = ModelCheckpoint('mediscan_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks_list)
