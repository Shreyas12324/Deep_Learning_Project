import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Categories must match the order used during training
CATEGORIES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
IMG_SIZE = 224


def predict_image(model, image_path, top_k=3):
    """Predict an image and return top-k probabilities and predicted label.

    Returns a dict with keys:
      - predicted_label (str)
      - probabilities (list of float) : full softmax vector
      - top_k (list of (label, prob)) : top-k label-prob pairs
      - error (str) : present if something went wrong
    """
    result = {'predicted_label': None, 'probabilities': None, 'top_k': None, 'error': None}
    try:
        img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_array is None:
            result['error'] = f'Could not read image: {image_path}'
            return result

        # Convert BGR (OpenCV) to RGB (what Keras pretrained models expect)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

        # Apply VGG16 preprocessing (scaling and channel-wise mean subtraction)
        new_array = preprocess_input(new_array)

        prediction = model.predict(new_array)

        # Flatten to 1D
        probs = prediction.flatten().tolist()
        result['probabilities'] = probs

        # Top-k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_list = [(CATEGORIES[i], float(probs[i])) for i in top_indices]
        result['top_k'] = top_list

        result['predicted_label'] = CATEGORIES[int(np.argmax(probs))]
        return result
    except Exception as e:
        result['error'] = str(e)
        return result


if __name__ == '__main__':
    # Load the trained model
    model = load_model('mediscan_model.h5')

    # Example usage: replace with the path to an image you want to classify
    image_path = 'dataset/cataract/_0_4015166.jpg'
    predicted = predict_image(model, image_path)
    if predicted.get('error'):
        print('Error:', predicted['error'])
    else:
        print(f"Predicted: {predicted['predicted_label']}")
        print('Top probabilities:', predicted['top_k'])
