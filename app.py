from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from src.predict import predict_image
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model (prefer fine-tuned weights if available)
MODEL_PATHS = ['mediscan_finetuned.h5', 'mediscan_model.h5']
model = None
model_loaded_from = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        try:
            model = load_model(p)
            model_loaded_from = p
            break
        except Exception:
            # try next path
            model = None
            model_loaded_from = None

if model is None:
    raise FileNotFoundError('No valid model found. Expected one of: ' + ','.join(MODEL_PATHS))

# Try to compute global metrics from test set if available
global_metrics = None
try:
    if os.path.exists('X_test.npy') and os.path.exists('y_test.npy'):
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        # Ensure types are numeric
        preds = model.predict(X_test)
        y_pred = np.argmax(preds, axis=1)
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, target_names=['cataract','diabetic_retinopathy','glaucoma','normal'], zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Plot confusion matrix and save to static
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        classes = ['cataract','diabetic_retinopathy','glaucoma','normal']
        ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)), xticklabels=classes, yticklabels=classes,
               ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')

        # Rotate the tick labels and set alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        cm_path = os.path.join('static', 'confusion_matrix.png')
        fig.savefig(cm_path)
        plt.close(fig)

        global_metrics = {'accuracy': acc, 'report': report, 'confusion_path': 'confusion_matrix.png'}
    else:
        global_metrics = None
except Exception as e:
    # If evaluation fails, don't crash the app — keep metrics as None
    global_metrics = {'error': str(e)}


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image(model, filepath)

            # Separate values for template clarity
            if prediction.get('error'):
                predicted_label = None
                top_k = None
                probabilities = None
                error = prediction.get('error')
            else:
                predicted_label = prediction.get('predicted_label')
                top_k = prediction.get('top_k')
                probabilities = prediction.get('probabilities')
                error = None

            return render_template('index.html', predicted_label=predicted_label,
                                   top_k=top_k, probabilities=probabilities,
                                   filename=filename, global_metrics=global_metrics, error=error,
                                   model_name=os.path.basename(model_loaded_from) if model_loaded_from else 'model')
    return render_template('index.html', global_metrics=global_metrics, model_name=os.path.basename(model_loaded_from) if model_loaded_from else 'model')


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
