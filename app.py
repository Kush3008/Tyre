from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import base64

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define class names
class_names = ['class_1', 'class_2', 'class_3']

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['file']
        
        # Read the image file
        img_bytes = file.read()
        
        # Convert the bytes into an image
        img = Image.open(io.BytesIO(img_bytes)).resize((128, 128))  # Resize the image to match model input size
        
        # Convert image to base64 string
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Make prediction
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        
        # Get the actual class name from the file name or any other method
        # For demonstration purposes, let's assume actual class index is 0
        actual_class_index = 0
        actual_class = class_names[actual_class_index]
        
        # Pass the input image base64 string, actual class name, and predicted class name to the template
        return render_template('result.html', input_image=img_base64, actual_class=actual_class, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
