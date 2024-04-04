from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\akush\\Downloads\\Tyre\\model.h5')

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
        img = tf.keras.preprocessing.image.load_img(io.BytesIO(img_bytes), target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]

        return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
