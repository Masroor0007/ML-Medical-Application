from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("lung_cancer.h5")

# Load the label encoder used during training
encoder = joblib.load("encoder.pkl")

def preprocess_image(input_image):
    # Resize the input image
    resized_image = input_image.resize((512, 512))
    
    # Convert the image to an array
    image_array = np.array(resized_image)
    
    # Normalize the image array
    normalized_image = image_array / 255.0
    
    # Reshape the input image to match the model's expected input shape
    input_image = np.expand_dims(normalized_image, axis=0)
    
    return input_image

def predict_image(input_image):
    # Make predictions
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)

    # Perform inverse transformation to get the predicted class label
    # using the loaded label encoder
    predicted_class_label = encoder.inverse_transform([predicted_class_index])[0]

    return predicted_class_label

@app.route('/')
def index():
    return render_template('Lung_home.html')

@app.route('/process', methods=['POST'])
def process():
    image = request.files['image']

    try:
        # Open the uploaded image using PIL
        img = Image.open(image)
        
        # Preprocess the image
        input_image = preprocess_image(img)

        # Perform image prediction
        predicted_class = predict_image(input_image)

        # Return the result to the template
        return render_template('result.html', result=predicted_class)
    except:
        return "Error: Invalid image file"

if __name__ == '__main__':
    app.run(debug=True)
