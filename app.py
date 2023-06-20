from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


model = pickle.load(open('diabetes.pkl', 'rb'))


scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = []
    data.append(float(request.form['a']))
    data.append(float(request.form['b']))
    data.append(float(request.form['c']))
    data.append(float(request.form['d']))
    data.append(float(request.form['e']))
    data.append(float(request.form['f']))
    data.append(float(request.form['g']))
    data.append(float(request.form['h']))

    
    input_data = pd.DataFrame([data])

    
    standardized_data = scaler.transform(input_data)

    
    pred = model.predict(standardized_data)

    if pred[0] == 0:
        pred_text = "You are not diabetic"
    else:
        pred_text = "You are diabetic"

    return render_template('after.html', data=pred_text)

if __name__ == "__main__":
    app.run(debug=True)
