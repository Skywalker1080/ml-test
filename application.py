from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import Ridge regressor and Scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        FFMC = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, FFMC]])
        result = ridge_model.predict(new_data_scaled)
        print(result)

        return render_template('home.html', results = result[0])

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)