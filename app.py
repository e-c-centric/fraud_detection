from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
import pandas as pd

# Load the model and scalars
model = joblib.load('model.joblib')
scalars = joblib.load('scalars.joblib')

app = Flask(__name__)

def scale_features(data, scalars):
    for column, params in scalars.items():
        if 'method' in params:  # Ensure the entry has a 'method' key
            if params['method'] == 'standardization':
                data[column] = (data[column] - params['mean']) / params['std']
            elif params['method'] == 'normalization':
                data[column] = (data[column] - params['min']) / (params['max'] - params['min'])
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    transaction_amount = request.form.get('Transaction_Amount')
    account_age = request.form.get('Account_Age')

    if not transaction_amount or not account_age:
        return render_template('index.html', error='Transaction_Amount and Account_Age are required')

    transaction_amount = float(transaction_amount)
    account_age = float(account_age)

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Transaction_Amount': [transaction_amount],
        'Account_Age': [account_age],
        'Transaction_Amount_Acc_Age': [transaction_amount * account_age]
    })

    # Apply scaling
    input_data = scale_features(input_data, scalars)

    # Convert to numpy array for prediction
    input_array = input_data[['Transaction_Amount', 'Account_Age', 'Transaction_Amount_Acc_Age']].values

    prediction = model.predict(input_array)
    confidence = model.predict_proba(input_array)

    return render_template('result.html', prediction=int(prediction[0]), confidence=confidence[0][int(prediction[0])])

@app.route('/back', methods=['POST'])
def back():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)