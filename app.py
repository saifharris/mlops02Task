from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

# Define the endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_data = [[
        float(data['sepal_length']),
        float(data['sepal_width']),
        float(data['petal_length']),
        float(data['petal_width'])
    ]]
    df = pd.DataFrame(input_data, columns=[
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ])
    prediction = model.predict(df)[0]
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    return jsonify({'prediction': species[prediction]})

if __name__ == '__main__':
    app.run(debug=True)
