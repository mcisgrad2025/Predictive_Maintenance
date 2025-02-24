# Import Flask and other tools
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model from your project folder
model = joblib.load('model.pkl')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the user
    data = request.json

    # Convert the data into a DataFrame (format the model expects)
    input_data = pd.DataFrame([data])

    # Make a prediction
    prediction = model.predict(input_data)

    # Return the result (0 = No failure, 1 = Failure)
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)