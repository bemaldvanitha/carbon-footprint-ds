from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
loaded_model = load_model('carbon_footprint_v2.h5')

# Initialize Flask app
app = Flask(__name__)


# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        input_data = np.array([list(data.values())])

        # Make predictions
        predictions = loaded_model.predict(input_data)

        print(predictions)

        # Convert predictions to a readable format
        prediction_result = predictions[0][0]

        # Return the prediction as JSON
        return jsonify({'prediction': float(prediction_result)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)