from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load("RandomForestClassifier.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """ Renders the HTML form for input """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type == "application/json":
            # JSON-based API request
            data = request.get_json()
        else:
            # Form-based request from HTML
            data = request.form

        # Extract features from the input
        features = np.array([
            float(data["ph"]),
            float(data["Hardness"]),
            float(data["Solids"]),
            float(data["Chloramines"]),
            float(data["Sulfate"]),
            float(data["Conductivity"]),
            float(data["Organic_Carbon"]),
            float(data["Trihalomethanes"]),
            float(data["Turbidity"])
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        result = "POTABLE (Safe to Drink)" if prediction == 1 else "NOT POTABLE (Unsafe to Drink)"
        
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Return error with HTTP 400 status

if __name__ == '__main__':
    app.run(debug=True)
