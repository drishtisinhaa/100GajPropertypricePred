import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from tensorflow.keras.models import load_model
from price import (
    predict_price,
    preprocess_area,
    find_best_match_area
)

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Pre-load everything once at app startup
print("Loading pipeline artifacts...")

pipeline_dict = {
    "rf_model": joblib.load("model/rf_model.pkl"),
    "dnn_model": load_model("model/dnn_model.keras"),
    "lgb_model": joblib.load("model/lgb_model.pkl"),
    "scaler": joblib.load("model/scaler.pkl"),
    "tfidf_vectorizer": joblib.load("model/tfidf.pkl"),
    "ohe_encoder": joblib.load("model/ohe_encoder.pkl"),
    "stack_model": joblib.load("model/stack_model.pkl"),
    "avg_price_lookup": joblib.load("model/avg_price_lookup.pkl"),
    "meta_ensemble": joblib.load("model/meta_ensemble.pkl"),
    "feature_names": joblib.load("model/feature_names.pkl"),
}




# Health check route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "House Price Predictor API is running!"})


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = [
            "area_name", "area", "bhk", "bathroom",
            "furnishing", "parking", "status",
            "transaction", "type"
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {missing}"
            }), 400

        # Extract fields
        area_name = data["area_name"]
        area = data["area"]
        bhk = data["bhk"]
        bathroom = data["bathroom"]
        furnishing = data["furnishing"]
        parking = data["parking"]
        status = data["status"]
        transaction = data["transaction"]
        prop_type = data["type"]

        # Derived features
        area_bhk_ratio = area / bhk if bhk else 0
        bathroom_per_bhk = bathroom / bhk if bhk else 0
        has_parking = 1 if parking > 0 else 0
        is_furnished = 1 if furnishing.lower() in ['semi-furnished', 'furnished'] else 0

        # Clean area name and find avg price
        cleaned_area = preprocess_area(area_name).lower()

        avg_area_price = find_best_match_area(
            cleaned_area,
            pipeline_dict["avg_price_lookup"],
            "avg_area_price",
            default_value=1.2e7
        )
        avg_area_price = np.clip(avg_area_price, 0, 1.3e7)

        # Form DataFrame for prediction
        new_df = pd.DataFrame({
            'area_name': [area_name],
            'area': [area],
            'bhk': [bhk],
            'bathroom': [bathroom],
            'furnishing': [furnishing],
            'parking': [parking],
            'status': [status],
            'transaction': [transaction],
            'type': [prop_type],
            'has_parking': [has_parking],
            'is_furnished': [is_furnished],
            'area_bhk_ratio': [area_bhk_ratio],
            'bathroom_per_bhk': [bathroom_per_bhk],
            'avg_area_price': [avg_area_price]
        })

        # Call pipeline
        preds = predict_price(new_df, pipeline_dict, model_type="dnn")

        result = {
            #"adjusted_price_dnn": round(float(preds[0][0])),
            #"adjusted_price_lgb": round(float(preds[1][0])),
            #"adjusted_price_rf": round(float(preds[2][0])),
            "adjusted_price_ensemble": round(float(preds[3][0])),
            #"adjusted_price_weighted": round(float(preds[4][0]))
        }

        return jsonify({
            "success": True,
            "input": data,
            "predictions": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
