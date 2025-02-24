from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import os

def load_scaler(category_id):
    """
    Load the scaler for the given category_id.
    """
    scaler_path = f"scalers/{category_id}_scaler.pkl"
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

def inverse_transform_prediction(prediction, category_id):
    """
    Inverse transform the predicted price to original scale.

    Args:
        prediction (np.array): Scaled predicted values.
        category_id (int): Category ID to load the correct scaler.

    Returns:
        np.array: Unscaled predictions.
    """
    scaler = load_scaler(category_id)
    
    if scaler:
        # Create a dummy array with the same shape as `numerical_features`
        dummy_array = np.zeros((prediction.shape[0], len(numerical_features)))

        # Replace only the discounted_price column (assuming it's at index 0)
        dummy_array[:, 0] = prediction.flatten()

        # Inverse transform and extract only the first column
        return scaler.inverse_transform(dummy_array)[:, 0]
    
    return prediction.flatten()

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("best_model.keras")

# Website mapping
website_map = {"Amazon": 0, "Flipkart": 1, "Snapdeal": 2}

# Define feature lists
numerical_features = ['discounted_price', 'listPrice']
sequence_length = 3

def preprocess_input(data, month):
    """
    Preprocess input data before making a prediction.

    Args:
        data (dict): Input JSON data containing product details.
        month (int): The selected month for prediction.

    Returns:
        np.array: Preprocessed data as a numpy array.
    """
    # Convert to DataFrame
    df = pd.DataFrame([data])

    # 1️⃣ Remove last 2 characters from ASIN
    df["asin"] = df["asin"].str[:-2]

    # 2️⃣ Map website to numerical values
    df["website"] = df["website"].map(website_map).fillna(-1).astype(int)

    df.at[0, "month"] = month

    # 3️⃣ Drop unwanted columns
    df = df[['listPrice', 'website', 'month', 'category_id', 'price_history']]

    # 5️⃣ Extract last 3 months' discounted_price from price_history
    price_history = df["price_history"].iloc[0]  # List of dictionaries

    # Convert price history into a DataFrame
    price_df = pd.DataFrame(price_history)

    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")

    # Ensure that price_df has a 'month' column extracted from the 'date' field
    price_df["month"] = price_df["date"].dt.month

    
    df["discounted_price"] = df["month"].map(price_df.set_index("month")["discounted_price"])

    df = df[['discounted_price', 'listPrice', 'website', 'month', 'category_id', 'price_history']]

    # 4️⃣ Scale numerical features using category-wise scaler
    category_id = df["category_id"].values[0]
    scaler = load_scaler(category_id)
    if scaler:
        df[numerical_features] = scaler.transform(df[numerical_features])
    else:
        return None, "Unknown category_id"

    # Get last 3 months from the given month (handling wrap-around for January)
    last_3_months = [(month - i) if (month - i) > 0 else (month - i + 12) for i in range(2, -1, -1)]

    # Extract discounted prices for the last 3 months
    selected_prices = []
    for m in last_3_months:
        row = price_df[price_df["month"] == m]
        if not row.empty:
            selected_prices.append(row["discounted_price"].values[0])
        else:
            selected_prices.append(0)  # If no data, fill with 0

    selected_prices_scaled = []

    if scaler:
        dummy_array = np.zeros((len(selected_prices), 2))
        dummy_array[:, 0] = selected_prices
        # Apply the scaler to this dummy array
        selected_prices_scaled = scaler.transform(dummy_array)[:, 0]
    else:
        selected_prices_scaled = selected_prices

    # 5️⃣ Create sequence data in required format
    sequences = []
    for i in range(sequence_length):
        row = [
            selected_prices_scaled[i],  # discounted_price for that month
            df["listPrice"].values[0],
            df["website"].values[0],
            last_3_months[i],  # month
            df["category_id"].values[0],
        ]
        sequences.append(row)

    # Convert to NumPy array
    sequence_data = np.array(sequences)

    return np.expand_dims(sequence_data, axis=0), None  # Adds batch dimension

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    month = data.get("month", 1)

    print(month)

    # Validate input
    if "category_id" not in data:
        return jsonify({"error": "Missing category_id"}), 400

    predictions = []

    # Initial prediction
    processed_data, error = preprocess_input(data, month)
    if error:
        return jsonify({"error": error}), 400

    prediction = model.predict(processed_data)
    unscaled_prediction = inverse_transform_prediction(prediction, data["category_id"])
    predictions.append(unscaled_prediction.tolist())

    # Predict for next 5 months
    for i in range(5):
        data['discounted_price'] = unscaled_prediction.tolist()
        month = (month % 12) + 1  # Ensure month stays within 1-12
        processed_data, error = preprocess_input(data, month)
        if error:
            return jsonify({"error": error}), 400

        prediction = model.predict(processed_data)
        unscaled_prediction = inverse_transform_prediction(prediction, data["category_id"])
        predictions.append(unscaled_prediction.tolist())

    return jsonify({"predictions": predictions})

port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
