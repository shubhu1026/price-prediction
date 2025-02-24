import requests
import pandas as pd
import json
from datetime import datetime

url = "http://127.0.0.1:5000/predict"

data = {
    "asin": "B07V79MXF2-1",
    "name": "64 Bit USB PC Handbrake, Sim Handbrake Compatible with Logitech G27 G29 G920 T500 T300 Linear E Brake for Sim Racing Games DiRT Rally 2/4, LFS, Project CARS 2/3, Assetto Corsa , WRC 7/8/9,Forza Horizon 4/5, Fanatecosw PC Windows Hall Sensor with 78 inch USB Cable (With Clamp, Black)",
    "image_url": "https://m.media-amazon.com/images/I/61OaPM7cPVL._AC_UY218_.jpg",
    "product_url": "https://www.amazon.com/dp/B07V79MXF2",
    "rating": 4.1,
    "reviews": 1259,
    "discounted_price": 4208.33,  # This is just for informational purposes, not needed for prediction
    "listPrice": 4976.75,
    "category_id": 255,
    "price_history": [
        {'date': '2024-01-01', 'discounted_price': 2361.74, 'discount_applied': 52.54, 'category': 255},
        {'date': '2024-02-01', 'discounted_price': 4976.75, 'discount_applied': 0, 'category': 255},
        {'date': '2024-03-01', 'discounted_price': 4204.27, 'discount_applied': 15.52, 'category': 255},
        {'date': '2024-04-01', 'discounted_price': 4976.75, 'discount_applied': 0, 'category': 255},
        {'date': '2024-05-01', 'discounted_price': 4976.75, 'discount_applied': 0, 'category': 255},
        {'date': '2024-06-01', 'discounted_price': 4976.75, 'discount_applied': 0, 'category': 255},
        {'date': '2024-07-01', 'discounted_price': 2178.55, 'discount_applied': 56.23, 'category': 255},
        {'date': '2024-08-01', 'discounted_price': 2299.31, 'discount_applied': 53.8, 'category': 255},
        {'date': '2024-09-01', 'discounted_price': 4976.75, 'discount_applied': 0, 'category': 255},
        {'date': '2024-10-01', 'discounted_price': 1108.78, 'discount_applied': 77.72, 'category': 255},
        {'date': '2024-11-01', 'discounted_price': 1507.13, 'discount_applied': 69.72, 'category': 255},
        {'date': '2024-12-01', 'discounted_price': 2839.72, 'discount_applied': 42.94, 'category': 255}
    ],
    "website": "Amazon",
}


# Convert timestamps in price_history to strings
for entry in data["price_history"]:
    entry["date"] = pd.to_datetime(entry["date"]).isoformat()  # Ensures proper date format

# Send request
response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

# Print response
print(response.json())
