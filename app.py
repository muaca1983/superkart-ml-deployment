
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "best_superkart_rf_model.pkl"
model = joblib.load(MODEL_PATH)

EXPECTED_FEATURES = [
    'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area', 
    'Product_MRP', 'Store_Size', 'Store_Location_City_Type', 
    'Store_Type', 'Product_Id_char', 'Store_Age_Years', 'Product_Type_Category'
]

# MASTER CATEGORY LISTS FROM YOUR DATASET
# This ensures single-row requests match your notebook's OneHotEncoder(drop='first') layout perfectly.
CATEGORIES = {
    'Product_Sugar_Content': ['Low Sugar', 'No Sugar', 'Regular'],
    'Store_Size': ['High', 'Medium', 'Small'],
    'Store_Location_City_Type': ['Tier 1', 'Tier 2', 'Tier 3'],
    'Store_Type': ['Departmental Store', 'Food Mart', 'Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'],
    'Product_Id_char': ['DR', 'FD', 'NC'],
    'Product_Type_Category': ['Non Perishables', 'Others', 'Perishables']
}

def map_product_category(item_type):
    perishables = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Breakfast', 'Breads', 'Seafood']
    non_perishables = ['Baking Goods', 'Frozen Foods', 'Canned', 'Hard Drinks', 'Soft Drinks', 'Health and Hygiene', 'Household', 'Others', 'Snack Foods', 'Starchy Foods']
    if item_type in perishables:
        return 'Perishables'
    elif item_type in non_perishables:
        return 'Non Perishables'
    else:
        return 'Others'

def preprocess_payload(df):
    if 'Product_Sugar_Content' in df.columns:
        df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace(
            {'reg': 'Regular', 'low sugar': 'Low Sugar', 'Low Sugar': 'Low Sugar'}
        )
        
    df['Product_Id_char'] = df['Product_Id'].str[:2] if 'Product_Id' in df.columns else "FD"
    df['Store_Age_Years'] = 2025 - df['Store_Establishment_Year'] if 'Store_Establishment_Year' in df.columns else 10
    df['Product_Type_Category'] = df['Product_Type'].apply(map_product_category) if 'Product_Type' in df.columns else 'Others'
        
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = None
            
    df = df[EXPECTED_FEATURES]
    
    #Force categories to use the strict master lists
    for col, cat_list in CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=cat_list)
        
    return df

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "message": "SuperKart Production API Ready!"}), 200

@app.route("/v1/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Missing request payload body"}), 400
            
        df_input = pd.DataFrame([data])
        df_input = preprocess_payload(df_input)

        prediction = model.predict(df_input)
        return jsonify({
            "status": "success",
            "predicted_sales": float(round(prediction[0], 2))
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
