


import seaborn as sns
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import json
import os

# 🎯 Graph generate karne ke liye library imports
import matplotlib
matplotlib.use('Agg')  # FastAPI background running ke liye zaroori hai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder checking jahan graph save hoga
GRAPH_DIR = os.path.join("static", "saved_model")
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

app.mount("/static", StaticFiles(directory="static"), name="static")

model = joblib.load('saved_model/final_simplest_model.joblib')
product_mapping = joblib.load('saved_model/product_mapping.joblib')


named_products = {
    "22730": {"name": "Wireless Bluetooth Headphones", "price": 3.75},
    "23254": {"name": "Ergonomic Mechanical Keyboard", "price": 4.15},
    "22556": {"name": "UltraHD Action Camera 4K", "price": 1.65},
    "22726": {"name": "Smart Fitness Watch v2", "price": 3.75},
    "22138": {"name": "Premium Leather Wallet", "price": 4.95},
    "85123A": {"name": "Vintage Round Sunglasses", "price": 2.55},
    "84879": {"name": "Noise Cancelling Earbuds", "price": 1.69},
    "23084": {"name": "Minimalist Waterproof Backpack", "price": 2.08},
    "22197": {"name": "Stainless Steel Thermos", "price": 0.85},
    "20725": {"name": "Cotton Comfort Hoodie", "price": 1.65},
    "23203": {"name": "Anti-Slip Exercise Yoga Mat", "price": 2.08},
    "21212": {"name": "LED RGB Desk Lamp", "price": 0.55},
    "22423": {"name": "Professional Running Shoes", "price": 12.75},
    "22666": {"name": "Portable Power Bank 20000mAh", "price": 2.95},
    "22961": {"name": "Gourmet Dark Chocolates Pack", "price": 1.45}
}

# Live tracking ke liye history lists
history_actual = []
history_predicted = []


class InputData(BaseModel):
    ProductNo: str
    Quantity: int

# 🎯 ADVANCED GRAPH GENERATOR FUNCTION


def generate_advanced_plots():
    if len(history_actual) < 2:
        return  # Kam se kam 2 data points chahiye graph banane ke liye

    plt.figure(figsize=(10, 4))

    # Left Plot: Errors/Residual Distribution Analysis
    plt.subplot(1, 2, 1)
    residuals = [act - pred for act,
                 pred in zip(history_actual, history_predicted)]
    sns.histplot(residuals, kde=True, color='purple', bins=10)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Errors Distribution (Residual Analysis)')
    plt.xlabel('Residual Error')

    # Right Plot: Sales Density Comparison: Actual vs Predicted
    plt.subplot(1, 2, 2)
    sns.kdeplot(history_actual, label='Actual Sales', fill=True, color='blue')
    sns.kdeplot(history_predicted, label='Predicted Sales',
                fill=True, color='green')
    plt.title('Sales Density Comparison: Actual vs Predicted')
    plt.xlabel('Revenue Amount')
    plt.legend()

    plt.tight_layout()
    graph_path = os.path.join(GRAPH_DIR, "saved_advanced_model_graphs.png")
    plt.savefig(graph_path)
    plt.close()


@app.post("/predict")
def predict_sales(data: InputData):
    encoded_product = product_mapping.get(str(data.ProductNo), 0)
    product_info = named_products.get(str(data.ProductNo), {"price": 1.0})
    unit_price = product_info["price"]

    input_df = pd.DataFrame([{
        'ProductNo_Encoded': encoded_product,
        'Quantity': data.Quantity,
        'UnitPrice': unit_price
    }])

    prediction = model.predict(input_df)
    final_number = float(prediction[0])

    # Mathematical True Value (Actual Revenue)
    actual_revenue = unit_price * data.Quantity

    # History data save karna live graph plot ke liye
    history_actual.append(actual_revenue)
    history_predicted.append(final_number)

    # 🎯 Backend par advanced statistical graph ko live refresh karne ka call
    generate_advanced_plots()

    return {"Predicted_Sales": round(final_number, 2)}


@app.get("/", response_class=HTMLResponse)
def get_ui():
    products_json = json.dumps(named_products)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart E-Commerce Sales Forecasting Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #f4f6f9; color: #333; }}
            .container {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0px 4px 15px rgba(0,0,0,0.05); max-width: 550px; margin: auto; text-align: left; }}
            h2 {{ text-align: center; color: #2c3e50; margin-bottom: 20px; }}
            label {{ font-weight: bold; display: block; margin-top: 15px; color: #555; }}
            select, input {{ width: 100%; padding: 12px; margin-top: 5px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 6px; font-size: 15px; }}
            button {{ background: #28a745; color: white; border: none; padding: 14px; font-size: 16px; border-radius: 6px; cursor: pointer; width: 100%; margin-top: 20px; font-weight: bold; transition: 0.2s; }}
            button:hover {{ background: #218838; }}
            #result {{ margin-top: 25px; padding: 15px; background: #e8f4fd; border-radius: 6px; text-align: center; font-size: 22px; font-weight: bold; color: #0056b3; display: none; }}
            .graph-box {{ margin-top: 40px; text-align: center; }}
            img {{ max-width: 95%; border-radius: 12px; box-shadow: 0px 4px 15px rgba(0,0,0,0.08); margin-top: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🛍️ Smart E-Commerce Sales Predictor</h2>
            
            <label for="productSelect">Choose a Product to Order:</label>
            <select id="productSelect" onchange="updatePrice()">
                <option value="">-- Click to Expand Product List (15 Real Items) --</option>
            </select>

            <label for="price">Unit Price ($):</label>
            <input type="number" id="price" readonly style="background-color: #e9ecef; cursor: not-allowed;">

            <label for="qty">Enter Quantity to Purchase:</label>
            <input type="number" id="qty" placeholder="e.g. 5" min="1">

            <button onclick="makePrediction()">Predict Order Revenue</button>
            <div id="result"></div>
        </div>

        <div class="graph-box">
            <h3>📊 Backend Model Performance Visual Analytics</h3>
            <!-- 🎯 ID added to image tag so JavaScript can live update it -->
            <img id="dashboardGraph" src="/static/saved_model/saved_advanced_model_graphs.png" alt="Advanced Analytics Dashboard" onerror="this.style.display='none'">
        </div>

        <script>
            const productData = {products_json};
            const selectEl = document.getElementById('productSelect');
            
            for (const [id, info] of Object.entries(productData)) {{
                let opt = document.createElement('option');
                opt.value = id;
                opt.innerHTML = info.name + " (Code: " + id + ")";
                selectEl.appendChild(opt);
            }}

            function updatePrice() {{
                const selectedProd = selectEl.value;
                if(selectedProd) {{
                    document.getElementById('price').value = productData[selectedProd].price;
                }} else {{
                    document.getElementById('price').value = '';
                }}
            }}

            async function makePrediction() {{
                const productNo = selectEl.value;
                const qty = document.getElementById('qty').value;
                const resultDiv = document.getElementById('result');
                const graphImg = document.getElementById('dashboardGraph');
                
                if(!productNo || !qty) {{ alert("Pehle Product aur Quantity select karein!"); return; }}

                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ 
                            ProductNo: productNo,
                            Quantity: parseInt(qty) 
                        }})
                    }});
                    
                    const result = await response.json();
                    if(result.Predicted_Sales !== undefined) {{
                        resultDiv.style.display = 'block';
                        resultDiv.innerHTML = "💰 Predicted Order Revenue: <b>$" + result.Predicted_Sales + "</b>";
                        
                        // 🎯 LIVE REFRESH GRAPH: Image cache break karne ke liye timestamp lagaya hai
                        graphImg.style.display = "inline-block";
                        graphImg.src = "/static/saved_model/saved_advanced_model_graphs.png?t=" + new Date().getTime();
                        
                    }} else {{
                        alert("Backend se galat response aaya!");
                    }}
                }} catch (error) {{
                    alert("Server connect nahi ho paa raha!");
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html_content




