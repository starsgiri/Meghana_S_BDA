#!/bin/bash
# Quick Start Script for Bank Customer Churn Prediction Project

echo "=========================================="
echo "Bank Customer Churn Prediction - Setup"
echo "=========================================="

# Step 1: Activate virtual environment
echo ""
echo "Step 1: Activating virtual environment..."
source venv/bin/activate

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
pip install -r requirements.txt

# Step 3: Download dataset (if not already downloaded)
echo ""
echo "Step 3: Checking dataset..."
if [ ! -f "dataset/Churn_Modelling.csv" ]; then
    echo "Dataset not found. Downloading from Kaggle..."
    echo "Note: Make sure you have configured Kaggle API credentials"
    kaggle datasets download -d shantanudhakadd/bank-customer-churn-prediction -p ./dataset --unzip
else
    echo "Dataset already exists!"
fi

# Step 4: Run data processing
echo ""
echo "Step 4: Running data processing..."
python data_processing.py

# Step 5: Run model training
echo ""
echo "Step 5: Running model training..."
python model_training.py

# Step 6: Launch Streamlit app
echo ""
echo "Step 6: Launching Streamlit application..."
echo "The app will open in your browser at http://localhost:8501"
streamlit run streamlit_app.py
