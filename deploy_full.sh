#!/bin/bash
# Full Deployment Script - Bank Customer Churn Prediction
# Runs complete pipeline: Data Processing V1 → Model Training V1 → Data Processing V2 → Model Training V2 → Streamlit

echo "=========================================="
echo "BANK CUSTOMER CHURN PREDICTION"
echo "Full Deployment Pipeline"
echo "=========================================="
echo ""

# Activate virtual environment
echo "Step 1: Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo ""
echo "Step 2: Checking dependencies..."
pip list | grep -q pyspark || pip install pyspark -q
pip list | grep -q streamlit || pip install streamlit -q
pip list | grep -q pandas || pip install pandas numpy -q

echo "✓ All dependencies installed"

# Step 3: Process V1 Data (10K records)
echo ""
echo "=========================================="
echo "Step 3: Processing V1 Data (10,000 records)"
echo "=========================================="
python3 data_processing.py
if [ $? -eq 0 ]; then
    echo "✓ V1 data processing complete"
else
    echo "✗ V1 data processing failed"
    exit 1
fi

# Step 4: Train V1 Models
echo ""
echo "=========================================="
echo "Step 4: Training V1 Models"
echo "=========================================="
python3 model_training.py
if [ $? -eq 0 ]; then
    echo "✓ V1 model training complete"
else
    echo "✗ V1 model training failed"
    exit 1
fi

# Step 5: Process V2 Data (500K records)
echo ""
echo "=========================================="
echo "Step 5: Processing V2 Data (500,000 records)"
echo "=========================================="
python3 data_processing_v2.py
if [ $? -eq 0 ]; then
    echo "✓ V2 data processing complete"
else
    echo "✗ V2 data processing failed"
    exit 1
fi

# Step 6: Train V2 Models
echo ""
echo "=========================================="
echo "Step 6: Training V2 Models"
echo "=========================================="
python3 model_training_v2.py
if [ $? -eq 0 ]; then
    echo "✓ V2 model training complete"
else
    echo "✗ V2 model training failed"
    exit 1
fi

# Step 7: Launch Streamlit App
echo ""
echo "=========================================="
echo "Step 7: Launching Streamlit Application"
echo "=========================================="
echo ""
echo "✓ All processing and training complete!"
echo ""
echo "Starting Streamlit app..."
echo "Access the app at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run streamlit_app.py
