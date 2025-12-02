#!/bin/bash
# Quick Deploy - Just run Streamlit app
# Assumes models are already trained

echo "=========================================="
echo "Bank Customer Churn Prediction"
echo "Quick Deploy - Streamlit App"
echo "=========================================="
echo ""

# Activate environment
source venv/bin/activate

# Check if models exist
if [ -d "models/best_churn_model" ] || [ -d "models/best_churn_model_v2" ]; then
    echo "✓ Models found!"
    echo ""
    echo "Launching Streamlit app..."
    echo "Access at: http://localhost:8501"
    echo ""
    streamlit run streamlit_app.py
else
    echo "✗ No models found!"
    echo ""
    echo "Please run one of these first:"
    echo "  python data_processing.py && python model_training.py  (V1)"
    echo "  python data_processing_v2.py && python model_training_v2.py  (V2)"
    echo ""
    echo "Or run the full deployment:"
    echo "  ./deploy_full.sh"
fi
