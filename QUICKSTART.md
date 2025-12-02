# Quick Start Guide for Bank Customer Churn Prediction

## Option 1: Automated Setup (Recommended)

Run the automated script:
```bash
chmod +x run_project.sh
./run_project.sh
```

## Option 2: Step-by-Step Manual Setup

### 1. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset (Already Done)
The dataset has been downloaded to `dataset/Churn_Modelling.csv`

### 4. Process Data
```bash
python data_processing.py
```

This will:
- Load the raw CSV data
- Perform exploratory data analysis
- Clean and preprocess the data
- Engineer new features
- Save processed data to `dataset/processed_churn_data.parquet`

### 5. Train Models
```bash
python model_training.py
```

This will:
- Load processed data
- Train 3 models (Logistic Regression, Random Forest, GBT)
- Compare model performance
- Save all models to `models/` folder
- Save the best model for deployment

### 6. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open at: http://localhost:8501

## Features of the Streamlit App

### Single Prediction Tab
- Enter customer details manually
- Get instant churn prediction
- View risk level and probability
- Receive personalized recommendations

### Batch Prediction Tab
- Upload CSV file with customer data
- Process multiple predictions
- Download results
- View summary statistics

### Model Insights Tab
- Model performance metrics
- Key churn indicators
- Business impact analysis
- Recommended actions

## File Structure After Setup

```
Meghana_S_BDA/
├── dataset/
│   ├── Churn_Modelling.csv                  # Original dataset
│   ├── processed_churn_data.parquet/        # Processed data (Parquet)
│   └── processed_churn_data_csv/            # Processed data (CSV)
├── models/
│   ├── best_churn_model/                    # Best model
│   ├── logistic_regression_model/           # LR model
│   ├── random_forest_model/                 # RF model
│   └── gradient_boosted_trees_model/        # GBT model
├── venv/                                    # Virtual environment
├── data_processing.py                       # Data processing script
├── model_training.py                        # Model training script
├── streamlit_app.py                         # Streamlit deployment
├── pyspark_data_processing.txt              # Data processing commands
├── pyspark_model_training.txt               # Model training commands
├── download_dataset.txt                     # Dataset download commands
├── requirements.txt                         # Python dependencies
├── README.md                                # Full documentation
├── QUICKSTART.md                            # This file
└── run_project.sh                           # Automated setup script
```

## Troubleshooting

### PySpark Not Found
```bash
pip install pyspark==3.5.0
```

### Java Not Installed
```bash
sudo apt-get install openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

### Streamlit Not Found
```bash
pip install streamlit
```

### Memory Issues
Edit the Spark configuration in the Python scripts:
```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "2g") \  # Reduce from 4g
    .getOrCreate()
```

## Usage Examples

### Example 1: Process and Train from Scratch
```bash
# Activate environment
source venv/bin/activate

# Install if needed
pip install -r requirements.txt

# Process data
python data_processing.py

# Train models
python model_training.py

# Launch app
streamlit run streamlit_app.py
```

### Example 2: Just Run the App (if models exist)
```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

### Example 3: Retrain Models Only
```bash
source venv/bin/activate
python model_training.py
```

## Expected Output

### Data Processing
- Dataset statistics
- Missing value analysis
- Feature distributions
- Correlation analysis
- Processed data saved

### Model Training
- Training progress for 3 models
- Performance metrics for each
- Feature importance
- Model comparison table
- Best model saved

### Streamlit App
- Interactive web interface
- Real-time predictions
- Risk assessment
- Downloadable results

## Next Steps

After setup:
1. ✅ Explore the Streamlit app interface
2. ✅ Test single predictions with different customer profiles
3. ✅ Try batch predictions with CSV files
4. ✅ Review model insights and metrics
5. ✅ Analyze feature importance
6. ✅ Customize retention strategies

## Performance Expectations

- Data Processing: ~30-60 seconds
- Model Training: ~2-5 minutes
- Prediction (single): <1 second
- Prediction (batch 100): ~2-3 seconds

## Support

For detailed information, see:
- `README.md` - Complete documentation
- `pyspark_data_processing.txt` - All PySpark data commands
- `pyspark_model_training.txt` - All PySpark model commands

Enjoy predicting churn! 🚀
