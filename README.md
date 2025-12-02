# Bank Customer Churn Prediction using PySpark MLlib

## Project Overview
This project builds a machine learning model to predict bank customer churn using PySpark MLlib. It helps banks identify customers at risk of closing their accounts and provides insights into churn reasons.

## Features
- 📊 Data processing with PySpark
- 🤖 Multiple ML algorithms (Logistic Regression, Random Forest, Gradient Boosted Trees)
- 📈 Model comparison and selection
- 🎯 Feature importance analysis
- 🌐 Interactive Streamlit web application for predictions
- 📁 Batch prediction capability

## Project Structure
```
Meghana_S_BDA/
├── dataset/                          # Dataset folder
│   └── Churn_Modelling.csv          # Raw dataset from Kaggle
├── models/                          # Trained models
│   ├── best_churn_model/           # Best performing model
│   ├── logistic_regression_model/  # Logistic Regression model
│   ├── random_forest_model/        # Random Forest model
│   └── gradient_boosted_trees_model/ # GBT model
├── download_dataset.txt            # Kaggle download commands
├── pyspark_data_processing.txt     # Data processing code
├── pyspark_model_training.txt      # Model training code
├── streamlit_app.py                # Streamlit deployment app
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Dataset
The dataset is downloaded from Kaggle: [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

**Features:**
- CreditScore: Customer credit score
- Geography: Customer location (France, Germany, Spain)
- Gender: Male/Female
- Age: Customer age
- Tenure: Years with the bank
- Balance: Account balance
- NumOfProducts: Number of bank products used
- HasCrCard: Whether customer has a credit card (0/1)
- IsActiveMember: Whether customer is active (0/1)
- EstimatedSalary: Customer's estimated salary
- Exited: Target variable (1 = churned, 0 = retained)

## Installation & Setup

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle API
```bash
# Download your kaggle.json from https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Download Dataset
```bash
kaggle datasets download -d shantanudhakadd/bank-customer-churn-prediction -p ./dataset --unzip
```

## Usage

### Step 1: Data Processing
Run the PySpark data processing commands:
```bash
# Copy the code from pyspark_data_processing.txt and run in Python
python -c "$(cat pyspark_data_processing.txt)"
```

Or create a Python file:
```bash
cp pyspark_data_processing.txt data_processing.py
python data_processing.py
```

### Step 2: Model Training
Run the model training code:
```bash
# Copy the code from pyspark_model_training.txt and run
cp pyspark_model_training.txt model_training.py
python model_training.py
```

This will:
- Train three different models (Logistic Regression, Random Forest, GBT)
- Compare their performance
- Save the best model to `models/best_churn_model`

### Step 3: Run Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Streamlit Application Features

### 1. Single Prediction
- Enter individual customer details
- Get instant churn prediction
- View churn probability and risk level
- Receive personalized recommendations

### 2. Batch Prediction
- Upload CSV file with multiple customers
- Get predictions for all customers
- Download results as CSV
- View summary statistics

### 3. Model Insights
- View model performance metrics
- Understand key churn indicators
- Learn about business impact
- Get recommended actions

## Model Performance

The models are evaluated on multiple metrics:
- **Accuracy**: Overall prediction accuracy
- **AUC-ROC**: Area under ROC curve (discrimination ability)
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual churners
- **F1-Score**: Harmonic mean of precision and recall

Typical performance:
- Accuracy: ~85-87%
- AUC-ROC: ~0.84-0.86
- Precision: ~70-75%
- Recall: ~60-65%

## Key Findings

### Top Churn Indicators:
1. **Inactive membership** - Strongest predictor
2. **Number of products** - Single-product customers more likely to churn
3. **Age** - Older customers show higher churn rates
4. **Geography** - Location-based differences
5. **Balance** - Both zero and very high balances indicate risk

### Business Recommendations:
- Target inactive members with engagement campaigns
- Offer additional products to single-product customers
- Provide age-specific retention programs
- Monitor customers with unusual balance patterns
- Implement early warning system for high-risk customers

## Technologies Used
- **PySpark**: Distributed data processing and ML
- **PySpark MLlib**: Machine learning library
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Kaggle API**: Dataset download

## Files Description

### download_dataset.txt
Contains commands to download the dataset from Kaggle using the Kaggle API.

### pyspark_data_processing.txt
Complete PySpark code for:
- Loading data
- Exploratory data analysis
- Data cleaning
- Feature engineering
- Data preprocessing
- Saving processed data

### pyspark_model_training.txt
Complete PySpark MLlib code for:
- Feature preparation
- Multiple model training
- Model evaluation and comparison
- Feature importance analysis
- Model saving
- Churn analysis and insights

### streamlit_app.py
Interactive web application for:
- Single customer prediction
- Batch predictions
- Model insights and metrics
- Risk assessment
- Actionable recommendations

## Running Individual Commands

All PySpark commands are saved in text files. You can either:

1. **Copy-paste sections** into a Python interpreter or Jupyter notebook
2. **Convert to Python files** and run them
3. **Use pyspark shell** for interactive execution

Example:
```bash
# Start PySpark shell
pyspark

# Then copy-paste commands from the .txt files
```

## Troubleshooting

### Java/Spark Issues
If you encounter Java-related errors:
```bash
# Install Java (if needed)
sudo apt-get install openjdk-11-jdk

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

### Memory Issues
If Spark runs out of memory, adjust the configuration:
```python
spark = SparkSession.builder \
    .appName("ChurnPrediction") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```

## Future Enhancements
- Real-time prediction API
- Advanced feature engineering
- Deep learning models
- A/B testing framework
- Customer segmentation
- Automated model retraining
- Integration with CRM systems

## Author
Meghana S - BDA Project

## License
This project is for educational purposes.

## Acknowledgments
- Dataset: Kaggle Bank Customer Churn Prediction
- Framework: Apache Spark MLlib
- Deployment: Streamlit

---

For questions or issues, please review the code comments in the .txt files or check Spark documentation.
