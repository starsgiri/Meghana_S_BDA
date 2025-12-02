import streamlit as st
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import *
from pyspark.sql.functions import col
import os

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Initialize Spark Session
@st.cache_resource
def init_spark():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("ChurnPredictionApp") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    return spark

@st.cache_resource
def load_model(_spark):
    """Load the trained model"""
    try:
        model_path = "models/best_churn_model"
        if os.path.exists(model_path):
            model = PipelineModel.load(model_path)
            return model, "best_churn_model"
        else:
            # Try loading Random Forest model as alternative
            model_path = "models/random_forest_model"
            if os.path.exists(model_path):
                model = PipelineModel.load(model_path)
                return model, "random_forest_model"
            else:
                return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_churn(spark, model, customer_data):
    """Make churn prediction for a customer"""
    try:
        # Create DataFrame from customer data
        schema = StructType([
            StructField("CreditScore", IntegerType(), True),
            StructField("Geography", StringType(), True),
            StructField("Gender", StringType(), True),
            StructField("Age", IntegerType(), True),
            StructField("Tenure", IntegerType(), True),
            StructField("Balance", DoubleType(), True),
            StructField("NumOfProducts", IntegerType(), True),
            StructField("HasCrCard", IntegerType(), True),
            StructField("IsActiveMember", IntegerType(), True),
            StructField("EstimatedSalary", DoubleType(), True)
        ])
        
        # Create Spark DataFrame
        df = spark.createDataFrame([customer_data], schema=schema)
        
        # Add engineered features
        df = df.withColumn("AgeGroup", 
            col("Age").cast("string"))  # Simplified for demo
        df = df.withColumn("BalanceCategory", 
            col("Balance").cast("string"))  # Simplified for demo
        df = df.withColumn("CreditScoreCategory", 
            col("CreditScore").cast("string"))  # Simplified for demo
        
        # Make prediction
        prediction = model.transform(df)
        
        # Extract prediction and probability
        result = prediction.select("prediction", "probability").collect()[0]
        churn_prediction = int(result["prediction"])
        probability = result["probability"].toArray()
        
        return churn_prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Main App
def main():
    # Header
    st.title("🏦 Bank Customer Churn Prediction System")
    st.markdown("### Predict customer churn using PySpark MLlib")
    st.markdown("---")
    
    # Initialize Spark
    spark = init_spark()
    
    # Load model
    model, model_name = load_model(spark)
    
    if model is None:
        st.error("❌ Model not found! Please train the model first using the PySpark training script.")
        st.info("Run the commands in 'pyspark_model_training.txt' to train the model.")
        return
    
    st.success(f"✅ Model loaded successfully: {model_name}")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Model Insights"])
    
    if page == "Single Prediction":
        show_single_prediction(spark, model)
    elif page == "Batch Prediction":
        show_batch_prediction(spark, model)
    else:
        show_model_insights()

def show_single_prediction(spark, model):
    """Single customer prediction page"""
    st.header("🔍 Single Customer Prediction")
    st.markdown("Enter customer details to predict churn probability")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 35)
    
    with col2:
        st.subheader("Account Information")
        credit_score = st.slider("Credit Score", 300, 850, 650)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0, step=1000.0)
    
    with col3:
        st.subheader("Banking Behavior")
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_credit_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0, step=1000.0)
    
    # Predict button
    if st.button("🎯 Predict Churn", type="primary"):
        # Prepare customer data
        customer_data = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_credit_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            churn_prediction, probability = predict_churn(spark, model, customer_data)
        
        if churn_prediction is not None:
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if churn_prediction == 1:
                    st.error("⚠️ **HIGH RISK**")
                    st.metric("Churn Prediction", "Will Churn")
                else:
                    st.success("✅ **LOW RISK**")
                    st.metric("Churn Prediction", "Will Stay")
            
            with col2:
                churn_prob = probability[1] * 100
                st.metric("Churn Probability", f"{churn_prob:.2f}%")
            
            with col3:
                retention_prob = probability[0] * 100
                st.metric("Retention Probability", f"{retention_prob:.2f}%")
            
            # Risk level indicator
            st.markdown("### Risk Level")
            if churn_prob > 70:
                st.error("🔴 Very High Risk - Immediate action required!")
            elif churn_prob > 50:
                st.warning("🟡 High Risk - Consider retention strategies")
            elif churn_prob > 30:
                st.info("🔵 Medium Risk - Monitor customer engagement")
            else:
                st.success("🟢 Low Risk - Customer is likely to stay")
            
            # Recommendations
            st.markdown("### 💡 Recommended Actions")
            if churn_prediction == 1:
                recommendations = []
                if is_active == 0:
                    recommendations.append("✓ Customer is inactive - engage with personalized offers")
                if num_products <= 2:
                    recommendations.append("✓ Offer additional products/services to increase engagement")
                if balance == 0:
                    recommendations.append("✓ Low balance - encourage deposits with promotional rates")
                if age > 50:
                    recommendations.append("✓ Senior customer - provide dedicated support and benefits")
                
                if recommendations:
                    for rec in recommendations:
                        st.write(rec)
                else:
                    st.write("✓ Schedule a personal call to understand customer needs")
                    st.write("✓ Offer loyalty rewards or exclusive benefits")
            else:
                st.write("✓ Customer shows positive engagement - maintain regular communication")
                st.write("✓ Consider upselling opportunities")

def show_batch_prediction(spark, model):
    """Batch prediction page"""
    st.header("📁 Batch Prediction")
    st.markdown("Upload a CSV file with customer data for batch predictions")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read CSV
        df_pandas = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df_pandas.head())
        
        if st.button("🚀 Run Batch Prediction"):
            with st.spinner("Processing predictions..."):
                # Convert to Spark DataFrame
                df_spark = spark.createDataFrame(df_pandas)
                
                # Make predictions
                predictions = model.transform(df_spark)
                
                # Convert back to Pandas
                results = predictions.select("*").toPandas()
                
                st.success("✅ Predictions completed!")
                
                # Show results
                st.subheader("Prediction Results")
                st.dataframe(results)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                total_customers = len(results)
                churn_count = len(results[results['prediction'] == 1])
                retention_count = total_customers - churn_count
                
                with col1:
                    st.metric("Total Customers", total_customers)
                with col2:
                    st.metric("Predicted Churn", churn_count)
                with col3:
                    st.metric("Predicted Retention", retention_count)
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )

def show_model_insights():
    """Model insights page"""
    st.header("📈 Model Insights")
    
    st.markdown("""
    ### Model Performance
    
    Our Bank Customer Churn Prediction model uses PySpark MLlib with multiple algorithms:
    - **Logistic Regression**: Baseline model for interpretability
    - **Random Forest**: Ensemble method for improved accuracy
    - **Gradient Boosted Trees**: Advanced ensemble technique
    
    ### Key Features Analyzed
    1. **Credit Score**: Customer creditworthiness
    2. **Age**: Customer demographic factor
    3. **Tenure**: Length of relationship with bank
    4. **Balance**: Account balance amount
    5. **Number of Products**: Products/services used
    6. **Active Membership**: Customer engagement level
    7. **Geography & Gender**: Demographic factors
    
    ### Top Churn Indicators
    - 🔴 **Inactive membership status**
    - 🔴 **Low number of products (1-2)**
    - 🔴 **Older age groups (50+)**
    - 🔴 **Zero or very high balances**
    - 🔴 **Short tenure (< 2 years)**
    
    ### Business Impact
    - Early identification of at-risk customers
    - Targeted retention campaigns
    - Resource optimization for customer service
    - Improved customer lifetime value
    - Reduced customer acquisition costs
    
    ### Recommended Actions
    1. **High-Risk Customers**: Immediate intervention with personalized offers
    2. **Medium-Risk Customers**: Engagement campaigns and regular follow-ups
    3. **Low-Risk Customers**: Upselling opportunities and loyalty programs
    """)
    
    # Sample statistics (you can load actual model metrics)
    st.markdown("### Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "~86%")
    with col2:
        st.metric("AUC-ROC", "~0.85")
    with col3:
        st.metric("Precision", "~75%")
    with col4:
        st.metric("Recall", "~65%")

if __name__ == "__main__":
    main()
