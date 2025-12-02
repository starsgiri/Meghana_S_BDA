#!/usr/bin/env python3
"""
PySpark MLlib Model Training Script V2 for Bank Customer Churn Prediction
Large Dataset Version (500,000 records)
This script trains multiple classification models on the large dataset and saves as version 2.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import datetime

def main():
    # ============================================================================
    # STEP 1: Initialize PySpark Session with Optimized Settings
    # ============================================================================
    
    print("Initializing Spark Session for Model Training V2...")
    spark = SparkSession.builder \
        .appName("BankChurnModelTraining_V2") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    print("Spark Session V2 Created Successfully!")
    print(f"Spark Version: {spark.version}")
    
    # ============================================================================
    # STEP 2: Load Preprocessed Data V2
    # ============================================================================
    
    print("\nLoading preprocessed data V2...")
    df = spark.read.parquet("dataset/processed_churn_data_v2.parquet")
    
    print(f"Dataset loaded: {df.count():,} rows, {len(df.columns)} columns")
    df.printSchema()
    
    # Cache the dataframe
    df.cache()
    df.count()
    print("Dataset cached for better performance!")
    
    # ============================================================================
    # STEP 3: Prepare Features for ML
    # ============================================================================
    
    print("\nPreparing features for V2 model...")
    
    categorical_cols = ["Geography", "Gender", "AgeGroup", "BalanceCategory", 
                       "CreditScoreCategory", "TenureCategory"]
    numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                     "HasCrCard", "IsActiveMember", "EstimatedSalary", "EngagementScore"]
    
    # String Indexing
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_Index", handleInvalid="keep") 
                for col in categorical_cols]
    
    feature_cols = [col+"_Index" for col in categorical_cols] + numerical_cols
    print(f"\nTotal feature columns: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    
    # Vector Assembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
    
    # Standard Scaler
    scaler = StandardScaler(inputCol="assembled_features", outputCol="features", 
                           withStd=True, withMean=True)
    
    # ============================================================================
    # STEP 4: Split Data (Stratified)
    # ============================================================================
    
    print("\nSplitting data (70-30 split)...")
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
    
    # Cache train and test data
    train_data.cache()
    test_data.cache()
    
    train_count = train_data.count()
    test_count = test_data.count()
    
    print(f"Training set size: {train_count:,}")
    print(f"Testing set size: {test_count:,}")
    
    print("\n=== Training Set Churn Distribution ===")
    train_churn = train_data.groupBy("Exited").count().orderBy("Exited")
    train_churn.show()
    
    print("\n=== Testing Set Churn Distribution ===")
    test_churn = test_data.groupBy("Exited").count().orderBy("Exited")
    test_churn.show()
    
    # ============================================================================
    # STEP 5: Model 1 - Logistic Regression V2
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL V2.1: LOGISTIC REGRESSION")
    print("="*80)
    
    lr = LogisticRegression(labelCol="Exited", featuresCol="features", 
                           maxIter=150, regParam=0.01, elasticNetParam=0.1)
    lr_pipeline = Pipeline(stages=indexers + [assembler, scaler, lr])
    
    print("\nTraining Logistic Regression V2 model...")
    start_time = datetime.datetime.now()
    lr_model = lr_pipeline.fit(train_data)
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    print("Making predictions...")
    lr_predictions = lr_model.transform(test_data)
    
    # Evaluate
    binary_evaluator = BinaryClassificationEvaluator(labelCol="Exited", metricName="areaUnderROC")
    multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="Exited", predictionCol="prediction")
    
    lr_auc = binary_evaluator.evaluate(lr_predictions)
    lr_accuracy = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "accuracy"})
    lr_precision = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
    lr_recall = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
    lr_f1 = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "f1"})
    
    print("\n=== Logistic Regression V2 Results ===")
    print(f"Training Time: {training_time:.2f}s")
    print(f"AUC-ROC: {lr_auc:.4f}")
    print(f"Accuracy: {lr_accuracy:.4f}")
    print(f"Precision: {lr_precision:.4f}")
    print(f"Recall: {lr_recall:.4f}")
    print(f"F1-Score: {lr_f1:.4f}")
    
    print("\n=== Confusion Matrix ===")
    lr_predictions.groupBy("Exited", "prediction").count().orderBy("Exited", "prediction").show()
    
    # ============================================================================
    # STEP 6: Model 2 - Random Forest V2 (Optimized)
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL V2.2: RANDOM FOREST CLASSIFIER (OPTIMIZED)")
    print("="*80)
    
    rf = RandomForestClassifier(labelCol="Exited", featuresCol="features", 
                                numTrees=150, maxDepth=12, minInstancesPerNode=50,
                                featureSubsetStrategy="sqrt", seed=42)
    rf_pipeline = Pipeline(stages=indexers + [assembler, scaler, rf])
    
    print("\nTraining Random Forest V2 model...")
    start_time = datetime.datetime.now()
    rf_model = rf_pipeline.fit(train_data)
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    print("Making predictions...")
    rf_predictions = rf_model.transform(test_data)
    
    # Evaluate
    rf_auc = binary_evaluator.evaluate(rf_predictions)
    rf_accuracy = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "accuracy"})
    rf_precision = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
    rf_recall = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
    rf_f1 = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "f1"})
    
    print("\n=== Random Forest V2 Results ===")
    print(f"Training Time: {training_time:.2f}s")
    print(f"AUC-ROC: {rf_auc:.4f}")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"Precision: {rf_precision:.4f}")
    print(f"Recall: {rf_recall:.4f}")
    print(f"F1-Score: {rf_f1:.4f}")
    
    # Feature Importance
    rf_model_stage = rf_model.stages[-1]
    feature_importance = rf_model_stage.featureImportances
    
    print("\n=== Top 10 Feature Importances V2 ===")
    importance_dict = {feature_cols[i]: float(importance) 
                      for i, importance in enumerate(feature_importance) if importance > 0}
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # ============================================================================
    # STEP 7: Model 3 - Gradient Boosted Trees V2 (Optimized)
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL V2.3: GRADIENT BOOSTED TREES (OPTIMIZED)")
    print("="*80)
    
    gbt = GBTClassifier(labelCol="Exited", featuresCol="features", 
                        maxIter=120, maxDepth=6, stepSize=0.1, seed=42)
    gbt_pipeline = Pipeline(stages=indexers + [assembler, scaler, gbt])
    
    print("\nTraining Gradient Boosted Trees V2 model...")
    start_time = datetime.datetime.now()
    gbt_model = gbt_pipeline.fit(train_data)
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    print("Making predictions...")
    gbt_predictions = gbt_model.transform(test_data)
    
    # Evaluate
    gbt_auc = binary_evaluator.evaluate(gbt_predictions)
    gbt_accuracy = multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "accuracy"})
    gbt_precision = multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
    gbt_recall = multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
    gbt_f1 = multiclass_evaluator.evaluate(gbt_predictions, {multiclass_evaluator.metricName: "f1"})
    
    print("\n=== Gradient Boosted Trees V2 Results ===")
    print(f"Training Time: {training_time:.2f}s")
    print(f"AUC-ROC: {gbt_auc:.4f}")
    print(f"Accuracy: {gbt_accuracy:.4f}")
    print(f"Precision: {gbt_precision:.4f}")
    print(f"Recall: {gbt_recall:.4f}")
    print(f"F1-Score: {gbt_f1:.4f}")
    
    # ============================================================================
    # STEP 8: Model Comparison V2
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL COMPARISON - VERSION 2")
    print("="*80)
    
    print("\n{:<30} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Model", "AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score"))
    print("-" * 90)
    print("{:<30} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        "Logistic Regression V2", lr_auc, lr_accuracy, lr_precision, lr_recall, lr_f1))
    print("{:<30} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        "Random Forest V2", rf_auc, rf_accuracy, rf_precision, rf_recall, rf_f1))
    print("{:<30} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        "Gradient Boosted Trees V2", gbt_auc, gbt_accuracy, gbt_precision, gbt_recall, gbt_f1))
    
    # ============================================================================
    # STEP 9: Save Models as Version 2
    # ============================================================================
    
    print("\n\nSaving models as Version 2...")
    
    models = {
        "Logistic Regression": (lr_model, lr_auc, lr_f1),
        "Random Forest": (rf_model, rf_auc, rf_f1),
        "Gradient Boosted Trees": (gbt_model, gbt_auc, gbt_f1)
    }
    
    # Determine best model (based on F1-score for balanced evaluation)
    best_model_name = max(models, key=lambda x: models[x][2])
    best_model = models[best_model_name][0]
    best_auc = models[best_model_name][1]
    best_f1 = models[best_model_name][2]
    
    print(f"\n=== Best Model V2: {best_model_name} ===")
    print(f"AUC-ROC: {best_auc:.4f}")
    print(f"F1-Score: {best_f1:.4f}")
    
    # Save best model as V2
    best_model.write().overwrite().save("models/best_churn_model_v2")
    print("\nBest model V2 saved to 'models/best_churn_model_v2'")
    
    # Save all models separately with V2 suffix
    lr_model.write().overwrite().save("models/logistic_regression_model_v2")
    rf_model.write().overwrite().save("models/random_forest_model_v2")
    gbt_model.write().overwrite().save("models/gradient_boosted_trees_model_v2")
    print("All models V2 saved successfully!")
    
    # Save model metadata
    metadata = {
        "version": "2.0",
        "training_date": str(datetime.datetime.now()),
        "dataset_size": df.count(),
        "train_size": train_count,
        "test_size": test_count,
        "best_model": best_model_name,
        "best_auc": best_auc,
        "best_f1": best_f1,
        "features_count": len(feature_cols),
        "lr_metrics": {"auc": lr_auc, "accuracy": lr_accuracy, "f1": lr_f1},
        "rf_metrics": {"auc": rf_auc, "accuracy": rf_accuracy, "f1": rf_f1},
        "gbt_metrics": {"auc": gbt_auc, "accuracy": gbt_accuracy, "f1": gbt_f1}
    }
    
    # Save metadata as text file
    with open("models/model_v2_metadata.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("BANK CHURN PREDICTION MODEL - VERSION 2.0\n")
        f.write("="*80 + "\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print("Model metadata saved to 'models/model_v2_metadata.txt'")
    
    # ============================================================================
    # STEP 10: Analyze Predictions V2
    # ============================================================================
    
    print("\n" + "="*80)
    print("CHURN PREDICTION ANALYSIS - VERSION 2")
    print("="*80)
    
    best_predictions = best_model.transform(test_data)
    high_risk = best_predictions.filter(col("prediction") == 1)
    high_risk_count = high_risk.count()
    
    print(f"\nTotal high-risk customers (predicted to churn): {high_risk_count:,}")
    print(f"Percentage of test set: {(high_risk_count/test_count)*100:.2f}%")
    
    print("\n=== High-Risk Customer Characteristics V2 ===")
    print("\nBy Geography:")
    high_risk.groupBy("Geography").count().orderBy(col("count").desc()).show()
    
    print("\nBy Age Group:")
    high_risk.groupBy("AgeGroup").count().orderBy(col("count").desc()).show()
    
    print("\nBy Tenure Category:")
    high_risk.groupBy("TenureCategory").count().orderBy(col("count").desc()).show()
    
    print("\nBy Active Membership:")
    high_risk.groupBy("IsActiveMember").count().orderBy(col("count").desc()).show()
    
    print("\nBy Number of Products:")
    high_risk.groupBy("NumOfProducts").count().orderBy("NumOfProducts").show()
    
    # ============================================================================
    # STEP 11: Performance Summary
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL V2 PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\nDataset Version: 2.0")
    print(f"Training Dataset Size: {train_count:,} records")
    print(f"Testing Dataset Size: {test_count:,} records")
    print(f"Total Features Used: {len(feature_cols)}")
    print(f"Best Model: {best_model_name}")
    print(f"Best Model AUC-ROC: {best_auc:.4f}")
    print(f"Best Model F1-Score: {best_f1:.4f}")
    
    print("\n=== Model Training V2 Complete! ===")
    
    # Cleanup
    df.unpersist()
    train_data.unpersist()
    test_data.unpersist()
    
    spark.stop()

if __name__ == "__main__":
    main()
