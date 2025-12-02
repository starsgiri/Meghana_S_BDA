# Version 2.0 Upgrade Guide

## 🚀 What's New in V2?

### Major Improvements:
1. **50x Larger Dataset**: 10,000 → 500,000 records
2. **Enhanced Features**: 11 → 16 features
3. **Optimized Models**: Better accuracy and generalization
4. **Dual Version Support**: Switch between V1 and V2 in app
5. **Better Performance**: Optimized Spark configurations

---

## 📊 Quick Comparison

| Feature | Version 1 | Version 2 |
|---------|-----------|-----------|
| Dataset Size | 10,000 records | 500,000 records |
| Features | 11 | 16 (+5 new) |
| Expected Accuracy | 85-86% | 87-88% |
| Training Time | 2-3 minutes | 5-10 minutes |
| Memory Required | 8GB RAM | 16GB RAM |
| File Size | 669 KB | 33.85 MB |

---

## ⚡ Quick Start V2

```bash
# Step 1: Generate large dataset
python generate_large_dataset.py

# Step 2: Process data (V2)
python data_processing_v2.py

# Step 3: Train models (V2)
python model_training_v2.py

# Step 4: Run app (auto-loads V2)
streamlit run streamlit_app.py
```

---

## 📝 New Features

### 1. TenureCategory
Categorizes customer loyalty:
- **New**: < 3 years
- **Regular**: 3-7 years  
- **Loyal**: 7+ years

### 2. EngagementScore
Composite metric (1-11):
- Products × 2
- Active Member × 3
- Credit Card × 1

**Higher score = Better engagement**

---

## 🎯 Which Version Should I Use?

### Use V1 (10K) if:
- ✅ Limited RAM (< 16GB)
- ✅ Quick training needed
- ✅ Learning/testing purposes
- ✅ Small production scale

### Use V2 (500K) if:
- ✅ Have 16GB+ RAM
- ✅ Want best accuracy
- ✅ Production deployment
- ✅ Large-scale predictions
- ✅ Better model generalization

---

## 💻 System Requirements

### Minimum (V1):
- **RAM**: 8GB
- **Disk**: 500MB free
- **CPU**: Dual-core

### Recommended (V2):
- **RAM**: 16GB+
- **Disk**: 1GB+ free
- **CPU**: Quad-core+

---

## 📂 Project Structure (V2)

```
Meghana_S_BDA/
├── dataset/
│   ├── Churn_Modelling.csv (V1 - 669KB)
│   ├── Churn_Modelling_Large_500K.csv (V2 - 33.85MB) ⭐NEW
│   ├── processed_churn_data.parquet (V1)
│   └── processed_churn_data_v2.parquet (V2) ⭐NEW
│
├── models/
│   ├── best_churn_model/ (V1)
│   ├── best_churn_model_v2/ (V2) ⭐NEW
│   ├── model_v2_metadata.txt ⭐NEW
│   └── [other models...]
│
├── V1 Scripts:
│   ├── data_processing.py
│   ├── model_training.py
│   └── pyspark_*.txt
│
├── V2 Scripts: ⭐NEW
│   ├── generate_large_dataset.py
│   ├── data_processing_v2.py
│   └── model_training_v2.py
│
├── streamlit_app.py (Updated for V1/V2 support)
│
└── Documentation:
    ├── README.md
    ├── CHANGELOG_V2.md ⭐NEW
    ├── VERSION_2_GUIDE.md ⭐NEW
    └── [other docs...]
```

---

## 🔄 How to Switch Versions

### In Streamlit App:
1. Launch app: `streamlit run streamlit_app.py`
2. Look for dropdown in top-right
3. Select "v1 (10K)" or "v2 (500K)"
4. App automatically reloads correct model

### In Code:
```python
from pyspark.ml import PipelineModel

# Load V1
model_v1 = PipelineModel.load("models/best_churn_model")

# Load V2
model_v2 = PipelineModel.load("models/best_churn_model_v2")
```

---

## 📈 Performance Expectations

### V1 (10K):
- **Training Time**: 2-3 minutes
- **Accuracy**: 85-86%
- **AUC-ROC**: 0.84-0.86
- **Memory Usage**: ~4GB

### V2 (500K):
- **Training Time**: 5-10 minutes
- **Accuracy**: 87-88%
- **AUC-ROC**: 0.86-0.88
- **Memory Usage**: ~8GB

---

## 🐛 Troubleshooting

### "Out of Memory" Error:
```bash
# Reduce memory in scripts:
# Edit data_processing_v2.py or model_training_v2.py
.config("spark.driver.memory", "4g")  # Instead of 8g
.config("spark.executor.memory", "4g")
```

### V2 Models Not Found:
```bash
# Train V2 models:
python model_training_v2.py

# Or use V1 in app dropdown
```

### Slow Training:
```bash
# Normal for V2 (50x larger dataset)
# Expected: 5-10 minutes
# Optimize: Use machine with SSD and 16GB+ RAM
```

---

## 📚 Documentation

- **Full Changelog**: `CHANGELOG_V2.md`
- **Main README**: `README.md`
- **Quick Start**: `QUICKSTART.md`
- **This Guide**: `VERSION_2_GUIDE.md`

---

## ✅ Migration Checklist

- [ ] Generate 500K dataset
- [ ] Process V2 data
- [ ] Train V2 models
- [ ] Test V2 predictions
- [ ] Compare V1 vs V2 results
- [ ] Deploy to Streamlit
- [ ] Switch between versions
- [ ] Document findings

---

## 🎯 Next Steps

1. **Try V2**: Run the quick start commands
2. **Compare**: Test both versions
3. **Evaluate**: Check accuracy improvements
4. **Deploy**: Use best version for your needs
5. **Iterate**: Provide feedback for V2.1

---

## 💡 Pro Tips

1. **Start with V1**: Learn the system
2. **Upgrade to V2**: When ready for production
3. **Keep Both**: Use V1 for testing, V2 for production
4. **Monitor Performance**: Track metrics over time
5. **Document Changes**: Note what works best

---

**Version 2.0 is ready!**  
Switch between versions anytime in the Streamlit app.

For detailed changes, see `CHANGELOG_V2.md`
