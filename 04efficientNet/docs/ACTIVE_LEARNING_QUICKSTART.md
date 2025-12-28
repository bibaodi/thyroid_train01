## Active Learning Quick Start Guide

### Overview

This guide shows you how to quickly get started with active learning for your thyroid nodule classification system.

---

## 📋 Prerequisites

✅ Existing trained model (EfficientNet-B0)  
✅ Training dataset with potential mislabeled samples  
✅ Python environment with PyTorch, pandas, scikit-learn

---

## 🚀 Quick Start (3 Steps)

### Step 1: Run Uncertainty Analysis (5 minutes)

```python
from active_learning_utils import ActiveLearningManager, UncertaintyEstimator
from train_nodule_feature_cnn_model_v75 import load_model, load_data

# Load your model and data
model = load_model('path/to/model.pth')
train_df, train_loader = load_data()

# Initialize active learning
al_manager = ActiveLearningManager(
    model=model,
    config={'al_strategy': 'hybrid'},
    output_dir='./active_learning_results'
)

# Identify uncertain samples
selected_indices, uncertainty_df = al_manager.identify_uncertain_samples(
    dataset=train_dataset,
    dataloader=train_loader,
    device=device,
    top_k=100  # Review 100 samples
)

# Export for review
al_manager.export_for_annotation(
    selected_indices=selected_indices,
    uncertainty_df=uncertainty_df,
    dataset_df=train_df,
    image_root='/path/to/images',
    cycle=0
)
```

**Output:** `active_learning_results/annotations/cycle_0_for_review.csv`

---

### Step 2: Annotate Samples (15-30 minutes)

**Option A: Command-line Interface**
```bash
python annotation_interface_simple.py active_learning_results/annotations/cycle_0_for_review.csv
```

**Option B: Manual CSV Editing**
1. Open `cycle_0_for_review.csv` in Excel/LibreOffice
2. For each row, fill in:
   - `annotation_action`: 'confirm', 'correct', 'flag', or 'skip'
   - `corrected_label`: 0 or 1 (if correcting)
   - `annotator_notes`: Your observations
3. Save as `cycle_0_reviewed.csv`

**Example:**
```csv
access_no,sop_uid,ground_truth,predicted_class,confidence,annotation_action,corrected_label,annotator_notes
PATIENT001,1.2.840.xxx,0,1,0.85,correct,1,"Clear malignant features"
PATIENT002,1.2.840.yyy,1,1,0.95,confirm,,"Correct label"
PATIENT003,1.2.840.zzz,0,1,0.65,flag,,"Unclear, need expert"
```

---

### Step 3: Update Dataset & Retrain (10 minutes)

```python
# Import annotations
annotations_df = al_manager.import_annotations(cycle=0)

# Update dataset
updated_train_df, stats = al_manager.update_dataset(
    annotations_df=annotations_df,
    original_df=train_df
)

# Save updated dataset
updated_train_df.to_csv('updated_train_data.csv', index=False)

# Retrain model with corrected labels
# (Use your existing training script)
```

**Output:**
```
✅ Dataset updated:
   - Labels corrected: 15
   - Labels confirmed: 80
   - Flagged for expert: 5
```

---

## 📊 What Gets Identified?

Active learning identifies samples that are:

1. **High Uncertainty** - Model is unsure (entropy > 0.7)
2. **Disagreement** - Model prediction ≠ Ground truth
3. **Low Confidence** - p_true < 0.2 (your current OOF threshold)
4. **Diverse** - Representative of different feature spaces

**Example uncertain samples:**
```
Sample 1: Ground truth=Benign, Predicted=Malignant, Confidence=85%
→ Likely mislabeled, should review

Sample 2: Ground truth=Malignant, Predicted=Malignant, Confidence=55%
→ Correct but uncertain, difficult case

Sample 3: Ground truth=Benign, Predicted=Benign, Confidence=51%
→ Correct but barely, edge case
```

---

## 🔄 Full Active Learning Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                   ACTIVE LEARNING CYCLE                      │
└─────────────────────────────────────────────────────────────┘

Cycle 1:
  1. Train model on initial data (120K samples)
  2. Identify 100 most uncertain samples
  3. Human reviews and corrects (15 corrections)
  4. Update dataset (120K samples, 15 corrected)
  5. Retrain model
  
Cycle 2:
  1. Train on updated data
  2. Identify 100 new uncertain samples
  3. Human reviews (10 corrections)
  4. Update dataset (120K samples, 25 total corrected)
  5. Retrain model
  
Cycle 3:
  1. Train on updated data
  2. Identify 100 new uncertain samples
  3. Human reviews (5 corrections)
  4. Update dataset (120K samples, 30 total corrected)
  5. Final training

Result: Cleaner dataset, better model performance
```

---

## 📈 Expected Results

Based on your current system:

**Before Active Learning:**
- Training data: 120K samples (~5% potentially mislabeled)
- OOF filtering: Removes ~6K suspicious samples
- Model performance: Baseline

**After 3 AL Cycles (300 samples reviewed):**
- Training data: 120K samples (~1% mislabeled)
- Corrections: ~30-50 labels fixed
- Model performance: +2-5% improvement
- Time invested: 1-2 hours annotation

**ROI:**
- Annotation time: 1-2 hours
- Performance gain: 2-5%
- Data quality: Significantly improved
- Confidence: Higher trust in predictions

---

## 🎯 Strategies

### Strategy 1: Uncertainty-Based (Recommended for Start)
```python
config = {'al_strategy': 'uncertainty'}
```
- Selects samples with highest entropy
- Good for finding difficult cases
- Fast and simple

### Strategy 2: Disagreement-Based (Your Current OOF Enhanced)
```python
config = {'al_strategy': 'disagreement'}
```
- Selects samples where model disagrees with label
- Good for finding mislabeled samples
- Similar to your current approach

### Strategy 3: Hybrid (Recommended for Production)
```python
config = {'al_strategy': 'hybrid'}
```
- Combines uncertainty + disagreement + diversity
- Balanced approach
- Best overall performance

---

## 🛠️ Configuration

```python
AL_CONFIG = {
    # Strategy
    'al_strategy': 'hybrid',  # 'uncertainty', 'disagreement', 'diversity', 'hybrid'
    
    # Sampling
    'samples_per_cycle': 100,  # How many to review per cycle
    'num_cycles': 3,           # How many AL cycles
    
    # Training
    'epochs_per_cycle': 10,    # Training epochs per cycle
    
    # Thresholds
    'uncertainty_threshold': 0.7,  # Entropy threshold
    'confidence_threshold': 0.2,   # Your current OOF threshold
    
    # Diversity
    'diversity_weight': 0.2,   # Weight for diversity sampling
}
```

---

## 📁 File Structure

```
active_learning_results/
├── annotations/
│   ├── cycle_0_for_review.csv      # Samples to annotate
│   ├── cycle_0_reviewed.csv        # Your annotations
│   ├── cycle_1_for_review.csv
│   └── cycle_1_reviewed.csv
├── results/
│   ├── cycle_0_results.json        # Metrics per cycle
│   ├── cycle_1_results.json
│   ├── corrections_log_*.json      # All corrections
│   └── updated_train_data_cycle_*.csv
└── models/
    ├── model_cycle_0.pth
    ├── model_cycle_1.pth
    └── model_final.pth
```

---

## 🔍 Monitoring Progress

### Check Uncertainty Distribution
```python
import pandas as pd
import matplotlib.pyplot as plt

uncertainty_df = pd.read_csv('uncertainty_analysis.csv')

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(uncertainty_df['entropy'], bins=50)
plt.title('Entropy Distribution')

plt.subplot(132)
plt.hist(uncertainty_df['p_true'], bins=50)
plt.title('P(True) Distribution')

plt.subplot(133)
plt.scatter(uncertainty_df['entropy'], uncertainty_df['p_true'])
plt.title('Entropy vs P(True)')

plt.tight_layout()
plt.show()
```

### Track Corrections Over Cycles
```python
corrections_per_cycle = [15, 10, 5]  # From your annotation results
plt.plot(corrections_per_cycle, marker='o')
plt.xlabel('Cycle')
plt.ylabel('Number of Corrections')
plt.title('Label Corrections per AL Cycle')
plt.show()
```

---

## ⚠️ Common Issues

### Issue 1: Too Many Uncertain Samples
**Problem:** 1000+ samples flagged as uncertain  
**Solution:** Increase confidence threshold or reduce top_k

```python
# Instead of:
top_k = 1000

# Use:
top_k = 100  # Start small
uncertainty_threshold = 0.8  # Higher threshold
```

### Issue 2: Annotation Fatigue
**Problem:** Reviewing 500 samples is exhausting  
**Solution:** Batch annotations, multiple annotators

```python
# Split into batches
batch_size = 50
for i in range(0, len(selected), batch_size):
    batch = selected[i:i+batch_size]
    # Annotate batch
    # Take break
```

### Issue 3: No Improvement After AL
**Problem:** Model performance doesn't improve  
**Solution:** Check if corrections are meaningful

```python
# Analyze corrections
corrections = pd.read_csv('corrections_log.json')
print(f"Benign→Malignant: {(corrections['old_label']==0).sum()}")
print(f"Malignant→Benign: {(corrections['old_label']==1).sum()}")

# If very few corrections, data might already be clean
```

---

## 🎓 Best Practices

1. **Start Small**: Begin with 50-100 samples per cycle
2. **Multiple Cycles**: 3-5 cycles usually sufficient
3. **Track Everything**: Save all annotations and corrections
4. **Validate**: Check if corrections improve validation performance
5. **Expert Review**: Flag difficult cases for expert radiologists
6. **Iterate**: Adjust strategy based on results

---

## 📚 Next Steps

1. ✅ Read the full analysis report: `ACTIVE_LEARNING_ANALYSIS_REPORT.md`
2. ✅ Run pilot with 100 samples
3. ✅ Evaluate results
4. ✅ Scale up if successful
5. ✅ Integrate into production pipeline

---

## 🆘 Support

**Questions?**
- Check `ACTIVE_LEARNING_ANALYSIS_REPORT.md` for detailed analysis
- Review `srcs/active_learning_utils.py` for implementation details
- See `srcs/train_with_active_learning_example.py` for integration example

**Issues?**
- Verify file paths are correct
- Check that model is loaded properly
- Ensure dataset has required columns
- Review error messages carefully

---

**Ready to start?** Run the Quick Start steps above! 🚀
