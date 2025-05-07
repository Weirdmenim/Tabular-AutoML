# 🚀 Tabular AutoML from Scratch

An advanced, modular AutoML pipeline for tabular data — built using Python, scikit-learn, and XGBoost.
This project automatically preprocesses data, compares multiple models using cross-validation, and evaluates the best-performing model on a test set with clear metrics and visualizations.

---

## 📈 Project Highlights

* 📦 Fully automated pipeline: from data loading to evaluation
* 🛠️ Modular codebase with `src/` components
* 🔍 Model comparison via cross-validation
* ✅ Final evaluation with F1, Precision, Recall, Confusion Matrix & ROC Curve
* 🧐 Best model: **XGBoost**, selected automatically

---

## 📂 Folder Structure

```
Tabular-AutoML-from-Scratch/
├── data/                     # Raw or preprocessed data (optional)
├── results/
│   └── best_model.joblib     # Saved best model
├── src/
│   ├── preprocess.py         # Preprocessing pipeline
│   ├── model_selector.py     # Model training & selection
│   └── evaluate.py           # Evaluation functions
├── main.py                   # Main script
├── requirements.txt          # Python dependencies
└── README.md                 # You're here
```

---

## 🧪 Dataset

* **Name**: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
* **Rows**: \~32,000
* **Columns**: 14
* **Task**: Binary classification — predict if income > 50K/year

---

## ⚙️ Setup & Usage

1. Clone the repo:

   ```bash
   git clone https://github.com/Weirdmenim/Tabular-AutoML/Tabular-AutoML-from-Scratch.git
   cd Tabular-AutoML-from-Scratch
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline:

   ```bash
   python main.py
   ```

---

## ✅ Evaluation Results

📊 **Test Metrics**:

* **F1 Score**: `0.7055`
* **Precision**: `0.7573`
* **Recall**: `0.6605`

🎯 **Best Model**: `XGBoost`

![ROC Curve](results/roc_curve.png)
![Confusion Matrix](results/confusion_matrix.png)

---

## 🔍 Key Learnings

* How to modularize an end-to-end ML pipeline
* Automating preprocessing for mixed-type tabular data
* Systematic model evaluation using cross-validation
* Visualizing model performance meaningfully

---

## 🔧 Potential Improvements

* 🔁 **Hyperparameter Tuning** — use `GridSearchCV` or `optuna` to fine-tune XGBoost
* 📊 **Feature Engineering** — add interaction terms or binning for numeric features
* ⚙️ **Feature Selection** — remove low-importance features (e.g., via SHAP, permutation importance)
* 🧪 **Ensembling** — average predictions across multiple models for better generalization
* 🧠 **Expand Model Pool** — add LightGBM, CatBoost, and stacking classifiers
* 📊 **Model Monitoring** — track metrics over time or across slices (e.g., gender, age)

---

## 🛠️ Future Features

* Add YAML-based config support
* CLI interface to control pipeline steps
* Add unit tests for all core components
* Build a simple Streamlit app for upload + prediction

---

## 📄 License

MIT License — free to use, adapt, and share.

---

## 🙌 Acknowledgments

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
* scikit-learn, XGBoost, pandas, matplotlib
