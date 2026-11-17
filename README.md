# Supervised Classification / Regression Modeling & Tuning

Primary deliverable: `supervised_classification_model_tuning.ipynb`

Includes:
- Four classifiers: Logistic Regression, SVM, Random Forest, Extra Trees
- GridSearchCV with StratifiedKFold
- Pipeline-based preprocessing to avoid leakage
- Precision, Recall, F1, ROC-AUC
- ROC curves, confusion matrices, classification reports
- Champion selection using mean CV F1
- Serialized champion model output

Run:
1. `pip install pandas numpy matplotlib seaborn scikit-learn joblib`
2. Open notebook in Jupyter or Colab.
3. Keep `customer-churn-training.csv` beside notebook.
4. Run all cells.

Dataset warning: The supplied CSV contains only 12 rows. Results are educational/demo-only and are not reliable production performance estimates.
