"""
Add this cell at the END of each notebook to save the models.
─────────────────────────────────────────────────────────────
NOTEBOOK 1  (ai_job_replacement)
─────────────────────────────────────────────────────────────
"""
import pickle, os

artifacts_job = {
    "xgb_regressor": xgb_reg,
    "xgb_classifier": xgb_clf,
    "rf_regressor": rf_reg,
    "gb_regressor": gb_reg,
    "label_encoders": le_dict,        # dict with keys: job_role, industry, country, automation_risk_category
    "feature_cols": feature_cols,
    "risk_labels": risk_labels.tolist(),   # ['High', 'Low', 'Medium'] or whatever order
}

with open("job_replacement_models.pkl", "wb") as f:
    pickle.dump(artifacts_job, f)
print(f"✅ Saved job_replacement_models.pkl  ({os.path.getsize('job_replacement_models.pkl')/1e6:.1f} MB)")


"""
─────────────────────────────────────────────────────────────
NOTEBOOK 2  (corporate_ai_adoption)  — already has most of this
Just make sure the key names match:
─────────────────────────────────────────────────────────────
"""
import pickle, os

artifacts_corp = {
    "xgb_regressor": xgb_reg,
    "xgb_classifier": xgb_clf,
    "rf_regressor": rf_reg,
    "gb_regressor": gb_reg,
    "label_encoders": le_dict,       # keys: industry, country, adoption_category
    "feature_cols": feature_cols,
    "adoption_labels": adoption_labels.tolist(),
}

with open("corporate_adoption_models.pkl", "wb") as f:
    pickle.dump(artifacts_corp, f)
print(f"✅ Saved corporate_adoption_models.pkl  ({os.path.getsize('corporate_adoption_models.pkl')/1e6:.1f} MB)")
