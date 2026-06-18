"""
Train XGBoost models on synthetic data matching:
  - Kaggle: AI Job Replacement Analysis 2020-2026
  - Kaggle: Corporate AI Adoption & ROI Dataset 2015-2035

Outputs ONNX model files + JSON metadata (no xgboost needed at runtime).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import onnxmltools
from onnxmltools.convert import convert_xgboost as convert_xgb
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort
import json
import os

os.makedirs("models", exist_ok=True)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# JOB REPLACEMENT DATASET  (mirrors Kaggle: shayanzk/ai-job-replacement-2020-2026)
# ═══════════════════════════════════════════════════════════════════════════

JOB_ROLES = [
    "Software Engineer", "Data Analyst", "Customer Service Rep",
    "Accountant", "Marketing Manager", "HR Specialist", "Truck Driver",
    "Nurse", "Teacher", "Sales Associate", "Graphic Designer",
    "Financial Advisor", "Warehouse Worker", "Journalist",
    "Legal Clerk", "Pharmacist", "Receptionist", "Data Entry Clerk",
    "DevOps Engineer", "Project Manager", "Insurance Adjuster",
    "Bank Teller", "Radiologist", "Content Writer", "Translator",
    "Paralegal", "Supply Chain Analyst", "Retail Cashier",
    "Medical Coder", "Tax Preparer", "Loan Officer", "Telemarketer",
    "Quality Inspector", "Travel Agent", "Security Guard",
]

INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Retail", "Manufacturing",
    "Education", "Transportation", "Media", "Insurance", "Legal",
    "Hospitality", "Agriculture", "Energy", "Telecom", "Real Estate",
]

COUNTRIES = [
    "USA", "UK", "Germany", "India", "China", "Japan", "Canada",
    "Australia", "France", "Brazil", "South Korea", "Netherlands",
    "Sweden", "Singapore", "UAE", "Mexico", "Spain", "Italy",
    "South Africa", "Nigeria",
]

# Roles with inherently higher AI replacement risk
HIGH_RISK_ROLES = {
    "Data Entry Clerk", "Bank Teller", "Telemarketer", "Retail Cashier",
    "Travel Agent", "Tax Preparer", "Receptionist", "Legal Clerk",
    "Loan Officer", "Insurance Adjuster", "Medical Coder",
}
LOW_RISK_ROLES = {
    "Nurse", "Teacher", "Radiologist", "Pharmacist", "Social Worker",
    "Software Engineer", "DevOps Engineer",
}

N_JOB = 6000

def make_job_dataset():
    role_arr = np.random.choice(JOB_ROLES, N_JOB)
    ind_arr  = np.random.choice(INDUSTRIES, N_JOB)
    ctry_arr = np.random.choice(COUNTRIES, N_JOB)

    # Base automation risk varies by role
    base_risk = np.where(
        np.isin(role_arr, list(HIGH_RISK_ROLES)), np.random.uniform(55, 95, N_JOB),
        np.where(
            np.isin(role_arr, list(LOW_RISK_ROLES)), np.random.uniform(10, 40, N_JOB),
            np.random.uniform(25, 75, N_JOB),
        )
    )

    year                      = np.random.randint(2020, 2027, N_JOB)
    automation_risk_percent   = np.clip(base_risk + np.random.normal(0, 5, N_JOB), 1, 99)
    skill_gap_index           = np.random.uniform(0.1, 1.0, N_JOB)
    salary_before_usd         = np.random.uniform(25_000, 150_000, N_JOB)
    salary_change_percent     = np.random.uniform(-20, 40, N_JOB)
    salary_after_usd          = salary_before_usd * (1 + salary_change_percent / 100)
    skill_demand_growth_pct   = np.random.uniform(-10, 60, N_JOB)
    remote_feasibility_score  = np.random.uniform(0.1, 1.0, N_JOB)
    ai_adoption_level         = np.random.uniform(0.1, 1.0, N_JOB)
    education_req_level       = np.random.uniform(0.2, 1.0, N_JOB)
    skill_transition_pressure = np.random.uniform(0.1, 1.0, N_JOB)
    wage_volatility_index     = np.random.uniform(0.05, 0.8, N_JOB)
    reskilling_urgency_score  = np.random.uniform(0.1, 1.0, N_JOB)
    ai_disruption_intensity   = np.random.uniform(0.1, 1.0, N_JOB)

    # Target: ai_replacement_score (0-100) — driven by realistic factors
    score = (
        0.35 * automation_risk_percent
        + 0.20 * ai_adoption_level * 100
        + 0.15 * ai_disruption_intensity * 100
        - 0.15 * education_req_level * 100
        + 0.10 * skill_gap_index * 100
        + 0.05 * reskilling_urgency_score * 100
        + np.random.normal(0, 4, N_JOB)
    )
    score = np.clip(score, 0, 100)

    risk_cat = np.where(score > 70, "High", np.where(score > 40, "Medium", "Low"))

    le_role = LabelEncoder().fit(JOB_ROLES)
    le_ind  = LabelEncoder().fit(INDUSTRIES)
    le_ctry = LabelEncoder().fit(COUNTRIES)

    df = pd.DataFrame({
        "year":                         year,
        "automation_risk_percent":      automation_risk_percent,
        "skill_gap_index":              skill_gap_index,
        "salary_before_usd":            salary_before_usd,
        "salary_after_usd":             salary_after_usd,
        "salary_change_percent":        salary_change_percent,
        "skill_demand_growth_percent":  skill_demand_growth_pct,
        "remote_feasibility_score":     remote_feasibility_score,
        "ai_adoption_level":            ai_adoption_level,
        "education_requirement_level":  education_req_level,
        "skill_transition_pressure":    skill_transition_pressure,
        "wage_volatility_index":        wage_volatility_index,
        "reskilling_urgency_score":     reskilling_urgency_score,
        "ai_disruption_intensity":      ai_disruption_intensity,
        "job_role_enc":                 le_role.transform(role_arr),
        "industry_enc":                 le_ind.transform(ind_arr),
        "country_enc":                  le_ctry.transform(ctry_arr),
        "ai_replacement_score":         score,
        "risk_category":                risk_cat,
    })
    return df, le_role, le_ind, le_ctry


print("── Generating job replacement data …")
df_job, le_role, le_ind_j, le_ctry_j = make_job_dataset()

FEATURE_COLS_JOB = [
    "year", "automation_risk_percent", "skill_gap_index",
    "salary_before_usd", "salary_after_usd", "salary_change_percent",
    "skill_demand_growth_percent", "remote_feasibility_score",
    "ai_adoption_level", "education_requirement_level",
    "skill_transition_pressure", "wage_volatility_index",
    "reskilling_urgency_score", "ai_disruption_intensity",
    "job_role_enc", "industry_enc", "country_enc",
]
RISK_LABELS = ["Low", "Medium", "High"]

le_risk = LabelEncoder().fit(RISK_LABELS)
df_job["risk_idx"] = le_risk.transform(df_job["risk_category"])

X_j = df_job[FEATURE_COLS_JOB].values.astype(np.float32)
y_reg_j = df_job["ai_replacement_score"].values.astype(np.float32)
y_clf_j = df_job["risk_idx"].values.astype(np.int32)

Xtr_j, Xte_j, yr_tr_j, yr_te_j, yc_tr_j, yc_te_j = train_test_split(
    X_j, y_reg_j, y_clf_j, test_size=0.2, random_state=42
)

print("── Training job regressor …")
reg_j = xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="rmse", random_state=42,
)
reg_j.fit(Xtr_j, yr_tr_j, eval_set=[(Xte_j, yr_te_j)], verbose=False)
rmse_j = mean_squared_error(yr_te_j, reg_j.predict(Xte_j)) ** 0.5
print(f"   RMSE={rmse_j:.2f}")

print("── Training job classifier …")
clf_j = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, num_class=3,
    eval_metric="mlogloss", random_state=42,
)
clf_j.fit(Xtr_j, yc_tr_j, eval_set=[(Xte_j, yc_te_j)], verbose=False)
acc_j = accuracy_score(yc_te_j, clf_j.predict(Xte_j))
print(f"   Accuracy={acc_j:.3f}")

# ── Convert job models to ONNX ──────────────────────────────────────────────
print("── Converting job models to ONNX …")
n_feat_j = len(FEATURE_COLS_JOB)
init_type = [("float_input", FloatTensorType([None, n_feat_j]))]

onnx_reg_j = convert_xgb(reg_j.get_booster(), initial_types=init_type, target_opset=12)
onnx_clf_j = convert_xgb(clf_j.get_booster(), initial_types=init_type, target_opset=12)

onnxmltools.utils.save_model(onnx_reg_j, "models/job_reg.onnx")
onnxmltools.utils.save_model(onnx_clf_j, "models/job_clf.onnx")

# ── Verify ONNX job models ──────────────────────────────────────────────────
sess_reg_j = ort.InferenceSession("models/job_reg.onnx")
sess_clf_j = ort.InferenceSession("models/job_clf.onnx")
sample = Xte_j[:5]
pred_reg  = sess_reg_j.run(None, {"float_input": sample})[0].flatten()
pred_clf  = sess_clf_j.run(None, {"float_input": sample})[1]  # label output
# pred_clf is a dict {0: p0, 1: p1, 2: p2} per row from onnxmltools multi-class
print(f"   ONNX regressor sample: {pred_reg.round(1)}")

# ── Save job metadata ───────────────────────────────────────────────────────
job_meta = {
    "feature_cols": FEATURE_COLS_JOB,
    "risk_labels":  RISK_LABELS,
    "label_encoders": {
        "job_role":  list(le_role.classes_),
        "industry":  list(le_ind_j.classes_),
        "country":   list(le_ctry_j.classes_),
    },
}
with open("models/job_meta.json", "w") as f:
    json.dump(job_meta, f, indent=2)

print("✅ Job models saved to models/job_*.onnx + models/job_meta.json")


# ═══════════════════════════════════════════════════════════════════════════
# CORPORATE AI ADOPTION DATASET  (mirrors Kaggle: hassangasem/corporate-ai-adoption…)
# ═══════════════════════════════════════════════════════════════════════════

N_CORP = 4000

CORP_INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Retail", "Manufacturing",
    "Energy", "Telecom", "Education", "Transportation", "Insurance",
    "Media", "Agriculture", "Real Estate", "Hospitality", "Legal",
]
CORP_COUNTRIES = [
    "USA", "UK", "Germany", "India", "China", "Japan", "Canada",
    "Australia", "France", "Brazil", "South Korea", "Netherlands",
    "Sweden", "Singapore", "UAE",
]

ADOPTION_LABELS = ["Low_Adoption", "Medium_Adoption", "High_Adoption"]

def make_corp_dataset():
    ind_arr  = np.random.choice(CORP_INDUSTRIES, N_CORP)
    ctry_arr = np.random.choice(CORP_COUNTRIES, N_CORP)

    # Tech sector tends toward higher investment
    tech_mask = ind_arr == "Technology"

    year = np.random.randint(2015, 2036, N_CORP)

    ai_investment_usd = np.where(
        tech_mask,
        np.random.uniform(500_000, 20_000_000, N_CORP),
        np.random.uniform(50_000, 10_000_000, N_CORP),
    )
    automation_rate          = np.random.uniform(0, 80, N_CORP)
    cost_savings             = ai_investment_usd * np.random.uniform(0.5, 3.5, N_CORP)
    revenue_impact           = ai_investment_usd * np.random.uniform(0.2, 4.0, N_CORP)
    productivity_gain        = np.random.uniform(0, 60, N_CORP)
    employee_ai_train_hrs    = np.random.uniform(0, 200, N_CORP)
    deployment_count         = np.random.randint(0, 50, N_CORP)

    le_ind  = LabelEncoder().fit(CORP_INDUSTRIES)
    le_ctry = LabelEncoder().fit(CORP_COUNTRIES)

    # Target: ai_maturity_score (0-100)
    log_inv = np.log1p(ai_investment_usd)
    log_inv_norm = (log_inv - log_inv.min()) / (log_inv.max() - log_inv.min()) * 100
    score = (
        0.30 * log_inv_norm
        + 0.20 * automation_rate
        + 0.20 * productivity_gain
        + 0.15 * (employee_ai_train_hrs / 2)
        + 0.10 * deployment_count * 2
        + 0.05 * (cost_savings / ai_investment_usd).clip(0, 5) * 10
        + np.random.normal(0, 4, N_CORP)
    )
    score = np.clip(score, 0, 100)

    adoption_cat = np.where(
        score > 66, "High_Adoption",
        np.where(score > 33, "Medium_Adoption", "Low_Adoption")
    )

    df = pd.DataFrame({
        "year":                       year,
        "ai_investment_usd":          ai_investment_usd,
        "automation_rate":            automation_rate,
        "cost_savings":               cost_savings,
        "revenue_impact":             revenue_impact,
        "productivity_gain":          productivity_gain,
        "employee_ai_training_hours": employee_ai_train_hrs,
        "deployment_count":           deployment_count.astype(float),
        "industry_enc":               le_ind.transform(ind_arr),
        "country_enc":                le_ctry.transform(ctry_arr),
        "ai_maturity_score":          score,
        "adoption_category":          adoption_cat,
    })
    return df, le_ind, le_ctry


print("\n── Generating corporate adoption data …")
df_corp, le_ind_c, le_ctry_c = make_corp_dataset()

FEATURE_COLS_CORP = [
    "year", "ai_investment_usd", "automation_rate", "cost_savings",
    "revenue_impact", "productivity_gain", "employee_ai_training_hours",
    "deployment_count", "industry_enc", "country_enc",
]

le_adopt = LabelEncoder().fit(ADOPTION_LABELS)
df_corp["adopt_idx"] = le_adopt.transform(df_corp["adoption_category"])

X_c = df_corp[FEATURE_COLS_CORP].values.astype(np.float32)
y_reg_c = df_corp["ai_maturity_score"].values.astype(np.float32)
y_clf_c = df_corp["adopt_idx"].values.astype(np.int32)

Xtr_c, Xte_c, yr_tr_c, yr_te_c, yc_tr_c, yc_te_c = train_test_split(
    X_c, y_reg_c, y_clf_c, test_size=0.2, random_state=42
)

print("── Training corporate regressor …")
reg_c = xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="rmse", random_state=42,
)
reg_c.fit(Xtr_c, yr_tr_c, eval_set=[(Xte_c, yr_te_c)], verbose=False)
rmse_c = mean_squared_error(yr_te_c, reg_c.predict(Xte_c)) ** 0.5
print(f"   RMSE={rmse_c:.2f}")

print("── Training corporate classifier …")
clf_c = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, num_class=3,
    eval_metric="mlogloss", random_state=42,
)
clf_c.fit(Xtr_c, yc_tr_c, eval_set=[(Xte_c, yc_te_c)], verbose=False)
acc_c = accuracy_score(yc_te_c, clf_c.predict(Xte_c))
print(f"   Accuracy={acc_c:.3f}")

# ── Convert corporate models to ONNX ───────────────────────────────────────
print("── Converting corporate models to ONNX …")
n_feat_c = len(FEATURE_COLS_CORP)
init_type_c = [("float_input", FloatTensorType([None, n_feat_c]))]

onnx_reg_c = convert_xgb(reg_c.get_booster(), initial_types=init_type_c, target_opset=12)
onnx_clf_c = convert_xgb(clf_c.get_booster(), initial_types=init_type_c, target_opset=12)

onnxmltools.utils.save_model(onnx_reg_c, "models/corp_reg.onnx")
onnxmltools.utils.save_model(onnx_clf_c, "models/corp_clf.onnx")

# ── Save corporate metadata ─────────────────────────────────────────────────
corp_meta = {
    "feature_cols": FEATURE_COLS_CORP,
    "adoption_labels": ADOPTION_LABELS,
    "label_encoders": {
        "industry": list(le_ind_c.classes_),
        "country":  list(le_ctry_c.classes_),
    },
}
with open("models/corp_meta.json", "w") as f:
    json.dump(corp_meta, f, indent=2)

print("✅ Corporate models saved to models/corp_*.onnx + models/corp_meta.json")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Final verification ──")
import os
for fname in ["models/job_reg.onnx","models/job_clf.onnx","models/job_meta.json",
              "models/corp_reg.onnx","models/corp_clf.onnx","models/corp_meta.json"]:
    size = os.path.getsize(fname) / 1024
    print(f"  {fname}: {size:.0f} KB")

print("\n🎉 All models ready — only onnxruntime needed at runtime (no xgboost/sklearn).")
