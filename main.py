
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
import json
import numpy as np
import onnxruntime as ort
import os
import subprocess
import tempfile
import shutil
from typing import Optional, List
from pathlib import Path

# ── Path Resolution (Task 11) ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AIvsHire API", version="2.0.0")
print("[STARTUP] FastAPI initialized")

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static libs (Firebase SDK served locally) ────────────────────────────────
lib_dir = BASE_DIR / "frontend" / "lib"
if lib_dir.is_dir():
    app.mount("/lib", StaticFiles(directory=str(lib_dir)), name="lib")
    print("[STARTUP] Frontend mounted (/lib)")
else:
    print(f"[STARTUP] WARNING: Frontend lib directory is missing at {lib_dir}")

index_path = BASE_DIR / "frontend" / "index.html"
if index_path.is_file():
    print("[STARTUP] frontend mounted (index.html)")
else:
    print(f"[STARTUP] WARNING: frontend/index.html is missing at {index_path}")

# ── Environment Variable Validation (Task 12) ───────────────────────────────
_gemini_key = os.getenv("GEMINI_API_KEY")
_google_api_key = os.getenv("GOOGLE_API_KEY")
_google_client_id = os.getenv("GOOGLE_CLIENT_ID")
_kaggle_username = os.getenv("KAGGLE_USERNAME")
_kaggle_key = os.getenv("KAGGLE_KEY")

if _gemini_key:
    print("[STARTUP] Gemini configured")
else:
    print("[STARTUP] WARNING: GEMINI_API_KEY is not configured")

if not _google_api_key:
    print("[STARTUP] WARNING: GOOGLE_API_KEY is not configured")
if not _google_client_id:
    print("[STARTUP] WARNING: GOOGLE_CLIENT_ID is not configured")
if not _kaggle_username:
    print("[STARTUP] WARNING: KAGGLE_USERNAME is not configured")
if not _kaggle_key:
    print("[STARTUP] WARNING: KAGGLE_KEY is not configured")

# ── Gemini (OpenAI compatibility) client (Task 3) ───────────────────────────
client = openai.OpenAI(
    api_key=_gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
) if _gemini_key else None

MODEL = "gemini-2.5-flash"


# ═══════════════════════════════════════════════════════════════════════════
# ONNX MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

class OnnxModelBundle:
    """Holds a regressor + classifier ONNX session pair plus metadata."""

    def __init__(self, reg_path: Path, clf_path: Path, meta_path: Path):
        self.reg   = ort.InferenceSession(str(reg_path),  providers=["CPUExecutionProvider"])
        self.clf   = ort.InferenceSession(str(clf_path),  providers=["CPUExecutionProvider"])
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_cols    = meta["feature_cols"]
        self.label_encoders  = meta["label_encoders"]    # {name: [class0, class1, ...]}
        self.risk_labels     = meta.get("risk_labels",     [])
        self.adoption_labels = meta.get("adoption_labels", [])
        self._enc_cache: dict[str, dict] = {
            col: {cls: i for i, cls in enumerate(classes)}
            for col, classes in self.label_encoders.items()
        }

    def encode(self, col: str, value: str) -> int:
        return self._enc_cache.get(col, {}).get(value, 0)

    def predict_reg(self, X: np.ndarray) -> float:
        inp = self.reg.get_inputs()[0].name
        return float(self.reg.run(None, {inp: X.astype(np.float32)})[0].flatten()[0])

    def predict_clf(self, X: np.ndarray) -> int:
        inp = self.clf.get_inputs()[0].name
        result = self.clf.run(None, {inp: X.astype(np.float32)})
        return int(result[0].flatten()[0])


# ── Production Startup Validation & Loading (Tasks 7, 8, 11 & 12) ────────────
job_bundle  = None
corp_bundle = None

# Validation existence of files
model_files = {
    "job_reg.onnx":  BASE_DIR / "models" / "job_reg.onnx",
    "job_clf.onnx":  BASE_DIR / "models" / "job_clf.onnx",
    "job_meta.json": BASE_DIR / "models" / "job_meta.json",
    "corp_reg.onnx":  BASE_DIR / "models" / "corp_reg.onnx",
    "corp_clf.onnx":  BASE_DIR / "models" / "corp_clf.onnx",
    "corp_meta.json": BASE_DIR / "models" / "corp_meta.json",
}

for name, path in model_files.items():
    if path.is_file():
        print(f"[STARTUP] {name} found")
    else:
        print(f"[STARTUP] WARNING: {name} is missing at {path}")

print("[STARTUP] Loading ONNX models")

try:
    if all(model_files[k].is_file() for k in ["job_reg.onnx", "job_clf.onnx", "job_meta.json"]):
        job_bundle = OnnxModelBundle(
            model_files["job_reg.onnx"],
            model_files["job_clf.onnx"],
            model_files["job_meta.json"],
        )
    else:
        print("[STARTUP] WARNING: Skipped loading Job models due to missing files")
except Exception as e:
    print(f"[STARTUP] ERROR: Failed to load job replacement models: {e}")

try:
    if all(model_files[k].is_file() for k in ["corp_reg.onnx", "corp_clf.onnx", "corp_meta.json"]):
        corp_bundle = OnnxModelBundle(
            model_files["corp_reg.onnx"],
            model_files["corp_clf.onnx"],
            model_files["corp_meta.json"],
        )
    else:
        print("[STARTUP] WARNING: Skipped loading Corporate models due to missing files")
except Exception as e:
    print(f"[STARTUP] ERROR: Failed to load corporate adoption models: {e}")

if job_bundle is not None and corp_bundle is not None:
    print("[STARTUP] Models loaded successfully")
else:
    print("[STARTUP] WARNING: Some or all ONNX models failed to load")


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class CompanyDetails(BaseModel):
    company_name: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    company_size: Optional[str] = None
    current_ai_investment_usd: Optional[float] = None
    current_training_hours: Optional[float] = None
    num_employees: Optional[int] = None
    avg_salary_usd: Optional[float] = None
    current_automation_rate: Optional[float] = None
    main_challenge: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    company_details: Optional[CompanyDetails] = None


class ROIComparisonRequest(BaseModel):
    company_details: CompanyDetails
    chat_history: list[Message]


class JobPredictRequest(BaseModel):
    job_role: str
    industry: str
    country: str
    year: int
    automation_risk_percent: float
    skill_gap_index: float
    salary_before_usd: float
    salary_after_usd: float
    salary_change_percent: float
    skill_demand_growth_percent: float
    remote_feasibility_score: float
    ai_adoption_level: float
    education_requirement_level: float
    skill_transition_pressure: float
    wage_volatility_index: float
    reskilling_urgency_score: float
    ai_disruption_intensity: float


class CorpPredictRequest(BaseModel):
    industry: str
    country: str
    year: int
    ai_investment_usd: float
    automation_rate: float
    cost_savings: float
    revenue_impact: float
    productivity_gain: float
    employee_ai_training_hours: float
    deployment_count: int


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

def build_system_prompt(company: Optional[CompanyDetails] = None) -> str:
    base = """You are AIvsHire, an AI-powered business advisor that helps companies answer one critical question:

"Should we invest more in AI tools OR in upskilling our employees — and which gives better ROI?"

You are talking to company decision-makers (CEOs, HR heads, operations managers).
Your job is to gather information about their company and give a clear, data-driven recommendation.

HOW YOU WORK:
- Ask smart questions one at a time about their industry, team size, budget, current challenges
- Understand what they are currently spending on AI vs employee training
- Understand their goals: cost reduction, productivity, growth, retention?
- Give a concrete recommendation: AI-first, Employee-first, or Hybrid approach
- Back your recommendation with numbers and reasoning

WHAT YOU COMPARE:
AI Investment side:
- Automation rate improvement
- Cost savings from replacing manual tasks
- Speed and scale gains
- Risk: job displacement, resistance, maintenance costs

Employee Training side:
- Skill gap reduction
- Salary impact (upskilled employees earn more but also produce more)
- Retention and loyalty
- Risk: time to ROI is slower, but more sustainable long-term

TONE: Direct, data-driven, no fluff. Like a McKinsey consultant but human.
Keep responses concise. Ask ONE question at a time.
Never give a recommendation until you have enough data."""

    if company and any(v is not None for v in company.model_dump().values()):
        lines = ["", "Company you're advising right now:"]
        if company.company_name:        lines.append(f"- Company: {company.company_name}")
        if company.industry:            lines.append(f"- Industry: {company.industry}")
        if company.country:             lines.append(f"- Country: {company.country}")
        if company.company_size:        lines.append(f"- Size: {company.company_size}")
        if company.num_employees:       lines.append(f"- Employees: {company.num_employees}")
        if company.avg_salary_usd:      lines.append(f"- Avg Salary: ${company.avg_salary_usd:,.0f} USD")
        if company.current_ai_investment_usd is not None:
            lines.append(f"- Current AI Investment: ${company.current_ai_investment_usd:,.0f} USD/year")
        if company.current_training_hours is not None:
            lines.append(f"- Training Hours/Employee: {company.current_training_hours} hrs/year")
        if company.current_automation_rate is not None:
            lines.append(f"- Automation Rate: {company.current_automation_rate}%")
        if company.main_challenge:      lines.append(f"- Main Challenge: {company.main_challenge}")
        lines.append("")
        lines.append("Use any info above that was provided. Ask for what's missing one question at a time.")
        base += "\n".join(lines)

    return base


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=FileResponse)
def root():
    index_path = BASE_DIR / "frontend" / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return FileResponse(str(index_path))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    fav_path = BASE_DIR / "frontend" / "favicon.ico"
    if not fav_path.is_file():
        raise HTTPException(status_code=404, detail="favicon.ico not found.")
    return FileResponse(str(fav_path))


@app.get("/api/firebase-config")
def firebase_config():
    return {
        "apiKey":            os.getenv("GOOGLE_API_KEY", ""),
        "authDomain":        "insider-fa38d.firebaseapp.com",
        "projectId":         "insider-fa38d",
        "storageBucket":     "insider-fa38d.firebasestorage.app",
        "messagingSenderId": "171312393255",
        "appId":             "1:171312393255:web:07d8e4081480162357f202",
        "measurementId":     "G-03ZP55MWBC",
        "clientId":          os.getenv("GOOGLE_CLIENT_ID", ""),
    }


@app.get("/health")
def health():
    index_path = BASE_DIR / "frontend" / "index.html"
    frontend_ok = index_path.is_file()
    models_ok = (job_bundle is not None) and (corp_bundle is not None)
    gemini_ok = os.getenv("GEMINI_API_KEY") is not None
    
    return {
        "status": "healthy" if (frontend_ok and models_ok) else "unhealthy",
        "frontend_loaded": frontend_ok,
        "models_loaded": models_ok,
        "gemini_configured": gemini_ok
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured.")
    system   = build_system_prompt(req.company_details)
    messages = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content} for m in req.messages
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages,
        )
        return {"reply": response.choices[0].message.content}
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="Gemini quota exceeded or rate limited. Please check your API key quota.")
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid Gemini API key.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured.")
    system   = build_system_prompt(req.company_details)
    messages = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content} for m in req.messages
    ]

    def generate():
        stream = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/predict/job")
def predict_job(req: JobPredictRequest):
    if job_bundle is None:
        raise HTTPException(status_code=503, detail="Job replacement models are not loaded/available. Please check server logs.")

    row = {
        "year":                         float(req.year),
        "automation_risk_percent":      req.automation_risk_percent,
        "skill_gap_index":              req.skill_gap_index,
        "salary_before_usd":            req.salary_before_usd,
        "salary_after_usd":             req.salary_after_usd,
        "salary_change_percent":        req.salary_change_percent,
        "skill_demand_growth_percent":  req.skill_demand_growth_percent,
        "remote_feasibility_score":     req.remote_feasibility_score,
        "ai_adoption_level":            req.ai_adoption_level,
        "education_requirement_level":  req.education_requirement_level,
        "skill_transition_pressure":    req.skill_transition_pressure,
        "wage_volatility_index":        req.wage_volatility_index,
        "reskilling_urgency_score":     req.reskilling_urgency_score,
        "ai_disruption_intensity":      req.ai_disruption_intensity,
        "job_role_enc":  float(job_bundle.encode("job_role", req.job_role)),
        "industry_enc":  float(job_bundle.encode("industry", req.industry)),
        "country_enc":   float(job_bundle.encode("country",  req.country)),
    }

    X = np.array([[row[col] for col in job_bundle.feature_cols]], dtype=np.float32)

    score    = round(job_bundle.predict_reg(X), 2)
    risk_idx = job_bundle.predict_clf(X)
    category = job_bundle.risk_labels[risk_idx]

    return {
        "ai_replacement_score": score,
        "risk_category":        category,
        "risk_idx":             risk_idx,
        "interpretation":       _interpret_job_risk(score, category),
        "valid_job_roles":      job_bundle.label_encoders["job_role"],
        "valid_industries":     job_bundle.label_encoders["industry"],
        "valid_countries":      job_bundle.label_encoders["country"],
    }


@app.post("/predict/corporate")
def predict_corporate(req: CorpPredictRequest):
    if corp_bundle is None:
        raise HTTPException(status_code=503, detail="Corporate adoption models are not loaded/available. Please check server logs.")

    row = {
        "year":                       float(req.year),
        "ai_investment_usd":          req.ai_investment_usd,
        "automation_rate":            req.automation_rate,
        "cost_savings":               req.cost_savings,
        "revenue_impact":             req.revenue_impact,
        "productivity_gain":          req.productivity_gain,
        "employee_ai_training_hours": req.employee_ai_training_hours,
        "deployment_count":           float(req.deployment_count),
        "industry_enc": float(corp_bundle.encode("industry", req.industry)),
        "country_enc":  float(corp_bundle.encode("country",  req.country)),
    }

    X = np.array([[row[col] for col in corp_bundle.feature_cols]], dtype=np.float32)

    score        = round(corp_bundle.predict_reg(X), 2)
    adoption_idx = corp_bundle.predict_clf(X)
    category     = corp_bundle.adoption_labels[adoption_idx]

    return {
        "ai_maturity_score":  score,
        "adoption_category":  category,
        "adoption_idx":       adoption_idx,
        "interpretation":     _interpret_corp_adoption(score, category),
        "valid_industries":   corp_bundle.label_encoders["industry"],
        "valid_countries":    corp_bundle.label_encoders["country"],
    }


@app.post("/compare/roi")
def compare_roi(req: ROIComparisonRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured.")

    system = """You are AIvsHire, an expert business ROI analyst.

Given a company's profile and their conversation, generate a structured ROI comparison report.

Output EXACTLY these sections:

## Company Snapshot
Brief summary of their current situation in 2-3 lines.

## If They Invest More in AI
- Estimated automation gain
- Cost savings potential
- Productivity impact
- Risks and downsides
- Timeline to see ROI

## If They Invest More in Employee Training
- Skill gap reduction
- Salary & retention impact
- Productivity improvement
- Risks and downsides
- Timeline to see ROI

## Head-to-Head Comparison
A simple table comparing both options across: Cost, ROI Timeline, Risk, Scalability, Sustainability

## Our Recommendation
Clear verdict: AI-First / Employee-First / Hybrid (50-50)
Why this recommendation specifically for their industry + country + size.
What they should do in the next 90 days — 3 concrete action steps.

## Watch Out For
2-3 risks specific to their situation they must plan for.

Keep numbers realistic. Be specific to their industry and country. No generic advice."""

    history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in req.chat_history)
    c = req.company_details

    user_prompt = f"""Company Profile:
Name: {c.company_name}
Industry: {c.industry} | Country: {c.country} | Size: {c.company_size}
Employees: {c.num_employees} | Avg Salary: ${c.avg_salary_usd:,.0f} USD
Current AI Investment: ${c.current_ai_investment_usd:,.0f} USD/year
Current Training Hours per Employee: {c.current_training_hours} hrs/year
Current Automation Rate: {c.current_automation_rate}%
Main Challenge: {c.main_challenge}

Conversation with company representative:
{history_text}

Generate the full ROI comparison report now."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )
    return {"roi_comparison": response.choices[0].message.content}


@app.post("/compare/roi/structured")
def compare_roi_structured(req: ROIComparisonRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured.")

    c = req.company_details
    history_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in req.chat_history)

    system = """You are an expert ROI data analyst. Based on the company profile and conversation,
return ONLY a valid JSON object (no markdown, no explanation) with these exact fields:
{
  "ai_roi_score": <number 0-100>,
  "training_roi_score": <number 0-100>,
  "ai_allocation": <recommended % to allocate to AI, 0-100>,
  "training_allocation": <recommended % to allocate to training, 0-100>,
  "ai_roi_months": <months until ROI for AI investment>,
  "training_roi_months": <months until ROI for training investment>,
  "ai_radar": [cost_efficiency, scalability, speed, sustainability, low_risk],
  "training_radar": [cost_efficiency, scalability, speed, sustainability, low_risk],
  "recommendation": "AI-First" | "Employee-First" | "Hybrid"
}
All radar values must be 0-100. ai_allocation + training_allocation must equal 100."""

    user_prompt = f"""Company: {c.company_name} | {c.industry} | {c.country} | {c.company_size}
Employees: {c.num_employees} | Avg Salary: ${c.avg_salary_usd:,.0f}
AI Investment: ${c.current_ai_investment_usd:,.0f}/yr | Training: {c.current_training_hours} hrs/employee/yr
Automation Rate: {c.current_automation_rate}% | Challenge: {c.main_challenge}

Conversation:
{history_text}

Return only the JSON object."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=400,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "ai_roi_score": 65, "training_roi_score": 70,
            "ai_allocation": 40, "training_allocation": 60,
            "ai_roi_months": 12, "training_roi_months": 18,
            "ai_radar": [75, 85, 80, 60, 55],
            "training_radar": [65, 60, 50, 85, 80],
            "recommendation": "Hybrid"
        }


# ═══════════════════════════════════════════════════════════════════════════
# SHAP / EXPLAIN ENDPOINTS  (perturbation-based feature attribution)
# ═══════════════════════════════════════════════════════════════════════════

# Baseline (reference) values used when masking a feature
JOB_BASELINES: dict[str, float] = {
    "year":                       2024.0,
    "automation_risk_percent":    50.0,
    "skill_gap_index":            50.0,
    "salary_before_usd":          60000.0,
    "salary_after_usd":           65000.0,
    "salary_change_percent":      8.0,
    "skill_demand_growth_percent": 5.0,
    "remote_feasibility_score":   5.0,
    "ai_adoption_level":          5.0,
    "education_requirement_level": 5.0,
    "skill_transition_pressure":  5.0,
    "wage_volatility_index":      5.0,
    "reskilling_urgency_score":   5.0,
    "ai_disruption_intensity":    5.0,
    "job_role_enc":               0.0,
    "industry_enc":               0.0,
    "country_enc":                0.0,
}

CORP_BASELINES: dict[str, float] = {
    "year":                       2024.0,
    "ai_investment_usd":          100000.0,
    "automation_rate":            20.0,
    "cost_savings":               50000.0,
    "revenue_impact":             80000.0,
    "productivity_gain":          10.0,
    "employee_ai_training_hours": 20.0,
    "deployment_count":           3.0,
    "industry_enc":               0.0,
    "country_enc":                0.0,
}

JOB_FEATURE_LABELS: dict[str, str] = {
    "year":                        "Year",
    "automation_risk_percent":     "Automation Risk %",
    "skill_gap_index":             "Skill Gap Index",
    "salary_before_usd":           "Salary Before (USD)",
    "salary_after_usd":            "Salary After (USD)",
    "salary_change_percent":       "Salary Change %",
    "skill_demand_growth_percent": "Skill Demand Growth %",
    "remote_feasibility_score":    "Remote Feasibility",
    "ai_adoption_level":           "AI Adoption Level",
    "education_requirement_level": "Education Level",
    "skill_transition_pressure":   "Skill Transition Pressure",
    "wage_volatility_index":       "Wage Volatility",
    "reskilling_urgency_score":    "Reskilling Urgency",
    "ai_disruption_intensity":     "AI Disruption Intensity",
    "job_role_enc":                "Job Role",
    "industry_enc":                "Industry",
    "country_enc":                 "Country",
}

CORP_FEATURE_LABELS: dict[str, str] = {
    "year":                       "Year",
    "ai_investment_usd":          "AI Investment (USD)",
    "automation_rate":            "Automation Rate %",
    "cost_savings":               "Cost Savings (USD)",
    "revenue_impact":             "Revenue Impact (USD)",
    "productivity_gain":          "Productivity Gain %",
    "employee_ai_training_hours": "Training Hours/Employee",
    "deployment_count":           "AI Deployments",
    "industry_enc":               "Industry",
    "country_enc":                "Country",
}


def _perturbation_shap(
    bundle: OnnxModelBundle,
    row: dict[str, float],
    baselines: dict[str, float],
    labels: dict[str, str],
) -> list[dict]:
    """
    Compute per-feature SHAP contributions via perturbation.
    For each feature i: contribution = predict(full) - predict(with feature i → baseline).
    Contributions are normalised so they sum to (base_score - baseline_score).
    """
    X_full = np.array([[row[col] for col in bundle.feature_cols]], dtype=np.float32)
    base_score = bundle.predict_reg(X_full)

    X_base = np.array(
        [[baselines.get(col, 0.0) for col in bundle.feature_cols]], dtype=np.float32
    )
    baseline_score = bundle.predict_reg(X_base)

    raw: list[dict] = []
    for i, col in enumerate(bundle.feature_cols):
        X_masked = X_full.copy()
        X_masked[0, i] = baselines.get(col, 0.0)
        score_without = bundle.predict_reg(X_masked)
        contribution = base_score - score_without
        raw.append({"col": col, "raw": contribution})

    total_raw = sum(abs(r["raw"]) for r in raw) or 1.0
    scale = (base_score - baseline_score) / total_raw if total_raw else 1.0

    contributions = []
    for r in raw:
        contrib = r["raw"] * scale
        contributions.append({
            "feature":      labels.get(r["col"], r["col"]),
            "col":          r["col"],
            "value":        round(float(row[r["col"]]), 4),
            "contribution": round(float(contrib), 4),
            "direction":    "positive" if contrib >= 0 else "negative",
        })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


@app.post("/explain/job")
def explain_job(req: JobPredictRequest):
    if job_bundle is None:
        raise HTTPException(status_code=503, detail="Job replacement models are not loaded/available. Please check server logs.")

    row = {
        "year":                         float(req.year),
        "automation_risk_percent":      req.automation_risk_percent,
        "skill_gap_index":              req.skill_gap_index,
        "salary_before_usd":            req.salary_before_usd,
        "salary_after_usd":             req.salary_after_usd,
        "salary_change_percent":        req.salary_change_percent,
        "skill_demand_growth_percent":  req.skill_demand_growth_percent,
        "remote_feasibility_score":     req.remote_feasibility_score,
        "ai_adoption_level":            req.ai_adoption_level,
        "education_requirement_level":  req.education_requirement_level,
        "skill_transition_pressure":    req.skill_transition_pressure,
        "wage_volatility_index":        req.wage_volatility_index,
        "reskilling_urgency_score":     req.reskilling_urgency_score,
        "ai_disruption_intensity":      req.ai_disruption_intensity,
        "job_role_enc":  float(job_bundle.encode("job_role", req.job_role)),
        "industry_enc":  float(job_bundle.encode("industry", req.industry)),
        "country_enc":   float(job_bundle.encode("country",  req.country)),
    }

    X = np.array([[row[col] for col in job_bundle.feature_cols]], dtype=np.float32)
    score    = round(job_bundle.predict_reg(X), 2)
    risk_idx = job_bundle.predict_clf(X)
    category = job_bundle.risk_labels[risk_idx]

    shap_values = _perturbation_shap(job_bundle, row, JOB_BASELINES, JOB_FEATURE_LABELS)

    return {
        "ai_replacement_score": score,
        "risk_category":        category,
        "interpretation":       _interpret_job_risk(score, category),
        "shap_values":          shap_values,
        "baseline_score":       round(
            job_bundle.predict_reg(
                np.array([[JOB_BASELINES.get(c, 0.0) for c in job_bundle.feature_cols]], dtype=np.float32)
            ), 2
        ),
    }


@app.post("/explain/corporate")
def explain_corporate(req: CorpPredictRequest):
    if corp_bundle is None:
        raise HTTPException(status_code=503, detail="Corporate adoption models are not loaded/available. Please check server logs.")

    row = {
        "year":                       float(req.year),
        "ai_investment_usd":          req.ai_investment_usd,
        "automation_rate":            req.automation_rate,
        "cost_savings":               req.cost_savings,
        "revenue_impact":             req.revenue_impact,
        "productivity_gain":          req.productivity_gain,
        "employee_ai_training_hours": req.employee_ai_training_hours,
        "deployment_count":           float(req.deployment_count),
        "industry_enc": float(corp_bundle.encode("industry", req.industry)),
        "country_enc":  float(corp_bundle.encode("country",  req.country)),
    }

    X = np.array([[row[col] for col in corp_bundle.feature_cols]], dtype=np.float32)
    score        = round(corp_bundle.predict_reg(X), 2)
    adoption_idx = corp_bundle.predict_clf(X)
    category     = corp_bundle.adoption_labels[adoption_idx]

    shap_values = _perturbation_shap(corp_bundle, row, CORP_BASELINES, CORP_FEATURE_LABELS)

    return {
        "ai_maturity_score": score,
        "adoption_category": category,
        "interpretation":    _interpret_corp_adoption(score, category),
        "shap_values":       shap_values,
        "baseline_score":    round(
            corp_bundle.predict_reg(
                np.array([[CORP_BASELINES.get(c, 0.0) for c in corp_bundle.feature_cols]], dtype=np.float32)
            ), 2
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# KAGGLE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

class DatasetFetchRequest(BaseModel):
    dataset: str  # e.g. "username/dataset-name"


def _setup_kaggle_env():
    """Write kaggle.json from env vars so the kaggle CLI works."""
    username = os.getenv("KAGGLE_USERNAME")
    key      = os.getenv("KAGGLE_KEY")
    if not username or not key:
        raise HTTPException(status_code=503, detail="KAGGLE_USERNAME or KAGGLE_KEY not configured.")
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    creds_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(creds_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(creds_path, 0o600)


@app.post("/datasets/fetch")
def fetch_dataset(req: DatasetFetchRequest):
    """Download a Kaggle dataset by 'owner/dataset-name' slug."""
    _setup_kaggle_env()
    dest = DATASETS_DIR / req.dataset.replace("/", "_")
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", req.dataset, "--unzip", "-p", str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=result.stderr.strip())
    files = []
    for root, _, filenames in os.walk(str(dest)):
        for fn in filenames:
            rel = os.path.relpath(os.path.join(root, fn), str(DATASETS_DIR))
            files.append(rel)
    return {
        "status":  "downloaded",
        "dataset": req.dataset,
        "path":    str(dest),
        "files":   files,
    }


@app.get("/datasets/list")
def list_datasets():
    """List all previously downloaded datasets."""
    if not DATASETS_DIR.is_dir():
        return {"datasets": []}
    entries = []
    for name in os.listdir(DATASETS_DIR):
        full = DATASETS_DIR / name
        if full.is_dir():
            files = []
            for root, _, filenames in os.walk(str(full)):
                for fn in filenames:
                    rel = os.path.relpath(os.path.join(root, fn), str(full))
                    files.append(rel)
            entries.append({"name": name, "files": files, "file_count": len(files)})
    return {"datasets": entries}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _interpret_job_risk(score: float, category: str) -> str:
    if category == "High" or score > 70:
        return "This role is at HIGH risk of AI replacement. Companies should consider redeployment or upskilling these employees before automating."
    elif category == "Medium" or score > 40:
        return "This role has MEDIUM risk. A hybrid approach works best — augment employees with AI tools rather than full replacement."
    else:
        return "This role has LOW AI replacement risk. Investing in employee training here gives better ROI than automation."


def _interpret_corp_adoption(score: float, category: str) -> str:
    if category == "High_Adoption":
        return "This company is AI-mature. Further AI investment will yield diminishing returns — focus now on employee upskilling to maximize AI utilization."
    elif category == "Medium_Adoption":
        return "Mid-stage AI adoption. A balanced investment in both AI tools AND training employees to use them will give the best ROI."
    else:
        return "Early-stage AI adoption. Start with targeted AI tools for high-impact areas, while training employees in parallel."


# ═══════════════════════════════════════════════════════════════════════════
# RUN (local dev only — deployment uses the deployConfig command)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5173, reload=True)
