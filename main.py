
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
import json
import numpy as np
import onnxruntime as ort
import os
from typing import Optional

app = FastAPI(title="AIvsHire API", version="2.0.0")

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Anthropic client ────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ═══════════════════════════════════════════════════════════════════════════
# ONNX MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

class OnnxModelBundle:
    """Holds a regressor + classifier ONNX session pair plus metadata."""

    def __init__(self, reg_path: str, clf_path: str, meta_path: str):
        self.reg   = ort.InferenceSession(reg_path,  providers=["CPUExecutionProvider"])
        self.clf   = ort.InferenceSession(clf_path,  providers=["CPUExecutionProvider"])
        with open(meta_path) as f:
            meta = json.load(f)
        self.feature_cols    = meta["feature_cols"]
        self.label_encoders  = meta["label_encoders"]    # {name: [class0, class1, ...]}
        # label lists (regressor target names differ between bundles)
        self.risk_labels     = meta.get("risk_labels",     [])
        self.adoption_labels = meta.get("adoption_labels", [])
        # fast reverse-lookup dicts
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
        # onnxmltools classifier outputs: [labels, probabilities]
        result = self.clf.run(None, {inp: X.astype(np.float32)})
        return int(result[0].flatten()[0])


# ── Load models at startup ───────────────────────────────────────────────────
job_bundle  = None
corp_bundle = None

try:
    job_bundle = OnnxModelBundle(
        "models/job_reg.onnx",
        "models/job_clf.onnx",
        "models/job_meta.json",
    )
    print("✅ Job replacement models loaded")
    print("   features:", job_bundle.feature_cols)
except FileNotFoundError:
    print("⚠️  models/job_*.onnx not found — /predict/job will return 503")
except Exception as e:
    print(f"⚠️  Could not load job models: {e}")

try:
    corp_bundle = OnnxModelBundle(
        "models/corp_reg.onnx",
        "models/corp_clf.onnx",
        "models/corp_meta.json",
    )
    print("✅ Corporate adoption models loaded")
    print("   features:", corp_bundle.feature_cols)
except FileNotFoundError:
    print("⚠️  models/corp_*.onnx not found — /predict/corporate will return 503")
except Exception as e:
    print(f"⚠️  Could not load corp models: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class CompanyDetails(BaseModel):
    company_name: str
    industry: str
    country: str
    company_size: str
    current_ai_investment_usd: float
    current_training_hours: float
    num_employees: int
    avg_salary_usd: float
    current_automation_rate: float
    main_challenge: str


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

    if company:
        base += f"""

Company you're advising right now:
- Company: {company.company_name}
- Industry: {company.industry}, Country: {company.country}
- Size: {company.company_size} ({company.num_employees} employees)
- Avg Salary: ${company.avg_salary_usd:,.0f} USD
- Current AI Investment: ${company.current_ai_investment_usd:,.0f} USD/year
- Current Training Hours/Employee: {company.current_training_hours} hrs/year
- Current Automation Rate: {company.current_automation_rate}%
- Main Challenge: {company.main_challenge}

You already have their basic details. Don't ask for them again.
Start by acknowledging their situation and asking deeper questions about their specific pain points and goals."""

    return base


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "AIvsHire API is running 🚀",
        "version": "2.0.0",
        "purpose": "AI vs Employee ROI Comparison for Companies",
        "ml_predictions": {
            "job_model":  job_bundle  is not None,
            "corp_model": corp_bundle is not None,
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "job_models_loaded":  job_bundle  is not None,
        "corp_models_loaded": corp_bundle is not None,
        "job_features":  job_bundle.feature_cols  if job_bundle  else [],
        "corp_features": corp_bundle.feature_cols if corp_bundle else [],
    }


@app.post("/chat")
def chat(req: ChatRequest):
    system   = build_system_prompt(req.company_details)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return {"reply": response.content[0].text}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    system   = build_system_prompt(req.company_details)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    def generate():
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
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
        raise HTTPException(status_code=503, detail="Job models not loaded.")

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
        raise HTTPException(status_code=503, detail="Corporate models not loaded.")

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
    system = """You are AIvsHire, an expert business ROI analyst.

Given a company's profile and their conversation, generate a structured ROI comparison report.

Output EXACTLY these sections:

## 📊 Company Snapshot
Brief summary of their current situation in 2-3 lines.

## 🤖 If They Invest More in AI
- Estimated automation gain
- Cost savings potential
- Productivity impact
- Risks and downsides
- Timeline to see ROI

## 👥 If They Invest More in Employee Training
- Skill gap reduction
- Salary & retention impact
- Productivity improvement
- Risks and downsides  
- Timeline to see ROI

## ⚖️ Head-to-Head Comparison
A simple table comparing both options across: Cost, ROI Timeline, Risk, Scalability, Sustainability

## 🏆 Our Recommendation
Clear verdict: AI-First / Employee-First / Hybrid (50-50)
Why this recommendation specifically for their industry + country + size.
What they should do in the next 90 days — 3 concrete action steps.

## ⚠️ Watch Out For
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

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return {"roi_comparison": response.content[0].text}


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
# RUN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
