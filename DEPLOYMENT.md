# Deployment & Production Hardening Guide

This document outlines the steps to run and deploy the **AIvsHire** application on Render (or any other hosting platform) and set up local development.

---

## 1. Required Environment Variables

Ensure these environment variables are set in your deployment environment (e.g., Render Dashboard under Environment Variables):

| Variable Name | Description | Example / Source |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API Key | `AIzaSy...` (from Google AI Studio) |
| `GOOGLE_API_KEY` | Firebase API Key | `AIzaSy...` (from Firebase Console Settings) |
| `GOOGLE_CLIENT_ID` | Google OAuth Client ID | `171312393255-...apps.googleusercontent.com` |
| `KAGGLE_USERNAME` | Kaggle Account Username | Used for fetching datasets |
| `KAGGLE_KEY` | Kaggle API Token Key | Generated from Kaggle Account Settings |

---

## 2. Render Deployment Settings

### Build Settings
* **Runtime:** Python 3.11.11 (specified via `runtime.txt`)
* **Build Command:**
  ```bash
  pip install -r requirements.txt
  ```

### Run Settings
* **Start Command:**
  ```bash
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```

---

## 3. Health Checks & Verification

FastAPI provides an enhanced health check endpoint used to verify deployment status.

* **Health Check Endpoint:** `/health`
* **Expected Response:**
  ```json
  {
    "status": "healthy",
    "frontend_loaded": true,
    "models_loaded": true,
    "gemini_configured": true
  }
  ```

### What gets checked:
1. **Frontend:** Checks if `frontend/index.html` exists in the file system.
2. **Models:** Checks if all ONNX models (`job_reg.onnx`, `job_clf.onnx`, `corp_reg.onnx`, `corp_clf.onnx`) and JSON metadata files exist and are loaded successfully.
3. **Gemini:** Checks if `GEMINI_API_KEY` environment variable is successfully configured.

---

## 4. Local Development

To run the application locally on Windows or Linux:

1. **Install Dependencies:**
   Ensure you are using Python 3.11:
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup Environment Variables:**
   Copy `.env.example` to `.env` and fill in the values:
   ```bash
   cp .env.example .env
   ```
3. **Run Dev Server:**
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

---

## 5. Troubleshooting & Common Issues

* **`metadata-generation-failed` / `pydantic-core` build failure on Render:**
  * **Cause:** Building with an incompatible Python version.
  * **Resolution:** Ensure `runtime.txt` pins the Python version to `python-3.11.11` and `requirements.txt` pins `pydantic==2.7.1`.
* **Model Loading Errors:**
  * If models are missing or corrupted, the application will not crash at startup. It will log a warning `[STARTUP] WARNING: <filename> is missing` and return `503 Service Unavailable` for model prediction routes. Exposes status in `/health`.
