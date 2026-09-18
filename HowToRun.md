# GridWise LLM — How to Run

## 1. Local Setup

### Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Environment Configuration (Optional)
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY="your-gemini-api-key"
```
> Note: If `GEMINI_API_KEY` is not provided or the Gemini service is unreachable, the system gracefully falls back to deterministic `no_op` interpretation without crashing.

---

## 2. Running the API Server

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- API Docs: http://localhost:8000/docs
- Health Check: `GET http://localhost:8000/health`
- Energy Optimization Endpoint: `POST http://localhost:8000/optimize-energy`

---

## 3. Running Automated Tests

### Run all tests
```powershell
pytest -v
```

### Run specific test suites
- **Optimizer Core Tests**: `pytest tests/test_optimizer.py -v`
- **Sample Cases Validation (All 10 benchmark cases)**: `pytest tests/test_sample_cases.py -v`
- **Guardrail Validator Tests**: `pytest tests/test_guardrails.py -v`
- **Preprocessor Tests**: `pytest tests/test_preprocessor.py -v`
- **API Endpoint Tests**: `pytest tests/test_api.py -v`

---

## 4. Docker Deployment

### Pull and Run from Docker Hub
```bash
docker pull s0jib/gridwise-llm:latest
docker run -d -p 8000:8000 -e GEMINI_API_KEY="your-api-key" s0jib/gridwise-llm:latest
```

### Or Build Locally
```bash
docker build -t s0jib/gridwise-llm:latest .
docker run -d -p 8000:8000 -e GEMINI_API_KEY="your-api-key" s0jib/gridwise-llm:latest
```