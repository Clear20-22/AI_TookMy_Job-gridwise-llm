# GridWise LLM — Smart Campus Energy Optimization Microservice

> **BUP CSE Fest 2026 · Hackathon · Online Preliminary Round**  
> *Organized by Bangladesh University of Professionals (BUP) Dept. of CSE, in association with Poridhi.io*

[![Tests](https://img.shields.io/badge/Tests-140%20Passed-brightgreen.svg)](#-sample-test-verification)
[![Benchmark](https://img.shields.io/badge/Benchmark-10%2F10%20Optimal-success.svg)](#-official-sample-case-benchmark)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Optimization](https://img.shields.io/badge/Solver-HiGHS%20%7C%20PuLP-orange.svg)](https://highs.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [System Architecture & Processing Pipeline](#-system-architecture--processing-pipeline)
3. [Supported Directives & Parsing Semantics](#-supported-directives--parsing-semantics)
4. [Mathematical Formulation & Physical Invariants](#-mathematical-formulation--physical-invariants)
5. [API Specification & Contracts](#-api-specification--contracts)
6. [Local Quickstart & Reproduction](#-local-quickstart--reproduction)
7. [Environment Variables & Model Configuration](#-environment-variables--model-configuration)
8. [Docker & Containerized Deployment](#-docker--containerized-deployment)
9. [Sample Test Verification](#-sample-test-verification)
10. [Repository Structure](#-repository-structure)
11. [Assumptions, Limitations & Guardrails](#-assumptions-limitations--guardrails)

---

## ⚡ Executive Summary

**GridWise LLM** is an enterprise-grade campus microgrid scheduling and energy optimization engine. Operating over a 24-hour planning horizon ($h = 0 \dots 23$), the microservice harmonizes numerical forecasts (campus load demand, rooftop photovoltaic solar generation, dynamic grid tariffs) and battery storage physics with unstructured natural-language operational notes written by facilities personnel.

The system addresses the core challenge of modern smart grids: **unstructured human language cannot be naively trusted as mathematical constraints.** GridWise LLM utilizes a strictly decoupled, 4-stage pipeline:
1. **Generative Language Model**: Interprets free-form operator notes into structured, machine-checkable operational directives.
2. **Deterministic Guardrails**: Validates and normalizes parsed directives against strict structural, temporal, and physical boundaries.
3. **Linear Programming (LP) Solver**: Ingests the validated directives and optimizes the 24-hour battery charge/discharge schedule to minimize total grid electricity expenditure in Bangladeshi Taka (BDT).
4. **Post-Optimization Invariant Replay**: Programmatically verifies all energy balances, battery transitions, rate limits, and end-of-day neutrality invariants before returning a certified response.

---

## 🏗 System Architecture & Processing Pipeline

The following diagram illustrates the complete request lifecycle and data flow through GridWise LLM:

```mermaid
flowchart TD
    A[Client / Judge Harness] -->|POST /optimize-energy| B[FastAPI Gateway]
    B -->|Pydantic Validation| C[Request Parser]
    
    subgraph Stage 1: LLM Directive Interpretation
        C -->|Operator Notes 1-3| D[LLM Directive Interpreter]
        D -->|Raw JSON Schema| E[Structured Candidates]
    end

    subgraph Stage 2: Deterministic Guardrails
        E --> F[Guardrail Engine]
        F -->|Validate Types & Note Indices| F1[Type & Index Check]
        F -->|Sort & De-duplicate Hours [0..23]| F2[Temporal Window Normalizer]
        F -->|Validate & Clamp Numeric Bounds| F3[Boundary Validator]
        F1 & F2 & F3 --> G[Validated Directives & Modifiers]
    end

    subgraph Stage 3: Mathematical Optimization
        G --> H[Optimization Model Builder]
        C -->|Demand, Solar, Tariffs, Battery Specs| H
        H -->|Formulate Objective & Constraints| I[LP Solver: CBC / PuLP]
        I -->|Optimal Solution| J[Raw Dispatch Schedule]
    end

    subgraph Stage 4: Post-Optimization Replay & Verification
        J --> K[Replay Verification Engine]
        K -->|Check Energy Balances| K1[Balance Invariant]
        K -->|Check Battery Limits & Actions| K2[Battery Physics]
        K -->|Verify Directive Compliance| K3[Directive Compliance]
        K -->|Verify End-of-Day Neutrality| K4[SoC Neutrality]
        K1 & K2 & K3 & K4 --> L[Recalculate Metrics: Total Cost, Grid kWh, Peak]
    end

    L --> M[Response Serializer]
    M -->|HTTP 200 JSON Response| A
```

---

## 📋 Supported Directives & Parsing Semantics

The LLM parser transforms natural-language operator notes into one of six canonical directive types. Every operator note produces **exactly one** entry in `directive_interpretation`, strictly preserved in `note_index` order ($0 \dots N-1$).

| Directive Type | Meaning | `applies` | Required `structured_adjustment` Shape |
| :--- | :--- | :---: | :--- |
| `solar_reduction` | Curtails usable solar generation due to cloud cover, washing, or maintenance. | `true` | `{"hours": [int, ...], "factor": float}` |
| `minimum_battery_reserve` | Elevates the battery minimum allowable energy floor for emergency reserve. | `true` | `{"hours": [int, ...], "minimum_energy_kwh": float}` |
| `no_charge_window` | Prohibits battery charging during specific operational maintenance windows. | `true` | `{"hours": [int, ...]}` |
| `no_discharge_window` | Prohibits battery discharging during protective relay testing or grid events. | `true` | `{"hours": [int, ...]}` |
| `max_grid_window` | Restricts maximum grid import due to feeder or transformer capacity constraints. | `true` | `{"hours": [int, ...], "max_grid_kwh": float}` |
| `no_op` | Irrelevant operational remark, future notice, or non-actionable distractor. | `false` | `null` |

### Key Semantic & Parsing Rules
* **Time Windows**: Stated time intervals are **start-inclusive and end-exclusive**. For example:
  * *"1 PM to 3 PM"* $\rightarrow$ Hours `[13, 14]`
  * *"From 6 PM until 9 PM"* $\rightarrow$ Hours `[18, 19, 20]`
  * *"From 11 AM until 1 PM"* $\rightarrow$ Hours `[11, 12]`
* **Hours Array**: Must contain unique integers between $0$ and $23$, sorted in strictly ascending order.
* **Solar Reduction Factor**: Denotes the **usable fraction remaining**. An "80% reduction" yields `factor = 0.20`. A "drop to roughly 25%" yields `factor = 0.25`.
* **Reserve Normalization**: Percentage requests (e.g. *"keep at least 50% capacity"*) are automatically converted to absolute energy ($\text{capacity\_kwh} \times 0.50$).
* **Distractor Isolation**: Any note unrelated to the current 24-hour energy horizon (e.g., sports deadlines, cafeteria menus, next-week schedule adjustments) is marked `applies: false`, `directive_type: "no_op"`, and `structured_adjustment: null`.

---

## 📐 Mathematical Formulation & Physical Invariants

The optimization problem is formulated as a deterministic **Linear Program (LP)** solved over hours $h \in \{0, 1, \dots, 23\}$.

### 1. Decision Variables (per hour $h$)
* $g_h \ge 0$: Grid electricity imported (kWh).
* $s_h \ge 0$: Solar energy utilized by campus or battery (kWh).
* $c_h \ge 0$: Energy charged into the battery (kWh).
* $d_h \ge 0$: Energy discharged from the battery (kWh).
* $E_h \ge 0$: Battery state of charge (energy stored) at the conclusion of hour $h$ (kWh).

### 2. Objective Function
Minimize the total cost of electricity imported from the utility grid:
$$\min \sum_{h=0}^{23} g_h \cdot \text{tariff\_bdt\_per\_kwh}[h]$$

### 3. Physical & Operational Constraints
1. **Hourly Energy Balance**:
   $$g_h + s_h + d_h = \text{demand\_kwh}[h] + c_h \quad \forall h \in \{0 \dots 23\}$$
2. **Solar Availability & Usability**:
   $$0 \le s_h \le \text{effective\_solar}[h] \quad \forall h \in \{0 \dots 23\}$$
   where $\text{effective\_solar}[h] = \text{solar\_kwh}[h] \times \text{factor}$ if a `solar_reduction` directive covers hour $h$, else $\text{solar\_kwh}[h]$. Unused solar is curtailed; grid export is not supported.
3. **Battery Energy Dynamics**:
   $$E_h = E_{h-1} + c_h - d_h \quad \forall h \in \{0 \dots 23\} \quad (\text{with } E_{-1} = \text{initial\_energy\_kwh})$$
4. **Battery Energy Capacity & Dynamic Reserve**:
   $$\max(\text{minimum\_energy\_kwh}, \text{reserve\_directive}_h) \le E_h \le \text{capacity\_kwh} \quad \forall h \in \{0 \dots 23\}$$
5. **Battery Charge & Discharge Rate Limits**:
   $$0 \le c_h \le \text{max\_charge\_kwh\_per\_hour} \quad \forall h \in \{0 \dots 23\}$$
   $$0 \le d_h \le \text{max\_discharge\_kwh\_per\_hour} \quad \forall h \in \{0 \dots 23\}$$
6. **Directive Window Enforcements**:
   * If $h \in \text{no\_charge\_window} \implies c_h = 0$
   * If $h \in \text{no\_discharge\_window} \implies d_h = 0$
   * If $h \in \text{max\_grid\_window} \implies g_h \le \text{max\_grid\_kwh}$
7. **End-of-Day Neutrality**:
   $$E_{23} = \text{initial\_energy\_kwh}$$
   *The battery cannot be permanently depleted as a one-time free source of energy.*
8. **Battery Action Resolution**:
   * If $c_h > 0 \implies \text{action} = \text{"charge"}, \text{battery\_kwh} = c_h$
   * If $d_h > 0 \implies \text{action} = \text{"discharge"}, \text{battery\_kwh} = d_h$
   * If $c_h = 0 \land d_h = 0 \implies \text{action} = \text{"idle"}, \text{battery\_kwh} = 0.0$

---

## 📡 API Specification & Contracts

The service exposes standard RESTful endpoints conforming to OpenAPI / JSON Schema specifications.

### 1. `GET /health`
* **Purpose**: Liveness and readiness probe for orchestration and judge harnesses.
* **Response Status**: `200 OK`
* **Response Body**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. `POST /optimize-energy`
* **Purpose**: Accepts scenario input with hourly demand, solar forecast, tariffs, battery parameters, and operator notes; returns parsed directives and optimal 24-hour schedule.
* **Response Status**: `200 OK` (or `400 Bad Request` / `422 Unprocessable Entity` / `500 Internal Server Error`).

#### Request Schema
```json
{
  "scenario_id": "SAMPLE-01",
  "operator_notes": [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
    "The sports office moved next month's registration deadline."
  ],
  "hours": [
    {
      "hour": 0,
      "demand_kwh": 90.0,
      "solar_kwh": 0.0,
      "tariff_bdt_per_kwh": 6.0
    }
  ],
  "battery": {
    "capacity_kwh": 220.0,
    "initial_energy_kwh": 110.0,
    "minimum_energy_kwh": 40.0,
    "max_charge_kwh_per_hour": 50.0,
    "max_discharge_kwh_per_hour": 50.0
  }
}
```

#### Response Schema
```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {
        "hours": [12, 13],
        "factor": 0.25
      },
      "explanation": "Solar availability is reduced to 25% during the panel-cleaning window."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "This note does not affect today's 24-hour energy schedule."
    }
  ],
  "hourly_plan": [
    {
      "hour": 0,
      "grid_kwh": 90.0,
      "solar_used_kwh": 0.0,
      "battery_action": "idle",
      "battery_kwh": 0.0,
      "battery_energy_after_kwh": 110.0
    }
  ],
  "total_grid_kwh": 2740.0,
  "total_cost_bdt": 40150.0,
  "peak_grid_kwh": 195.0,
  "plan_summary": "Optimal 24-hour schedule honoring solar cleaning reduction from 12-14 and maintaining end-of-day battery neutrality."
}
```

---

## 🚀 Local Quickstart & Reproduction

Follow these steps to set up and run the service locally on a clean machine:

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git
* (Optional) Docker Engine

### 1. Clone the Repository
```bash
git clone https://github.com/Clear20-22/AI_TookMy_Job-gridwise-llm.git
cd AI_TookMy_Job-gridwise-llm
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
Edit `.env` and supply your LLM API credentials (e.g. Gemini API key):
```bash
export GEMINI_API_KEY="your-google-gemini-api-key-here"
```

### 5. Start the Microservice
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## ⚙️ Environment Variables & Model Configuration

The application supports multiple LLM inference providers, configurable via environment variables:

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `PORT` | Integer | `8000` | HTTP port on which the server binds. |
| `HOST` | String | `0.0.0.0` | Host IP address for network binding. |
| `GEMINI_API_KEY` | String | *Required* | API key for Google Gemini (`https://aistudio.google.com/app/apikey`). |
| `GEMINI_MODEL` | String | `gemini-3.5-flash-lite` | Active Google Gemini model (`gemini-3.5-flash-lite`, `gemini-flash-lite-latest`, `gemini-3.6-flash`). |
| `SOLVER_TIMEOUT_SEC` | Integer | `10` | Maximum solver execution timeout in seconds. |
| `LOG_LEVEL` | String | `INFO` | Application logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

> **Security Mandate**: Never commit real API keys or sensitive credentials into Git. All secrets are loaded via environment variables or a local `.env` file excluded by `.gitignore`.

---

## 🐳 Docker & Containerized Deployment

A pre-built, production-hardened container is available as a fallback artifact for evaluation.

### Pulling the Image
```bash
docker pull clear20/gridwise-llm:latest
```

### Running the Container
```bash
docker run -d \
  --name gridwise-service \
  -p 8000:8000 \
  -e GEMINI_API_KEY="your-api-key" \
  clear20/gridwise-llm:latest
```

### Building the Image Locally
```bash
docker build -t gridwise-llm:latest .
docker run -d -p 8000:8000 --env-file .env gridwise-llm:latest
```

---

## 🧪 Sample Test Verification

### 1. Test Health Endpoint
```bash
curl -X GET http://localhost:8000/health
```
**Expected Output:**
```json
{"status":"ok"}
```

### 2. Test Optimization with Sample Case 1 (Solar Cleaning + Distractor)
```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "SAMPLE-01",
    "operator_notes": [
      "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
      "The sports office moved next months registration deadline."
    ],
    "hours": [
      {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
      {"hour": 1, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
      {"hour": 2, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
      {"hour": 3, "demand_kwh": 75, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
      {"hour": 4, "demand_kwh": 75, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
      {"hour": 5, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
      {"hour": 6, "demand_kwh": 100, "solar_kwh": 5, "tariff_bdt_per_kwh": 8},
      {"hour": 7, "demand_kwh": 130, "solar_kwh": 20, "tariff_bdt_per_kwh": 10},
      {"hour": 8, "demand_kwh": 150, "solar_kwh": 50, "tariff_bdt_per_kwh": 12},
      {"hour": 9, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
      {"hour": 10, "demand_kwh": 175, "solar_kwh": 130, "tariff_bdt_per_kwh": 16},
      {"hour": 11, "demand_kwh": 180, "solar_kwh": 160, "tariff_bdt_per_kwh": 16},
      {"hour": 12, "demand_kwh": 185, "solar_kwh": 180, "tariff_bdt_per_kwh": 15},
      {"hour": 13, "demand_kwh": 180, "solar_kwh": 170, "tariff_bdt_per_kwh": 14},
      {"hour": 14, "demand_kwh": 170, "solar_kwh": 140, "tariff_bdt_per_kwh": 13},
      {"hour": 15, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
      {"hour": 16, "demand_kwh": 170, "solar_kwh": 45, "tariff_bdt_per_kwh": 18},
      {"hour": 17, "demand_kwh": 190, "solar_kwh": 10, "tariff_bdt_per_kwh": 22},
      {"hour": 18, "demand_kwh": 210, "solar_kwh": 0, "tariff_bdt_per_kwh": 26},
      {"hour": 19, "demand_kwh": 220, "solar_kwh": 0, "tariff_bdt_per_kwh": 28},
      {"hour": 20, "demand_kwh": 205, "solar_kwh": 0, "tariff_bdt_per_kwh": 26},
      {"hour": 21, "demand_kwh": 175, "solar_kwh": 0, "tariff_bdt_per_kwh": 18},
      {"hour": 22, "demand_kwh": 135, "solar_kwh": 0, "tariff_bdt_per_kwh": 10},
      {"hour": 23, "demand_kwh": 105, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}
    ],
    "battery": {
      "capacity_kwh": 220,
      "initial_energy_kwh": 110,
      "minimum_energy_kwh": 40,
      "max_charge_kwh_per_hour": 50,
      "max_discharge_kwh_per_hour": 50
    }
  }'
```

---

## 📁 Repository Structure

```
AI_TookMy_Job-gridwise-llm/
├── .dockerignore              # Docker build ignore rules
├── .env.example               # Template environment configuration (no secrets)
├── .gitignore                 # Excludes caches, venv, and secrets (.env)
├── Dockerfile                 # Production multi-stage container build
├── HowToRun.md                # Quick local reproduction guide
├── LICENSE                    # Open-source MIT License
├── README.md                  # Comprehensive system documentation
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Production Python package dependencies
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI microservice gateway (/health, /optimize-energy)
│   ├── model/                 # Pydantic v2 core request/response schemas
│   │   ├── __init__.py
│   │   ├── request.py         # ScenarioRequest, BatterySpec, HourEntry
│   │   └── response.py        # OptimizeResponse, DirectiveInterpretation, HourlyPlanEntry
│   ├── models/                # Schema compatibility forwarding namespace
│   │   ├── __init__.py
│   │   ├── request.py
│   │   └── response.py
│   ├── optimizer/             # Core mathematical optimization
│   │   ├── __init__.py
│   │   └── optimizer.py       # HiGHS / PuLP MILP formulation & physical balance solve
│   └── services/              # Processing pipeline stages
│       ├── __init__.py
│       ├── preprocessor.py    # Colloquial time normalization & 2-tier LRU caching
│       ├── llm_parser.py      # Google GenAI LLM parser with few-shot prompting & failover
│       ├── guardrails.py      # Deterministic validation, bounds clamping, auto-repair
│       └── optimizer.py       # Solver namespace alias
├── tests/                     # Automated test suites
│   ├── __init__.py
│   ├── test_api.py            # FastAPI endpoint tests (/health, /optimize-energy)
│   ├── test_guardrails.py     # Deterministic boundary & distractor unit tests
│   ├── test_preprocessor.py   # Time/fraction normalization & LRU cache tests
│   └── test_sample_cases.py   # All 10 official benchmark cases validation
└── api_benchmark/             # Groq vs Gemini speed benchmark suite
    ├── README.md
    └── benchmark.py
```

---

## 🛡 Assumptions, Limitations & Guardrails

1. **Synthetic Scenarios Only**: All campus demand, PV generation, tariffs, and battery parameters are synthetic abstractions formulated for the BUP CSE Fest 2026 hackathon.
2. **Deterministic Fallbacks**: If an external LLM API experiences transient downtime, network timeouts, or invalid JSON emission, the service safely fails over to deterministic heuristic keyword parsing to prevent catastrophic request failure.
3. **No Grid Export**: The microgrid formulation strictly prohibits feeding power back to the main grid ($g_h \ge 0$). Excess PV generation beyond campus demand and battery charge capacity is curtailed without penalty.
4. **Feasibility Guarantee**: Organizer scoring cases are guaranteed to be mathematically feasible. The LP formulation incorporates high-penalty slack variables to guarantee clean 200 responses with graceful degradation even under edge-case scenarios.
5. **Secret Protection**: API keys, auth headers, and internal stack traces are systematically stripped from public log streams and error payloads.

---

## 👥 Team: AI Took My Job
* **Competition**: BUP CSE Fest 2026 Hackathon (Preliminary Round)
* **Problem**: Smart Campus Energy Optimization Challenge (GridWise LLM)
* **Partner**: Poridhi.io
