# Data Cleaning Agent — Requirements

## 🎯 Objective
Build an OpenEnv-compatible RL environment where an agent cleans messy tabular data (CSV format). The environment evaluates the agent’s ability to transform raw datasets into clean, structured outputs.

---

## ✅ Functional Requirements

### 1. Real-World Task Simulation
- The environment must simulate realistic data cleaning workflows:
  - Handling missing values
  - Removing duplicates
  - Normalizing formats (dates, emails, etc.)
  - Fixing inconsistent schema

---

### 2. OpenEnv Spec Compliance
The environment must implement:

- `reset()` → returns initial observation  
- `step(action)` → returns (observation, reward, done, info)  
- `state()` → returns current state  

Also required:
- Typed Pydantic models:
  - Observation
  - Action
  - Reward
- `openenv.yaml` with metadata
- Must pass `openenv validate`

---

### 3. Minimum 3 Tasks (Difficulty Levels)

#### Easy Task
- Remove rows with missing/null values  
- Deterministic expected output  

#### Medium Task
- Normalize date formats (e.g., DD/MM/YYYY → YYYY-MM-DD)  
- Remove duplicate rows  

#### Hard Task
- Full dataset cleaning:
  - Schema normalization
  - Email validation
  - Column consistency
  - Mixed formatting fixes  

---

### 4. Graders (Critical)

Each task must include a grader that:
- Outputs a score between **0.0 – 1.0**
- Is deterministic and reproducible

#### Easy
- Exact match with ground truth dataset

#### Medium
- Partial scoring:
  - Correct date format
  - Duplicates removed

#### Hard
- Multi-factor scoring:
  - Schema correctness
  - Valid email structure
  - Data consistency

---

### 5. Reward Function
- Must provide **dense or semi-dense rewards**
- Reward partial progress (not just pass/fail)
- Penalize:
  - Invalid formats
  - Data loss
  - Incorrect transformations

---

### 6. Baseline Inference Script
- Must:
  - Call the environment API
  - Run all 3 tasks
  - Produce reproducible scores
- Uses `OPENAI_API_KEY` (optional LLM baseline)

---

## ⚙️ Non-Functional Requirements

### 1. Hugging Face Space Deployment
- Must run as a containerized HF Space
- Must respond to API calls

---

### 2. Docker Support
- Working `Dockerfile`
- Must pass:
  - `docker build`
  - `docker run`

---

### 3. Documentation
README must include:
- Project description & motivation
- Task descriptions (easy/medium/hard)
- Observation & Action format
- Reward design explanation
- Setup & run instructions
- Baseline results

---

### 4. Code Quality
- Clean project structure
- Modular design:
  - tasks/
  - graders/
  - utils/
- Readable and documented code

---

## 🚫 Disqualification Risks

- Environment does not deploy or respond
- Graders return constant score
- Missing baseline script
- No real-world applicability
- Tasks too trivial or unclear

---

## 🏆 Evaluation Alignment

This project is optimized for:

- Real-world utility (30%) ✅
- Task & grader quality (25%) ✅
- Environment design (20%) ✅
- Code quality & spec compliance (15%) ✅
- Creativity (10%) ✅