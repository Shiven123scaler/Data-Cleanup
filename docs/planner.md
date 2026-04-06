# 3-Day Build Plan — Data Cleaning Agent

## 👥 Team
- Harman → Backend + Environment + Deployment
- Shiven → Tasks + Graders + Data + Testing

---

# 🗓️ Day 1 — Core Setup + Easy Task

## 🎯 Goal:
Get a working OpenEnv environment with 1 task (easy)

---

## 👨‍💻 Harman
- Setup project structure
- Implement:
  - `env.py` (reset, step, state)
  - `main.py` (FastAPI server)
  - `models.py` (Pydantic schemas)
- Create `openenv.yaml`
- Test local API (`/reset`, `/step`)

✅ Deliverable:
- Working environment with dummy logic

---

## 👨‍💻 Shiven
- Create dataset:
  - `raw_easy.csv`
  - `clean_easy.csv`
- Implement:
  - `task_easy.py`
  - `grader_easy.py` (exact match using pandas)
- Validate grader correctness

✅ Deliverable:
- Easy task fully working

---

## 🔁 End of Day Check
- Can run:
  - `reset()`
  - `step()`
- Easy task returns correct reward

---

# 🗓️ Day 2 — Medium + Hard Tasks + Graders

## 🎯 Goal:
Complete all tasks and grading system

---

## 👨‍💻 Harman
- Improve environment:
  - Add task randomization
  - Improve observation format
- Implement:
  - proper reward flow
- Start Docker setup

---

## 👨‍💻 Shiven
- Create datasets:
  - medium + hard
- Implement:
  - `task_medium.py`, `task_hard.py`
  - `grader_medium.py` (partial scoring)
  - `grader_hard.py` (multi-factor scoring)

---

## 🤝 Joint Work
- Integrate tasks into environment
- Run multiple test episodes
- Ensure:
  - rewards vary properly
  - graders are NOT constant

---

## 🔁 End of Day Check
- All 3 tasks working
- Rewards between 0–1
- No crashes

---

# 🗓️ Day 3 — Polish + Deployment + Baseline

## 🎯 Goal:
Make it submission-ready

---

## 👨‍💻 Harman
- Finalize:
  - Dockerfile
  - HF Spaces deployment
- Run:
  - `docker build`
  - `docker run`
- Ensure API responds

---

## 👨‍💻 Shiven
- Implement:
  - `baseline/run_baseline.py`
- Write README:
  - tasks
  - reward design
  - usage
- Add examples

---

## 🤝 Joint Work
- Run full pipeline:
  - baseline → scores
- Fix bugs
- Test reproducibility

---

## 🔁 Final Checklist
- ✅ openenv validate passes
- ✅ HF Space runs
- ✅ baseline works
- ✅ 3 tasks complete
- ✅ graders deterministic
- ✅ README complete

---

# ⚡ Stretch Goals (if time permits)

- Add better reward shaping (row-level scoring)
- Improve grader robustness (schema-aware)
- Add visualization (before/after table)

---

# 🏁 Final Deliverable

- GitHub repo
- HF Space deployment
- Working OpenEnv environment
- Baseline results