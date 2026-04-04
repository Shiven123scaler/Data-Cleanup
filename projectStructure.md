data-cleaning-agent/
│
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
│
├── app/
│   ├── main.py              # FastAPI / OpenEnv server
│   ├── env.py               # Core environment logic
│   ├── state.py             # Environment state
│   ├── models.py            # Pydantic models (Obs, Action, Reward)
│   │
│   ├── tasks/
│   │   ├── task_easy.py
│   │   ├── task_medium.py
│   │   ├── task_hard.py
│   │
│   ├── graders/
│   │   ├── grader_easy.py
│   │   ├── grader_medium.py
│   │   ├── grader_hard.py
│   │
│   └── utils/
│       ├── data_loader.py
│       ├── cleaning_rules.py
│
├── data/
│   ├── raw_easy.csv
│   ├── clean_easy.csv
│   ├── raw_medium.csv
│   ├── clean_medium.csv
│   ├── raw_hard.csv
│   ├── clean_hard.csv
│
└── baseline/
    ├── run_baseline.py