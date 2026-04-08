from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from typing import Optional
from pydantic import BaseModel

from app.env import env
from app.models import Action, StepResponse, ResetResponse

app = FastAPI(
    title="Data Cleaning Agent",
    description="An OpenEnv-compatible RL environment for training agents to clean messy tabular data.",
    version="1.0.0",
)

class ResetRequest(BaseModel):
    task_id: str = "easy"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    session_id: str
    action: Action

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok", "service": "data-cleaning-agent"}

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    try:
        session_id, observation = env.reset(
            task_id=request.task_id,
            session_id=request.session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ResetResponse(
        observation=observation,
        message=f"Episode started. task='{request.task_id}', session='{session_id}'",
    )

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    try:
        response = env.step(session_id=request.session_id, action=request.action)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return response

@app.get("/state")
def state(session_id: str = Query(..., description="Session ID returned by /reset")):
    try:
        return env.state(session_id=session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))