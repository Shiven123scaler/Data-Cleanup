from __future__ import annotations
import uuid
import pandas as pd
from typing import Optional

from app.models import Action, Observation, RewardInfo, SchemaInfo, StepResponse

TASK_CONFIG = {
    "easy":   {"max_steps": 5,  "raw_data_path": "data/raw_easy.csv",   "clean_data_path": "data/clean_easy.csv"},
    "medium": {"max_steps": 15, "raw_data_path": "data/raw_medium.csv", "clean_data_path": "data/clean_medium.csv"},
    "hard":   {"max_steps": 30, "raw_data_path": "data/raw_hard.csv",   "clean_data_path": "data/clean_hard.csv"},
}

class EpisodeState:
    def __init__(self, task_id: str, df_raw: pd.DataFrame, df_clean: pd.DataFrame):
        self.task_id = task_id
        self.df = df_raw.copy()
        self.df_clean = df_clean.copy()
        self.step_count = 0
        self.max_steps = TASK_CONFIG[task_id]["max_steps"]
        self.prev_score = 0.0
        self.done = False

    def is_done(self) -> bool:
        return self.done or self.step_count >= self.max_steps
    

class DataCleaningEnv:
    def __init__(self):
        self._sessions: dict[str, EpisodeState] = {}
    
    def reset(self, task_id: str = "easy", session_id: Optional[str] = None):
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_CONFIG)}")

        df_raw, df_clean = self._load_data(task_id)

        sid = session_id or str(uuid.uuid4())
        self._sessions[sid] = EpisodeState(task_id, df_raw, df_clean)

        obs = self._build_observation(self._sessions[sid])
        return sid, obs
    
    def step(self, session_id: str, action: Action) -> StepResponse:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found. Call /reset first.")

        ep = self._sessions[session_id]

        if ep.is_done():
            obs = self._build_observation(ep, is_terminal=True)
            return StepResponse(
                observation=obs,
                reward_info=RewardInfo(reward=0.0, score=ep.prev_score, done=True)
            )

        penalty, info = self._apply_action(ep, action)

        ep.step_count += 1

        new_score = self._grade(ep)

        score_delta = new_score - ep.prev_score
        reward = score_delta + penalty
        ep.prev_score = new_score

        done = ep.is_done() or new_score >= 1.0
        ep.done = done

        obs = self._build_observation(ep, is_terminal=done)
        reward_info = RewardInfo(
            reward=round(reward, 4),
            score=round(new_score, 4),
            done=done,
            penalty=round(penalty, 4),
            info=info,
        )

        return StepResponse(observation=obs, reward_info=reward_info)
    
    
    def state(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found.")

        ep = self._sessions[session_id]
        return {
            "session_id": session_id,
            "task_id": ep.task_id,
            "step_count": ep.step_count,
            "max_steps": ep.max_steps,
            "done": ep.done,
            "prev_score": ep.prev_score,
            "df_shape": ep.df.shape,
            "df_head": ep.df.head(5).to_dict(orient="records"),
        }
    
    def _load_data(self, task_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = TASK_CONFIG[task_id]
        try:
            df_raw = pd.read_csv(cfg["raw_data_path"], encoding="utf-8-sig")
        except UnicodeDecodeError:
            df_raw = pd.read_csv(cfg["raw_data_path"], encoding="latin-1")

        try:
            df_clean = pd.read_csv(cfg["clean_data_path"], encoding="utf-8-sig")
        except UnicodeDecodeError:
            df_clean = pd.read_csv(cfg["clean_data_path"], encoding="latin-1")

        return df_raw, df_clean
    
    def _build_observation(self, ep: EpisodeState, is_terminal: bool = False) -> Observation:
        df = ep.df

        return Observation(
            task_id=ep.task_id,
            step=ep.step_count,
            max_steps=ep.max_steps,
            df_preview=df.head(5).to_dict(orient="records"),
            schema_info=SchemaInfo(
                columns=list(df.columns),
                dtypes={col: str(df[col].dtype) for col in df.columns},
                shape=df.shape,
            ),
            null_count=df.isnull().sum().to_dict(),
            duplicate_count=int(df.duplicated().sum()),
            schema_errors=self._detect_schema_errors(df),
            is_terminal=is_terminal,
        )
    
    def _detect_schema_errors(self, df: pd.DataFrame) -> list[str]:
        errors = []
        seen = set()
        for col in df.columns:
            if col in seen:
                errors.append(f"Duplicate column name: '{col}'")
            seen.add(col)
        if df.empty:
            errors.append("DataFrame is empty")
        return errors
    
    def _grade(self, ep: EpisodeState) -> float:
        df = ep.df
        gt = ep.df_clean

        if df.empty:
            return 0.0

        null_score = 1.0 - (df.isnull().sum().sum() / max(df.size, 1))
        dup_score = 1.0 - (df.duplicated().sum() / max(len(df), 1))
        shape_ratio = min(len(df), len(gt)) / max(len(df), len(gt)) if len(gt) > 0 else 0.0

        score = 0.4 * null_score + 0.3 * dup_score + 0.3 * shape_ratio
        return round(min(max(score, 0.0), 1.0), 4)
    
    def _normalize_date_column(self, df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, bool]:
        import dateutil.parser
        def _parse(val):
            if pd.isnull(val):
                return val
            try:
                ts = float(val)
                return pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass
            try:
                return dateutil.parser.parse(str(val)).strftime("%Y-%m-%d")
            except Exception:
                return None
        original = df[col].copy()
        df = df.copy()
        df[col] = df[col].apply(_parse)
        success = df[col].notna().mean() >= 0.5
        if not success:
            df[col] = original
        return df, success

    def _drop_invalid_emails(self, df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, int]:
        import re
        EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
        def is_valid(val):
            if pd.isnull(val):
                return False
            return bool(EMAIL_RE.match(str(val).strip()))
        mask = df[col].apply(is_valid)
        dropped = int((~mask).sum())
        result = df[mask].reset_index(drop=True)
        assert isinstance(result, pd.DataFrame)
        return result, dropped
    
    def _apply_action(self, ep: EpisodeState, action: Action) -> tuple[float, dict]:
        op = action.operation
        col = action.column
        params = action.params
        df = ep.df
        penalty = 0.0
        info: dict = {"operation": op, "column": col}

        if op == "no_op":
            info["reason"] = "no_op: nothing changed"

        elif op == "drop_nulls":
            before = len(df)
            if col:
                if col not in df.columns:
                    penalty = -0.02
                    info["reason"] = f"Column '{col}' not found"
                else:
                    ep.df = df.dropna(subset=[col]).reset_index(drop=True)
            else:
                ep.df = df.dropna().reset_index(drop=True)

            after = len(ep.df)
            info["rows_dropped"] = before - after

            if after < 0.1 * before:
                ep.df = df
                penalty = -0.5
                info["reason"] = "Blocked: would drop >90% of data"

        elif op == "drop_duplicates":
            before = len(df)
            ep.df = df.drop_duplicates().reset_index(drop=True)
            info["rows_dropped"] = before - len(ep.df)

        elif op == "normalize_dates":
            if col is None or col not in df.columns:
                penalty = -0.02
                info["reason"] = f"Column '{col}' not found"
            else:
                ep.df, success = self._normalize_date_column(ep.df, col)
                if not success:
                    penalty = -0.05
                    info["reason"] = f"Could not parse dates in '{col}'"

        elif op == "rename_column":
            new_name = params.get("new_name")
            if col not in df.columns or not new_name:
                penalty = -0.02
                info["reason"] = f"Bad rename: col='{col}', new_name='{new_name}'"
            else:
                ep.df = df.rename(columns={col: new_name})
                info["renamed_to"] = new_name

        elif op == "validate_emails":
            if col is None or col not in df.columns:
                penalty = -0.02
                info["reason"] = f"Column '{col}' not found"
            else:
                ep.df, dropped = self._drop_invalid_emails(ep.df, col)
                info["invalid_rows_dropped"] = dropped

        elif op == "lowercase_column":
            if col is None or col not in df.columns:
                penalty = -0.02
                info["reason"] = f"Column '{col}' not found"
            else:
                ep.df[col] = df[col].str.lower()

        else:
            penalty = -0.02
            info["reason"] = f"Unknown operation: '{op}'"

        return penalty, info


env = DataCleaningEnv()