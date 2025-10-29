# -*- coding: utf-8 -*-
"""
『アプリ減災教室』バックエンド（改）
- 設定の一元化（環境変数）
- 例外ハンドラ/ログ強化
- ファイル監視でのホットリロード
- YOLO遅延ロードの排他制御
- 入力バリデーション強化
- 静的配信の柔軟化
"""

from __future__ import annotations
import os
import re
import csv
import json
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Iterable

from fastapi import (
    FastAPI, UploadFile, File, Query, HTTPException, Form, Body, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, conint, confloat

# YOLO は必要時のみロード
from ultralytics import YOLO

from inventory_vision import analyze_photo_file  # YOLO ラッパ

# --------------------------------------------------------------------
# 設定（環境変数）
# --------------------------------------------------------------------
APP_NAME = os.environ.get("APP_NAME", "Disaster Demo API (Water-only)")
APP_VERSION = os.environ.get("APP_VERSION", "2025.10.29")

# ディレクトリ
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data")).resolve()
STATIC_DIR = Path(os.environ.get("STATIC_DIR", BASE_DIR / "static")).resolve()
UNITY_DIR = Path(os.environ.get("UNITY_DIR", STATIC_DIR / "unity")).resolve()     # 任意
FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR", BASE_DIR / "frontend_dist")).resolve()  # 任意

# CORS
DEFAULT_CORS = ["http://localhost:5173", "http://127.0.0.1:5173"]
CORS_ALLOW_ORIGINS = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", ",".join(DEFAULT_CORS)).split(",") if o.strip()]
CORS_ALLOW_CREDENTIALS = os.environ.get("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"

# モデル
MODEL_PATH = os.environ.get(
    "WATER_BOTTLE_MODEL",
    "D:/画像認識/2025_0805/runs/train/water_bottle/weights/best.pt",
)

# アップロード制限
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "12"))  # 12MB
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# スコア設定
W_ANSWERS = float(os.environ.get("W_ANSWERS", "0.8"))
W_INVENTORY = float(os.environ.get("W_INVENTORY", "0.2"))
SCORE_MAX = int(os.environ.get("SCORE_MAX", "100"))
TARGET_DAYS = float(os.environ.get("TARGET_DAYS", "3.0"))
MAX_ADVICE = int(os.environ.get("MAX_ADVICE", "5"))  # /advice?top_k= で上書き可（1~20）

# --------------------------------------------------------------------
# FastAPI 本体
# --------------------------------------------------------------------
app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)

# 静的配信（存在するものだけマウント）
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")
if UNITY_DIR.exists():
    app.mount("/unity", StaticFiles(directory=str(UNITY_DIR), html=False), name="unity")
if FRONTEND_DIR.exists():
    # Vite/React のビルド成果物（dist）を置けば /frontend で見られる
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# ログ + 例外ハンドラ
# --------------------------------------------------------------------
@app.exception_handler(Exception)
async def _unhandled_exc(_: Request, exc: Exception):
    # ここでログだけ出して、ユーザーには汎用メッセージ
    try:
        print(f"[ERROR] {exc!r}")
    except Exception:
        pass
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# --------------------------------------------------------------------
# データ読み込み（ホットリロード対応）
# --------------------------------------------------------------------
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _read_quiz_csv(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["weight"] = int(r["weight"])
                r["no"] = int(r["no"])
                out.append(r)
            except Exception:
                continue
    return out

class HotFile:
    """更新時刻を持ってキャッシュ&ホットリロードする簡易ヘルパ"""
    def __init__(self, path: Path, loader):
        self.path = path
        self.loader = loader
        self._mtime = None
        self._cache = None

    def get(self):
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if mtime != self._mtime:
            self._cache = self.loader(self.path)
            self._mtime = mtime
        return self._cache

# 構成ファイル
SCENARIOS = HotFile(DATA_DIR / "scenarios_gifu_gotanda.json", lambda p: _read_json(p, {"items": []}))
STOCK = HotFile(DATA_DIR / "stock_baseline_v1.json", lambda p: _read_json(p, {}))
QUIZ = HotFile(DATA_DIR / "quiz_v1.csv", _read_quiz_csv)
RULES = HotFile(DATA_DIR / "advice_rules_v1.json", lambda p: _read_json(p, []))

def _get_stock() -> dict:
    d = dict(STOCK.get() or {})
    ppd = d.setdefault("per_person_daily", {})
    ppd["water_l"] = float(ppd.get("water_l", 3))
    return d

# --------------------------------------------------------------------
# Pydantic モデル
# --------------------------------------------------------------------
class Answer(BaseModel):
    id: str
    value: Literal["yes", "some", "no"]

class ScoreRequest(BaseModel):
    answers: List[Answer] = Field(default_factory=list)
    inventory_days: confloat(ge=0) = 0
    flood_depth_m: confloat(ge=0) = 0
    scenario_path: List[dict] = Field(default_factory=list)  # 将来拡張に備えて受ける

class InventoryAnalyzeBody(BaseModel):
    water_l: confloat(ge=0) = 0
    persons: conint(ge=1) = 1

class AdviceItem(BaseModel):
    msg: str
    done: bool = False

class Progress(BaseModel):
    user_id: str
    user_name: Optional[str] = None
    group_id: str
    score: Optional[int] = None
    rank: Optional[str] = None
    advice: List[AdviceItem] = Field(default_factory=list)
    last_updated: Optional[str] = None

class SavedResponse(BaseModel):
    id: str                 # uuid
    user_id: str
    user_name: Optional[str] = None
    answers: List[dict]     # [{id, value}]
    scenario_path: List[dict] = Field(default_factory=list)
    inventory_days: float = 0.0
    score: Optional[dict] = None
    advice: List[str] = Field(default_factory=list)
    created_at: str         # isoformat

class SaveRequest(BaseModel):
    user_id: str
    user_name: Optional[str] = None
    answers: List[Answer] = Field(default_factory=list)
    scenario_path: List[dict] = Field(default_factory=list)
    inventory_days: confloat(ge=0) = 0
    score: Optional[dict] = None
    advice: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------
# グループ/進捗（メモリ保持）
# --------------------------------------------------------------------
GROUPS: Dict[str, Dict] = {}
USERS: Dict[str, Dict] = {}

# --------------------------------------------------------------------
# 補助（JSONL 永続化）
# --------------------------------------------------------------------
RESPONSES_PATH = DATA_DIR / "responses.jsonl"
_RESP_LOCK = Lock()

def _append_jsonl(obj: dict, path: Path = RESPONSES_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _RESP_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _iter_jsonl(path: Path = RESPONSES_PATH) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _read_last_for_user(user_id: str) -> Optional[dict]:
    """末尾から逆走査（大量データでも高速）"""
    if not RESPONSES_PATH.exists():
        return None
    try:
        with RESPONSES_PATH.open("rb") as f:
            f.seek(0, 2)
            pos = f.tell()
            buf = b""
            while pos > 0:
                step = min(4096, pos)
                pos -= step
                f.seek(pos)
                chunk = f.read(step)
                buf = chunk + buf
                while b"\n" in buf:
                    line, sep, buf = buf.rpartition(b"\n")
                    rec = line.strip()
                    if not rec:
                        continue
                    try:
                        obj = json.loads(rec.decode("utf-8"))
                        if obj.get("user_id") == user_id:
                            return obj
                    except Exception:
                        continue
    except Exception:
        pass
    return None

# --------------------------------------------------------------------
# ルール評価
# --------------------------------------------------------------------
def answers_to_map(answers: List[Answer]) -> Dict[str, str]:
    return {a.id.strip(): a.value.strip() for a in answers}

_num_cond_single_re = re.compile(r'(depth|inventory_days)\s*(<=|>=|<|>|==)\s*([0-9]*\.?[0-9]+)$', re.I)

def _eval_one_numeric(expr: str, ctx: dict) -> bool:
    m = _num_cond_single_re.fullmatch(expr.strip())
    if not m:
        return False
    key, op, sval = m.group(1).lower(), m.group(2), m.group(3)
    lhs = ctx["depth"] if key == "depth" else ctx["inventory_days"]
    rhs = float(sval)
    if   op == "<":  return lhs < rhs
    elif op == ">":  return lhs > rhs
    elif op == "<=": return lhs <= rhs
    elif op == ">=": return lhs >= rhs
    elif op == "==": return lhs == rhs
    return False

def evaluate_numeric(cond: str, ctx: dict) -> bool:
    if not cond:
        return False
    # "A and B" / "A && B" をすべて満たす場合に True
    parts = re.split(r'\s+(?:and|&&)\s+', cond.strip(), flags=re.I)
    return parts and all(_eval_one_numeric(p, ctx) for p in parts)

def evaluate_question(cond: str, amap: Dict[str, str]) -> bool:
    m = re.fullmatch(r'(q[0-9]+)\s*=\s*(yes|some|no)', (cond or "").strip(), flags=re.I)
    if not m:
        return False
    qid, v = m.group(1), m.group(2).lower()
    return amap.get(qid) == v

def evaluate_beginner_flag(cond: str, answers: List[Answer]) -> bool:
    if (cond or "").strip().lower() != "beginner":
        return False
    yes_count = sum(1 for a in answers if a.value == "yes")
    return (len(answers) == 0) or (yes_count <= 2)

def pick_advices_by_rules(req: ScoreRequest, rules: List[dict], max_items: int = MAX_ADVICE) -> List[str]:
    amap = answers_to_map(req.answers)
    ctx = {"depth": float(req.flood_depth_m or 0), "inventory_days": float(req.inventory_days or 0)}
    matched: List[dict] = []
    for r in rules:
        cond = (r.get("cond") or "").strip()
        ok = (
            evaluate_beginner_flag(cond, req.answers) or
            evaluate_numeric(cond, ctx) or
            evaluate_question(cond, amap)
        )
        if ok:
            matched.append(r)
    matched.sort(key=lambda x: int(x.get("priority", 0)), reverse=True)
    out, seen = [], set()
    for r in matched:
        adv = r.get("advice")
        if adv and adv not in seen:
            out.append(adv); seen.add(adv)
        if len(out) >= max_items:
            break
    return out

# --------------------------------------------------------------------
# ヘルス・メタ情報
# --------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Backend OK"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/version")
def version():
    return {"name": APP_NAME, "version": APP_VERSION}

@app.get("/config")
def config():
    return {
        "cors": CORS_ALLOW_ORIGINS,
        "static": str(STATIC_DIR),
        "unity_mounted": UNITY_DIR.exists(),
        "frontend_mounted": FRONTEND_DIR.exists(),
        "max_upload_mb": MAX_UPLOAD_MB,
        "model_path": MODEL_PATH,
    }

# --------------------------------------------------------------------
# コンテンツAPI
# --------------------------------------------------------------------
@app.get("/scenarios")
def scenarios(place: str = Query("gifu_gotanda")):
    # placeの切替に応じてファイル名を変えたい場合は HotFile を動的に作ってもOK
    return SCENARIOS.get()

@app.get("/quiz")
def quiz():
    return {"items": QUIZ.get()}

# --------------------------------------------------------------------
# 備蓄診断（数値入力）
# --------------------------------------------------------------------
@app.post("/inventory/analyze")
def inventory_analyze(
    body: Optional[InventoryAnalyzeBody] = Body(None),
    water_l: Optional[float] = Query(None, description="合計水量[L]"),
    persons: Optional[int]  = Query(None, description="人数"),
):
    b = body or InventoryAnalyzeBody()
    water = float(water_l) if water_l is not None else float(b.water_l)
    ppl   = int(persons) if persons is not None else int(b.persons)
    if ppl < 1:
        ppl = 1

    stock = _get_stock()
    daily_need_per_person = float(stock["per_person_daily"].get("water_l", 3))
    need_water_per_day = daily_need_per_person * ppl

    days_water = (water / need_water_per_day) if need_water_per_day > 0 else 0.0
    inventory_days = round(days_water, 1)

    return {
        "persons": ppl,
        "inputs": {"water_l": water},
        "per_person_daily": {"water_l": daily_need_per_person},
        "estimated_days": inventory_days,
        "bottlenecks": {"water_days": round(days_water, 1)}
    }

# --------------------------------------------------------------------
# スコア
# --------------------------------------------------------------------
@app.post("/levels/score")
def levels_score(req: ScoreRequest):
    meta = {row["id"]: row for row in QUIZ.get() or []}
    answers_total = 0
    answers_max = 0
    detail: List[dict] = []

    for a in req.answers:
        if a.id in meta:
            w = int(meta[a.id]["weight"])
            raw = 10 if a.value == "yes" else 6 if a.value == "some" else 0
            answers_total += raw * w
            answers_max += 10 * w
            detail.append({
                "id": a.id,
                "level": meta[a.id]["level"],
                "weight": w,
                "no": meta[a.id]["no"],
                "question": meta[a.id]["question"],
                "raw": raw,
                "score": raw * w
            })

    if answers_max <= 0:
        answers_max = 1
    answers_rate = answers_total / answers_max

    inv_days = float(req.inventory_days or 0.0)
    target_days = TARGET_DAYS if TARGET_DAYS > 0 else 3.0
    inv_rate = max(0.0, min(inv_days / target_days, 1.0))

    final_rate = (W_ANSWERS * answers_rate) + (W_INVENTORY * inv_rate)
    final_total = int(round(final_rate * SCORE_MAX))
    final_max = SCORE_MAX

    if final_rate < 0.45:
        rank = "Beginner"
    elif final_rate < 0.75:
        rank = "Intermediate"
    else:
        rank = "Advanced"

    priorities = []
    for a in req.answers:
        if a.id in meta and meta[a.id]["weight"] >= 2 and a.value == "no":
            priorities.append({
                "id": a.id,
                "level": meta[a.id]["level"],
                "next": f"対策：『{meta[a.id]['question']}』を満たす準備"
            })
            if len(priorities) >= 5:
                break

    return {
        "score_total": final_total,
        "score_max": final_max,
        "score_rate": round(final_rate, 3),
        "rank": rank,
        "breakdown": {
            "answers_total": answers_total,
            "answers_max": answers_max,
            "answers_rate": round(answers_rate, 3),
            "inventory_days": inv_days,
            "target_days": target_days,
            "inventory_rate": round(inv_rate, 3),
            "weights": {"answers": W_ANSWERS, "inventory": W_INVENTORY}
        },
        "answered": len([a for a in req.answers if a.id in meta]),
        "detail": detail,
        "priorities": priorities
    }

# --------------------------------------------------------------------
# アドバイス
# --------------------------------------------------------------------
@app.post("/advice")
def advice(req: ScoreRequest, top_k: int = Query(MAX_ADVICE, ge=1, le=20)):
    actions = pick_advices_by_rules(req, RULES.get() or [], max_items=top_k)
    if not actions:
        try:
            print("[ADVICE][no-match]", {
                "ctx": {"depth": float(req.flood_depth_m or 0), "inventory_days": float(req.inventory_days or 0)},
                "answers": answers_to_map(req.answers),
                "rules_count": len(RULES.get() or [])
            })
        except Exception:
            pass
    return {"actions": actions}

# --------------------------------------------------------------------
# グループ機能
# --------------------------------------------------------------------
@app.post("/groups/create")
def create_group(name: str = Form(...)):
    group_id = str(uuid.uuid4())[:8]
    GROUPS[group_id] = {"name": name, "members": []}
    return {"group_id": group_id, "invite_code": group_id}

@app.post("/groups/join")
def join_group(
    user_id: str = Form(...),
    group_id: str = Form(...),
    user_name: str = Form("", description="任意・空なら未設定"),
):
    if group_id not in GROUPS:
        return {"error": "Group not found"}
    if user_id not in USERS:
        USERS[user_id] = {"name": (user_name or None), "progress": None}
    else:
        if user_name:
            USERS[user_id]["name"] = user_name
    if user_id not in GROUPS[group_id]["members"]:
        GROUPS[group_id]["members"].append(user_id)
    return {"message": "Joined group", "group_id": group_id}

@app.post("/progress/update")
def update_progress(p: Progress):
    data = p.dict()
    if not data.get("last_updated"):
        data["last_updated"] = datetime.utcnow().isoformat()
    prev = USERS.get(p.user_id, {"name": None, "progress": None})
    if data.get("user_name"):
        prev["name"] = data["user_name"]
    prev["progress"] = data
    USERS[p.user_id] = prev
    return {"message": "Progress updated"}

@app.get("/groups/{group_id}/progress")
def group_progress(group_id: str):
    if group_id not in GROUPS:
        return {"error": "Group not found"}
    members = GROUPS[group_id]["members"]
    data = []
    for uid in members:
        u = USERS.get(uid, {})
        prog = u.get("progress")
        if prog:
            data.append(prog)
        else:
            data.append({
                "user_id": uid,
                "user_name": u.get("name"),
                "score": None,
                "rank": None,
                "advice": [],
                "last_updated": None,
            })
    return {"group_id": group_id, "members": data}

# --------------------------------------------------------------------
# 回答・履歴 永続化
# --------------------------------------------------------------------
@app.post("/responses/save")
def responses_save(req: SaveRequest):
    rec = SavedResponse(
        id=uuid.uuid4().hex,
        user_id=req.user_id,
        user_name=req.user_name,
        answers=[a.dict() for a in req.answers],
        scenario_path=req.scenario_path or [],
        inventory_days=float(req.inventory_days or 0),
        score=req.score,
        advice=req.advice or [],
        created_at=datetime.utcnow().isoformat()
    )
    _append_jsonl(rec.dict())
    return {"ok": True, "id": rec.id}

@app.get("/responses/last")
def responses_last(user_id: str):
    last = _read_last_for_user(user_id) or {}
    return last

@app.get("/responses/history")
def responses_history(user_id: str, limit: int = 50):
    out = [obj for obj in _iter_jsonl() if obj.get("user_id") == user_id]
    out.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return out[:max(1, min(limit, 200))]

# --------------------------------------------------------------------
# 画像アップロード → YOLO で備蓄診断（水）
# --------------------------------------------------------------------
_app_model_lock = Lock()
app.state.yolo_model = None  # 遅延ロード

def _get_yolo_model() -> YOLO:
    if app.state.yolo_model is None:
        with _app_model_lock:
            if app.state.yolo_model is None:
                try:
                    app.state.yolo_model = YOLO(MODEL_PATH)
                    print(f"[YOLO] Loaded model: {MODEL_PATH}")
                except Exception as e:
                    print(f"[YOLO] Failed to load model: {e}")
                    raise HTTPException(status_code=500, detail=f"YOLO model load failed")
    return app.state.yolo_model

def _validate_upload(image: UploadFile):
    # サイズ（Content-Length を信用できないケースがあるため read 後にチェック）
    ext = (Path(image.filename).suffix or "").lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    return ext

@app.post("/inventory/photo")
async def inventory_photo(
    image: UploadFile = File(...),
    persons: int = Form(1),
    conf_thresh: float = Form(0.5)
):
    if persons < 1:
        persons = 1

    ext = _validate_upload(image)

    try:
        # 受信 → サイズ制限チェック
        content = await image.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_MB} MB)")

        # 一時保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        model = _get_yolo_model()
        out_name = f"{uuid.uuid4().hex}.jpg"
        out_path = str(RESULTS_DIR / out_name)

        res = analyze_photo_file(
            model,
            tmp_path,
            persons=persons,
            conf_thresh=float(conf_thresh),
            out_path=out_path
        )

        image_url = f"/static/results/{Path(res['visual_path']).name}"
        return {
            "persons": res["persons"],
            "counts": res["counts"],
            "count_500ml": res["counts"]["water_500ml"],
            "count_2l":   res["counts"]["water_2l"],
            "total_l": res["total_l"],
            "estimated_days": res["estimated_days"],
            "image_url": image_url,
            "visual_path": image_url
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except Exception:
            pass

# 互換API（未使用）
@app.post("/inventory/upload")
async def inventory_upload(file: UploadFile = File(...)):
    return {"filename": file.filename}
