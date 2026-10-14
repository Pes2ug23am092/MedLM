"""FastAPI backend with JWT auth, hybrid RAG retrieval, and real-time MedGemma streaming from Colab."""
import asyncio
import json
import queue
import threading
from typing import Literal, Optional, Union
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse
from medinj import Detector
from medinj.calibration import load_threshold
from src.agents.emergency_triage import EMERGENCY_WARNING, check_emergency_triage
from src.agents.guardrails import check_hard_rules, format_guardrail_intercept, parse_embedded_options
from src.agents.intake_extraction import extract_intake_fields
from src.agents.prompting import build_prompt, format_patient_profile, sanitize_patient_output
from src.agents.verifier import verify_evidence
from src.auth.dependencies import get_current_user
from src.auth.security import create_access_token, hash_password, verify_password
from src.config import DOCS_DIR, MODEL_ID, PROMPT_STYLE, ROOT
from src.db.models import ApiUser, Conversation, ConversationMessage
from src.db.orm import ENGINE, get_db_session, init_db
from src.llm.client import health_sync, stream_generate
from src.retrieval.retriever import collection, embedder, reranker, retrieve_context


# Prompt-injection detector: block high-confidence injection attempts before retrieval.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_PATH = PROJECT_ROOT / "outputs" / "thresholds.json"
INJECTION_LOG_PATH = PROJECT_ROOT / "logs" / "injection_decisions.jsonl"

# Ensure the log directory exists in a fresh checkout.
INJECTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

INJECTION_BACKEND = "prompt_guard_2"
INJECTION_THRESHOLD = load_threshold(
    str(THRESHOLD_PATH),
    INJECTION_BACKEND,
)

injection_detector = Detector(
    backend=INJECTION_BACKEND,
    threshold=INJECTION_THRESHOLD,
    shadow_mode=False,
    log_path=str(INJECTION_LOG_PATH),
    log_text=False,
)

PROMPT_INJECTION_RESPONSE = (
    "Sorry, this prompt seems suspicious and I cannot process it. "
    "Please rephrase your question without instructions about changing the assistant's behavior."
)

app = FastAPI(title="Medical RAG & MedGemma Clinical API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8600",
        "http://127.0.0.1:8600",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup():
    init_db()


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[EmailStr] = None


class UserPublic(BaseModel):
    id: int
    username: str
    email: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PatientProfile(BaseModel):
    age: Optional[str] = None
    sex: Optional[str] = None
    duration: Optional[str] = None
    conditions: Optional[str] = None
    medications: Optional[str] = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatStreamRequest(BaseModel):
    conversation_id: Optional[Union[str, int]] = None
    message: str
    patient_profile: Optional[PatientProfile] = None
    history: list[ChatTurn] = Field(default_factory=list)
    mode: Literal["patient", "clinician"] = "patient"


class NewConversationRequest(BaseModel):
    title: Optional[str] = None
    patient_profile: Optional[PatientProfile] = None


class MedicationCheckRequest(BaseModel):
    medications: Optional[str] = None
    conditions: Optional[str] = None


def _format_patient_profile(profile: Optional[PatientProfile]) -> str:
    if not profile:
        return ""
    return format_patient_profile(
        age=profile.age,
        sex=profile.sex,
        duration=profile.duration,
        conditions=profile.conditions,
        medications=profile.medications,
    )


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------
def _save_message(db: Session, conversation_id: int, role: str, content: str, meta: Optional[dict] = None) -> None:
    msg = ConversationMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        meta=json.dumps(meta) if meta is not None else None,
    )
    db.add(msg)
    db.commit()


def _conversation_history(db: Session, conversation_id: int, last_n: int = 6) -> list:
    rows = db.exec(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.id.desc())
        .limit(last_n)
    ).all()
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def _conversation_messages_full(db: Session, conversation_id: int) -> list:
    rows = db.exec(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.id.asc())
    ).all()
    return [{"role": r.role, "content": r.content, "created_at": r.created_at.isoformat()} for r in rows]


def _conversation_payload(db: Session, conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "date": conv.created_at.isoformat(),
        "subject": conv.title or f"Consultation #{conv.id}",
        "patient_profile": json.loads(conv.patient_profile) if conv.patient_profile else None,
        "messages": _conversation_messages_full(db, conv.id),
    }


def _get_owned_conversation(db: Session, conversation_id: int, user_id: int) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


# --------------------------------------------------------------------------
# Health Endpoint
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    colab_health = health_sync()

    try:
        chromadb_count = collection.count()
        chromadb_ok = chromadb_count > 0
    except Exception:
        chromadb_count = 0
        chromadb_ok = False

    reranker_ok = reranker is not None and embedder is not None

    return {
        "colab_medgemma": colab_health.get("ok", False),
        "colab_details": colab_health,
        "chromadb": chromadb_ok,
        "chromadb_chunks": chromadb_count,
        "reranker": reranker_ok,
        "model": MODEL_ID,
        "prompt_style": PROMPT_STYLE,
        "status": "ok" if (colab_health.get("ok", False) and chromadb_ok and reranker_ok) else "degraded",
    }


# --------------------------------------------------------------------------
# Evaluation & Benchmark Endpoints
# --------------------------------------------------------------------------
_BENCHMARK_RESULTS_PATH = DOCS_DIR / "benchmark_results.json"
_BENCHMARK_REPORT_PATH = DOCS_DIR / "BENCHMARK_REPORT.md"


@app.get("/api/evaluation/summary")
def evaluation_summary():
    if not _BENCHMARK_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark results found in docs/benchmark_results.json",
        )
    try:
        payload = json.loads(_BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"benchmark_results.json is not valid JSON: {exc}")
    return {
        "generated_at": payload.get("generated_at"),
        "model": payload.get("model", MODEL_ID),
        "dataset_version": payload.get("dataset_version"),
        "num_cases": payload.get("num_cases"),
        "metrics": payload.get("metrics"),
        "cases": payload.get("cases", []),
    }


@app.get("/api/evaluation/report")
def evaluation_report():
    if not _BENCHMARK_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark report found in docs/BENCHMARK_REPORT.md",
        )
    return {"markdown": _BENCHMARK_REPORT_PATH.read_text(encoding="utf-8")}


# --------------------------------------------------------------------------
# Auth Endpoints
# --------------------------------------------------------------------------
@app.post("/api/auth/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db_session)):
    username = payload.username.strip()
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    existing = db.exec(select(ApiUser).where(ApiUser.username == username)).first()
    if existing:
        raise HTTPException(status_code=409, detail="That username is already taken.")

    user = ApiUser(username=username, email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserPublic(id=user.id, username=user.username, email=user.email)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db_session)):
    user = db.exec(select(ApiUser).where(ApiUser.username == form_data.username.strip())).first()
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=user.username)
    return TokenResponse(access_token=token)


@app.get("/api/auth/me", response_model=UserPublic)
def read_me(current_user: ApiUser = Depends(get_current_user)):
    return UserPublic(id=current_user.id, username=current_user.username, email=current_user.email)


# --------------------------------------------------------------------------
# Conversation Endpoints
# --------------------------------------------------------------------------
@app.get("/api/conversations")
def list_conversations(
    current_user: ApiUser = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    rows = db.exec(
        select(Conversation).where(Conversation.user_id == current_user.id).order_by(Conversation.id.desc())
    ).all()
    return [_conversation_payload(db, c) for c in rows]


@app.get("/api/conversations/{conversation_id}")
def get_conversation_detail(
    conversation_id: int,
    current_user: ApiUser = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    conv = _get_owned_conversation(db, conversation_id, current_user.id)
    return _conversation_payload(db, conv)


@app.post("/api/conversations/new")
def new_conversation(
    payload: NewConversationRequest = NewConversationRequest(),
    current_user: ApiUser = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    title = (payload.title or "New Consultation").strip()[:60] or "New Consultation"
    profile_json = json.dumps(payload.patient_profile.model_dump()) if payload.patient_profile else None
    conv = Conversation(user_id=current_user.id, title=title, patient_profile=profile_json)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {"conversation_id": conv.id, "title": title}


# --------------------------------------------------------------------------
# Clinical Utility Endpoints
# --------------------------------------------------------------------------
@app.post("/api/medications/check")
def check_medications(payload: MedicationCheckRequest, current_user: ApiUser = Depends(get_current_user)):
    medications_text = (payload.medications or "").strip()
    if not medications_text:
        return {"medications": [], "alert": None}

    medications = [m.strip() for m in medications_text.split(",") if m.strip()]
    combined_context = f"{medications_text}. Reported conditions: {payload.conditions or 'none reported'}."
    hit = check_hard_rules(combined_context)
    alert = hit[2] if hit else None
    return {"medications": medications, "alert": alert}


_PATIENT_FRIENDLY_FIELDS = {"clinical_guideline", "indications"}


@app.get("/api/evidence/search")
def search_evidence(q: str = "", mode: str = "clinician", current_user: ApiUser = Depends(get_current_user)):
    query = q.strip()
    if not query:
        return {"results": []}
    results = retrieve_context(query, top_k=8)
    payload = [
        {
            "chunk_id": str(chunk_id),
            "title": meta.get("title", "Guideline"),
            "source": meta.get("source", ""),
            "field": meta.get("field", ""),
            "score": round(float(meta.get("cross_score", 0.0)), 4),
            "excerpt": text[:400],
        }
        for chunk_id, text, meta in results
    ]
    if mode == "patient":
        payload.sort(key=lambda r: 0 if r["field"] in _PATIENT_FRIENDLY_FIELDS else 1)
    return {"results": payload}


# --------------------------------------------------------------------------
# Real-Time Chat Stream (SSE)
# --------------------------------------------------------------------------
def _run_pipeline(req: ChatStreamRequest, conversation_id: int, sink: "queue.Queue"):
    with Session(ENGINE) as db:
        try:
            raw_input = req.message.strip()

            # Layer 0: Emergency Red-Flag Triage
            triage_category = check_emergency_triage(raw_input)
            if triage_category:
                _save_message(db, conversation_id, "user", raw_input)
                _save_message(
                    db,
                    conversation_id,
                    "assistant",
                    EMERGENCY_WARNING,
                    meta={"emergency_triage": triage_category},
                )
                sink.put({"event": "sources", "data": []})
                sink.put({"event": "triage_status", "data": {"level": "EMERGENCY", "category": triage_category}})
                sink.put({"event": "token", "data": EMERGENCY_WARNING})
                sink.put({"event": "done", "data": {"status": "EMERGENCY_ESCALATION", "score": 1.0}})
                return

            # Prompt-injection detection (shadow mode: observe, do not block).
            injection_result = injection_detector.detect(raw_input)

            print(
                "[prompt-injection] "
                f"label={injection_result.label} "
                f"score={injection_result.score:.6f} "
                f"threshold={injection_result.threshold:.6f} "
                f"latency_ms={injection_result.latency_ms:.2f}"
            )

            if injection_detector.should_block(injection_result):
                _save_message(db, conversation_id, "user", raw_input)
                _save_message(
                    db,
                    conversation_id,
                    "assistant",
                    PROMPT_INJECTION_RESPONSE,
                    meta={
                        "prompt_injection": True,
                        "detector_score": injection_result.score,
                    },
                )
                sink.put({"event": "sources", "data": []})
                sink.put({"event": "token", "data": PROMPT_INJECTION_RESPONSE})
                sink.put(
                    {
                        "event": "done",
                        "data": {
                            "status": "PROMPT_INJECTION_BLOCKED",
                            "score": injection_result.score,
                        },
                    }
                )
                return

            # Layer 1: Adversarial Guardrail
            hit = check_hard_rules(raw_input)
            if hit:
                entity, _, warning = hit
                response_text = format_guardrail_intercept(entity, warning, mode=req.mode)
                _save_message(db, conversation_id, "user", raw_input)
                _save_message(db, conversation_id, "assistant", response_text, meta={"guardrail_intercept": True})
                sink.put({"event": "sources", "data": []})
                sink.put({"event": "token", "data": response_text})
                sink.put({"event": "done", "data": {"status": "INTERCEPTED", "score": 1.0}})
                return

            # Layer 2: Extract real-time profile update
            extracted_profile = extract_intake_fields(raw_input)
            if extracted_profile:
                sink.put({"event": "profile_update", "data": extracted_profile})

            # Layer 3: Hybrid RAG Retrieval (BM25 + ChromaDB + RRF + MedCPT Rerank)
            vignette_text, detected_options = parse_embedded_options(raw_input)
            evidence = retrieve_context(vignette_text, top_k=3)

            sources = [
                {
                    "chunk_id": str(chunk_id),
                    "title": meta.get("title", "Guideline"),
                    "source": meta.get("source", ""),
                    "score": round(float(meta.get("cross_score", 0.0)), 4),
                    "excerpt": text[:280],
                }
                for chunk_id, text, meta in evidence
            ]
            sink.put({"event": "sources", "data": sources})

            evidence_md = ""
            for idx, (_, text, meta) in enumerate(evidence, 1):
                evidence_md += f"### [Evidence {idx}] {meta.get('title', 'Guideline')} (`{meta.get('source', '')}`)\n"
                evidence_md += f"*MedCPT Rerank Score:* `{meta.get('cross_score', 0):.4f}`\n\n> {text}\n\n---\n"

            # Layer 4: Tone-aware MedGemma Prompt Construction
            profile_note = _format_patient_profile(req.patient_profile)
            vignette_with_profile = f"{profile_note}\n{vignette_text}" if profile_note else vignette_text
            conversation_history = [turn.model_dump() for turn in req.history] or _conversation_history(
                db, conversation_id, last_n=6
            )
            forced_tone = "patient" if req.mode == "patient" else "clinical_qa"
            prompt = build_prompt(
                vignette_with_profile,
                detected_options,
                evidence_md,
                conversation_history,
                forced_tone=forced_tone,
                prompt_style=PROMPT_STYLE,
            )

            # Layer 5: Real-time token streaming from Colab MedGemma
            accumulated_response = ""
            for delta in stream_generate(prompt):
                accumulated_response += delta
                sink.put({"event": "token", "data": delta})

            # Sanitize output
            accumulated_response = sanitize_patient_output(accumulated_response)

            # Layer 6: Post-generation evidence audit
            audit = verify_evidence(accumulated_response, evidence, vignette_text)

            _save_message(db, conversation_id, "user", raw_input)
            _save_message(
                db,
                conversation_id,
                "assistant",
                accumulated_response,
                meta={"faithfulness_score": audit["score"], "status": audit["status"]},
            )

            sink.put(
                {
                    "event": "done",
                    "data": {
                        "status": audit["status"],
                        "score": audit["score"],
                        "verified_claims": audit["verified_claims"],
                        "flagged_claims": audit["flagged_claims"],
                    },
                }
            )
        except Exception as exc:
            sink.put({"event": "error", "data": str(exc)})
        finally:
            sink.put(None)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamRequest, current_user: ApiUser = Depends(get_current_user)):
    with Session(ENGINE) as db:
        if req.conversation_id:
            try:
                conversation_id = int(req.conversation_id)
            except ValueError:
                conversation_id = None

            if conversation_id:
                _get_owned_conversation(db, conversation_id, current_user.id)
            else:
                title = req.message.strip().splitlines()[0][:60] if req.message.strip() else "New Consultation"
                profile_json = json.dumps(req.patient_profile.model_dump()) if req.patient_profile else None
                conv = Conversation(user_id=current_user.id, title=title, patient_profile=profile_json)
                db.add(conv)
                db.commit()
                db.refresh(conv)
                conversation_id = conv.id
        else:
            title = req.message.strip().splitlines()[0][:60] if req.message.strip() else "New Consultation"
            profile_json = json.dumps(req.patient_profile.model_dump()) if req.patient_profile else None
            conv = Conversation(user_id=current_user.id, title=title, patient_profile=profile_json)
            db.add(conv)
            db.commit()
            db.refresh(conv)
            conversation_id = conv.id

    sink: "queue.Queue" = queue.Queue()
    threading.Thread(target=_run_pipeline, args=(req, conversation_id, sink), daemon=True).start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        yield {"event": "conversation", "data": str(conversation_id)}
        while True:
            item = await loop.run_in_executor(None, sink.get)
            if item is None:
                break
            data = item["data"]
            if not isinstance(data, str):
                data = json.dumps(data)
            yield {"event": item["event"], "data": data}

    return EventSourceResponse(event_generator())


from fastapi.responses import FileResponse

# Mount static directory
static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

dist_dir = ROOT / "frontend" / "dist"
if dist_dir.exists():
    app.mount("/app", StaticFiles(directory=dist_dir, html=True), name="frontend_app")


@app.get("/")
def index():
    if (ROOT / "static" / "index.html").exists():
        return FileResponse(ROOT / "static" / "index.html")
    if dist_dir.exists():
        return FileResponse(dist_dir / "index.html")
    return {"message": "Medical RAG API is running."}
