import os
import io
import re
import json
import time
import uuid
import shutil
import random
import zipfile
import secrets
import threading
import base64
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Deque, Tuple
from collections import defaultdict, deque
from urllib.parse import urlparse
from pathlib import Path
from itsdangerous import URLSafeSerializer, BadSignature

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import google.generativeai as genai
from elevenlabs import ElevenLabs

# ===== DEFINISI BASE_DIR HARUS DI PALING ATAS =====
BASE_DIR = Path(__file__).resolve().parent.parent  # Naik 1 level dari api/ ke root

# ===== KONSTANTA =====
SOUND_TEXT_FILE_NAME = "sound.txt"

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "RWiGLY9uXI70QL540WNd")
ELEVEN_MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_multilingual_v2")

SOUND_TXT_URL = os.getenv("SOUND_TXT_URL", "").strip()

LICENSE_REQUIRED = os.getenv("LICENSE_REQUIRED", "1") == "1"
OWNER_LICENSE = os.getenv("OWNER_LICENSE", "OWNER-CHANGE-ME")

DATA_DIR = os.getenv("DATA_DIR", "/tmp/data")
WORKSPACES_DIR = os.path.join(DATA_DIR, "workspaces")
WORKSPACE_TTL_HOURS = int(os.getenv("WORKSPACE_TTL_HOURS", "24"))
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "6"))

ADOBE_BASE = "https://cclight-transient-user.adobe.io"
ADOBE_UPLOAD_URL = f"{ADOBE_BASE}/assets/upload?storage=CSS"
ADOBE_DOWNLOAD_URLS_API = f"{ADOBE_BASE}/assets/download_urls?storage=CSS"
ADOBE_SUBMIT_URL = f"{ADOBE_BASE}/chx/submit"

CLOSE_HEADERS = {"Connection": "close", "Cache-Control": "no-cache", "Pragma": "no-cache"}
TARGET_SUBMIT_PATH = "/chx/submit"
DROP_HEADER_KEYS = {"content-length", "content-type", "host", "connection"}

MAX_TERMINAL_LINES = 2000
GEMINI_LOCK = threading.Lock()

SESSION_COOKIE = "ss"
SESSION_SECRET = os.getenv("SESSION_SECRET", "CHANGE_ME_LONG_RANDOM")
SER = URLSafeSerializer(SESSION_SECRET, salt="shortstudio")

# ===== INISIALISASI FASTAPI APP =====
app = FastAPI(title="ShortStudio")

# ===== BUAT DIREKTORI =====
os.makedirs(WORKSPACES_DIR, exist_ok=True)

# ===== MOUNT STATIC FILES =====
public_dir = os.path.join(BASE_DIR, "public")
if os.path.exists(public_dir):
    app.mount("/public", StaticFiles(directory=public_dir), name="public")

# ===== LOG BUS =====
class LogBus:
    def __init__(self) -> None:
        self._queues: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=MAX_TERMINAL_LINES))

    def push(self, wid: str, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._queues[wid].append(f"[{ts}] {line}")

    def pop_since(self, wid: str, last_idx: int) -> Tuple[List[str], int]:
        q = list(self._queues[wid])
        if last_idx < 0:
            last_idx = 0
        if last_idx > len(q):
            last_idx = len(q)
        return q[last_idx:], len(q)

    def clear(self, wid: str) -> None:
        self._queues[wid].clear()

BUS = LogBus()

def _now() -> float:
    return time.time()

def _new_sess() -> dict:
    return {
        "created": _now(),
        "license_ok": False if LICENSE_REQUIRED else True,
        "is_owner": False,
        "workspace_id": None,
        "gemini_key": None,
        "eleven_key": None,
    }

def _load_sess_from_cookie(request: Request) -> dict:
    raw = request.cookies.get(SESSION_COOKIE)
    if not raw:
        return _new_sess()
    try:
        s = SER.loads(raw)
        if not isinstance(s, dict):
            return _new_sess()
        base = _new_sess()
        base.update(s)
        return base
    except BadSignature:
        return _new_sess()

def _save_sess_to_cookie(resp, sess: dict, request: Request):
    val = SER.dumps(sess)
    resp.set_cookie(
        SESSION_COOKIE,
        val,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_HOURS * 3600,
        secure=(request.url.scheme == "https"),
        path="/",
    )

def _get_session(request: Request) -> dict:
    return request.state.sess

@app.middleware("http")
async def session_mw(request: Request, call_next):
    request.state.sess = _load_sess_from_cookie(request)
    request.state.skip_cookie_save = False

    resp = await call_next(request)

    if getattr(request.state, "skip_cookie_save", False):
        return resp

    _save_sess_to_cookie(resp, request.state.sess, request)
    return resp

def _require_license(sess: dict):
    if LICENSE_REQUIRED and not sess.get("license_ok"):
        raise HTTPException(status_code=401, detail="LICENSE_REQUIRED")

@dataclass
class WorkspaceMeta:
    workspace_id: str
    created_at: float
    is_persistent: bool
    channel_name: Optional[str] = None

def _ws_dir(wid: str) -> str:
    return os.path.join(WORKSPACES_DIR, wid)

def _save_meta(meta: WorkspaceMeta) -> None:
    d = _ws_dir(meta.workspace_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, ensure_ascii=False, indent=2)

def _load_meta(wid: str) -> WorkspaceMeta:
    p = os.path.join(_ws_dir(wid), "meta.json")
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    return WorkspaceMeta(**j)

def _write_json(wid: str, name: str, data: Any) -> str:
    p = os.path.join(_ws_dir(wid), name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p

def _read_json(wid: str, name: str) -> Any:
    p = os.path.join(_ws_dir(wid), name)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_text(wid: str, name: str, text: str) -> str:
    p = os.path.join(_ws_dir(wid), name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p

def _list_files(wid: str) -> List[str]:
    d = _ws_dir(wid)
    out = []
    for root, _, files in os.walk(d):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), d).replace("\\", "/")
            out.append(rel)
    return sorted(out)

def _reset_workspace(wid: str) -> None:
    d = _ws_dir(wid)
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

    keep = {"meta.json"}
    for root, _, files in os.walk(d):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), d).replace("\\", "/")
            if rel in keep:
                continue
            try:
                os.remove(os.path.join(root, fn))
            except Exception:
                pass

    for sub in ["sound", "animasi"]:
        shutil.rmtree(os.path.join(d, sub), ignore_errors=True)

    try:
        meta = _load_meta(wid)
        meta.channel_name = None
        _save_meta(meta)
    except Exception:
        pass

    BUS.clear(wid)

def _cleanup_expired_workspaces():
    ttl = WORKSPACE_TTL_HOURS * 3600
    now = _now()
    if not os.path.isdir(WORKSPACES_DIR):
        return
    for wid in os.listdir(WORKSPACES_DIR):
        d = _ws_dir(wid)
        meta_path = os.path.join(d, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("is_persistent"):
                continue
            created = float(meta.get("created_at", 0))
            if now - created > ttl:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            continue

def _ensure_workspace(sess: dict) -> str:
    _cleanup_expired_workspaces()

    wid = sess.get("workspace_id")
    if wid:
        d = _ws_dir(wid)
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            meta = WorkspaceMeta(
                workspace_id=wid,
                created_at=_now(),
                is_persistent=bool(sess.get("is_owner", False)),
            )
            _save_meta(meta)
            BUS.push(wid, "Workspace folder hilang, dibuat ulang (serverless cold start).")
        return wid

    wid = uuid.uuid4().hex
    os.makedirs(_ws_dir(wid), exist_ok=True)
    meta = WorkspaceMeta(
        workspace_id=wid,
        created_at=_now(),
        is_persistent=bool(sess.get("is_owner", False)),
    )
    _save_meta(meta)
    sess["workspace_id"] = wid
    BUS.push(wid, "Workspace dibuat.")
    return wid

def _verify_license(key: str) -> Tuple[bool, bool]:
    if not LICENSE_REQUIRED:
        return True, False
    key = (key or "").strip()
    if not key:
        return False, False
    if key == OWNER_LICENSE:
        return True, True
    return True, False

def _mask_tokens(obj: Any) -> Any:
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            lk = k.lower()
            if lk in ("authorization", "x-anonymous-auth"):
                new[k] = "***MASKED***"
            elif lk in ("x-api-key",):
                new[k] = "***MASKED_API_KEY***"
            elif lk in ("resource_token",):
                new[k] = "***MASKED_RESOURCE_TOKEN***"
            else:
                new[k] = _mask_tokens(v)
        return new
    if isinstance(obj, list):
        return [_mask_tokens(x) for x in obj]
    if isinstance(obj, str):
        return re.sub(r"Bearer\s+[A-Za-z0-9\-\._]+", "Bearer ***MASKED***", obj)
    return obj

def _safe_join(wid: str, relpath: str) -> str:
    root = os.path.abspath(_ws_dir(wid))
    full = os.path.abspath(os.path.normpath(os.path.join(root, relpath)))
    if os.path.commonpath([root, full]) != root:
        raise HTTPException(status_code=400, detail="INVALID_PATH")
    return full

@app.get("/", response_class=HTMLResponse)
def home():
    try:
        index_path = BASE_DIR / "public" / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        else:
            return HTMLResponse(
                content="""
                <!DOCTYPE html>
                <html>
                <head><title>ShortStudio API</title></head>
                <body>
                    <h1>ShortStudio API</h1>
                    <p>Server is running. Public files not found in deployment.</p>
                    <p>API Endpoints:</p>
                    <ul>
                        <li>POST /api/auth/license</li>
                        <li>POST /api/settings/api-keys</li>
                        <li>GET /api/settings/status</li>
                    </ul>
                </body>
                </html>
                """,
                status_code=200
            )
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>ShortStudio API</h1><p>Error: {str(e)}</p>",
            status_code=500
        )

@app.post("/api/auth/license")
async def auth_license(request: Request):
    sess = _get_session(request)
    body = await request.json()
    ok, is_owner = _verify_license(body.get("license_key"))
    if not ok:
        raise HTTPException(status_code=401, detail="LICENSE_INVALID")
    sess["license_ok"] = True
    sess["is_owner"] = is_owner
    wid = _ensure_workspace(sess)
    BUS.push(wid, f"License OK. Owner={is_owner}")
    return {"ok": True, "is_owner": is_owner}

@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    request.state.skip_cookie_save = True

    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp

@app.post("/api/settings/api-keys")
async def set_keys(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    body = await request.json()
    gem = (body.get("gemini_key") or "").strip()
    elv = (body.get("eleven_key") or "").strip()
    if not gem or not elv:
        raise HTTPException(status_code=400, detail="MISSING_KEYS")
    wid = _ensure_workspace(sess)
    sess["gemini_key"] = gem
    sess["eleven_key"] = elv
    BUS.push(wid, "Apikey diset (session-only).")
    return {"ok": True}

@app.get("/api/settings/status")
async def settings_status(request: Request):
    sess = _get_session(request)
    if LICENSE_REQUIRED and not sess.get("license_ok"):
        return {"licensed": False}
    wid = _ensure_workspace(sess)
    return {
        "licensed": True,
        "is_owner": bool(sess.get("is_owner")),
        "has_gemini": bool(sess.get("gemini_key")),
        "has_eleven": bool(sess.get("eleven_key")),
        "workspace_id": wid,
    }

@app.get("/api/events/stream")
async def events_stream(request: Request, last: int = 0):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)

    async def gen():
        nonlocal last
        import asyncio

        while True:
            if await request.is_disconnected():
                break
            lines, new_last = BUS.pop_since(wid, last)
            last = new_last
            for ln in lines:
                yield f"data: {ln}\n\n"
            await asyncio.sleep(0.35)

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/api/events/clear")
async def events_clear(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)
    BUS.clear(wid)
    return {"ok": True}

def _load_sound_reference(wid: str) -> List[str]:
    if SOUND_TXT_URL:
        BUS.push(wid, f"Mengambil sound.txt")
        try:
            s = requests.Session()
            s.trust_env = False
            s.proxies = {}
            r = s.get(SOUND_TXT_URL, timeout=30, headers={"User-Agent": "ShortStudioWeb/1.0"})
            r.raise_for_status()
            lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
            BUS.push(wid, f"sound.txt loaded dari URL: {len(lines)} baris.")
            if lines:
                return lines
        except Exception as e:
            BUS.push(wid, f"WARNING: gagal fetch SOUND_TXT_URL: {repr(e)}")

    local_path = SOUND_TEXT_FILE_NAME
    if os.path.isfile(local_path):
        BUS.push(wid, "Mengambil sound.txt lokal dari repo...")
        with open(local_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        BUS.push(wid, f"sound.txt loaded lokal: {len(lines)} baris.")
        return lines

    BUS.push(wid, "ERROR: sound.txt tidak ditemukan.")
    raise RuntimeError("sound.txt tidak tersedia.")

def _generate_scripts(wid: str, gemini_key: str, channel_name: str, count: int, sound_lines: List[str]) -> List[str]:
    reference_text = "\n".join(f"- {t}" for t in sound_lines)
    prompt = f"""
BERIKUT ADALAH CONTOH NASKAH KONTEN SHORT / REELS
YANG MENJADI ACUAN GAYA BAHASA:
{reference_text}

BERDASARKAN POLA DAN GAYA DI ATAS,
BUATKAN {count} NASKAH BARU DENGAN KETENTUAN:
- GUNAKAN NAMA "{channel_name}" JIKA RELEVAN.
- FORMAT: FULL HURUF KAPITAL.
- DURASI: < 15 DETIK.
- GAYA: PERSUASIF, AJAKAN LANGSUNG.
- KALIMAT HARUS BARU.
OUTPUT HANYA LIST NASKAH.
""".strip()

    BUS.push(wid, "Init Gemini...")
    with GEMINI_LOCK:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        BUS.push(wid, "Request ke Gemini...")
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.8),
        )

    text = (resp.text or "").strip()
    if not text:
        BUS.push(wid, "WARNING: Gemini response kosong.")
        return []

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cleaned: List[str] = []
    for ln in lines:
        ln = re.sub(r"^\s*[\-\*\d\.\)]\s*", "", ln).strip()
        if ln:
            cleaned.append(ln)

    BUS.push(wid, f"Gemini selesai: {len(cleaned)} naskah.")
    return cleaned

@app.post("/api/step1/generate-text")
async def step1_generate_text(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)

    if not sess.get("gemini_key"):
        raise HTTPException(status_code=400, detail="GEMINI_KEY_NOT_SET")

    body = await request.json()
    channel = (body.get("channel_name") or "").strip()
    count = int(body.get("count") or 0)
    if not channel or count <= 0 or count > 80:
        raise HTTPException(status_code=400, detail="INVALID_INPUT")

    try:
        sound_lines = _load_sound_reference(wid)
        scripts = _generate_scripts(wid, sess["gemini_key"], channel, count, sound_lines)
    except Exception as e:
        BUS.push(wid, f"Generate text gagal: {repr(e)}")
        raise HTTPException(status_code=400, detail="GEMINI_ERROR")

    _write_json(wid, "generated_texts.json", {"channel": channel, "texts": scripts})

    out_lines = []
    out_lines.append("=" * 40)
    out_lines.append(f"CHANNEL : {channel}")
    out_lines.append(f"TOTAL : {len(scripts)}")
    out_lines.append("=" * 40)
    out_lines.append("")
    for i, t in enumerate(scripts, 1):
        out_lines.append(f"[{i:02d}] {t}")
    out_lines.append("")
    _write_text(wid, "output_sound.txt", "\n".join(out_lines))

    meta = _load_meta(wid)
    meta.channel_name = channel
    _save_meta(meta)

    return {"ok": True, "count": len(scripts)}

@app.get("/api/step1/texts")
async def step1_get_texts(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)
    try:
        j = _read_json(wid, "generated_texts.json")
        return {"ok": True, "channel": j.get("channel"), "texts": j.get("texts", [])}
    except Exception:
        return {"ok": True, "channel": None, "texts": []}

def _eleven_stream_tts(api_key: str, text: str):
    s = requests.Session()
    s.trust_env = False
    s.proxies = {}

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}/stream"
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "User-Agent": "ShortStudioWeb/1.0",
    }
    payload = {
        "text": text,
        "model_id": ELEVEN_MODEL_ID,
        "voice_settings": {
            "speed": 0.93,
            "stability": 0.55,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }

    r = s.post(url, headers=headers, json=payload, stream=True, timeout=(20, 180))
    return r

def _tts_generate(wid: str, eleven_key: str, texts: List[str], selected_indexes: List[int]) -> dict:
    tts_dir = os.path.join(_ws_dir(wid), "sound")
    os.makedirs(tts_dir, exist_ok=True)

    meta = _load_meta(wid)
    channel = (meta.channel_name or "CHANNEL").strip()
    safe_channel = re.sub(r"[^A-Za-z0-9_\-]+", "_", channel).strip("_")[:40] or "CHANNEL"

    # hanya kalau 1 item: kita buat data_url (anti serverless)
    want_data_url = (len(selected_indexes) == 1)
    primary_data_url = None
    primary_relpath = None

    generated = []
    for idx in selected_indexes:
        text = texts[idx]
        BUS.push(wid, f"Sound generating [{idx+1}] ...")

        r = _eleven_stream_tts(eleven_key, text)

        if r.status_code == 401:
            BUS.push(wid, f"ERROR ElevenLabs 401: {r.text[:200]}")
            raise RuntimeError("ELEVEN_401_UNAUTHORIZED_OR_BLOCKED")
        if r.status_code >= 400:
            BUS.push(wid, f"ERROR ElevenLabs HTTP {r.status_code}: {r.text[:200]}")
            raise RuntimeError(f"ELEVEN_HTTP_{r.status_code}")

        fn = f"sound_{idx+1:03}.mp3"
        path = os.path.join(tts_dir, fn)
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 64):
                if chunk:
                    f.write(chunk)

        rel = f"sound/{fn}"
        BUS.push(wid, f"Sound saved: {fn}")
        generated.append({"index": idx, "filename": fn, "relpath": rel})

        # kalau cuma 1 file, bikin data_url untuk MP3 itu
        if want_data_url and primary_data_url is None:
            try:
                with open(path, "rb") as fp:
                    b = fp.read()
                primary_data_url = "data:audio/mpeg;base64," + base64.b64encode(b).decode("ascii")
                primary_relpath = rel
                BUS.push(wid, f"Primary data_url siap untuk: {rel}")
            except Exception as e:
                BUS.push(wid, f"WARNING: gagal buat data_url mp3: {repr(e)}")

    channel_mp3_rel = None
    if len(generated) == 1:
        src = os.path.join(_ws_dir(wid), generated[0]["relpath"])
        dst_name = f"sound_{safe_channel}.mp3"
        dst = os.path.join(tts_dir, dst_name)
        shutil.copyfile(src, dst)
        channel_mp3_rel = f"sound/{dst_name}"
        BUS.push(wid, f"Also saved: {dst_name} (channel-named)")

    zip_rel = None
    if len(generated) > 1:
        zip_name = f"sound_{safe_channel}.zip"
        zip_path = os.path.join(tts_dir, zip_name)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for g in generated:
                z.write(os.path.join(_ws_dir(wid), g["relpath"]), arcname=g["filename"])
        zip_rel = f"sound/{zip_name}"
        BUS.push(wid, f"ZIP dibuat: {zip_name}")

    return {
        "generated": generated,
        "zip_rel": zip_rel,
        "channel_mp3_rel": channel_mp3_rel,
        "primary_data_url": primary_data_url,
        "primary_relpath": primary_relpath,
    }

@app.post("/api/step2/generate-tts")
async def step2_generate_tts(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)

    if not sess.get("eleven_key"):
        raise HTTPException(status_code=400, detail="ELEVEN_KEY_NOT_SET")

    try:
        j = _read_json(wid, "generated_texts.json")
        texts = j.get("texts", [])
    except Exception:
        raise HTTPException(status_code=400, detail="NO_TEXTS_YET")

    body = await request.json()
    indexes = body.get("indexes") or []
    if not indexes:
        raise HTTPException(status_code=400, detail="NO_SELECTION")

    idxs = sorted(set(int(x) for x in indexes))
    idxs = [i for i in idxs if 0 <= i < len(texts)]
    if not idxs:
        raise HTTPException(status_code=400, detail="INVALID_SELECTION")

    try:
        res = _tts_generate(wid, sess["eleven_key"], texts, idxs)
    except Exception as e:
        BUS.push(wid, f"Generate sound gagal: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    _write_json(
        wid,
        "generated_audio.json",
        {
            "selected_text_indexes": idxs,
            "generated": res["generated"],
            "zip": res["zip_rel"],
            "channel_mp3": res["channel_mp3_rel"],
            # simpan data_url cuma untuk kasus 1 mp3 (biar gak bengkak)
            "primary_data_url": res["primary_data_url"],
            "primary_relpath": res["primary_relpath"],
        },
    )
    return {
        "ok": True,
        "generated": len(res["generated"]),
        "zip": bool(res["zip_rel"]),
        "channel_mp3": res["channel_mp3_rel"],
        "primary_data_url": res["primary_data_url"],
        "primary_relpath": res["primary_relpath"],
    }

@app.get("/api/step2/audios")
async def step2_get_audios(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)
    try:
        j = _read_json(wid, "generated_audio.json")
        return {"ok": True, **j}
    except Exception:
        return {
            "ok": True,
            "generated": [],
            "zip": None,
            "channel_mp3": None,
            "selected_text_indexes": [],
            "primary_data_url": None,
            "primary_relpath": None,
        }

def _headers_list_to_dict(hlist):
    out = {}
    for h in hlist or []:
        n = h.get("name")
        v = h.get("value")
        if n:
            out[n] = v
    return out

def _cookies_list_to_dict(clist):
    out = {}
    for c in clist or []:
        n = c.get("name")
        v = c.get("value")
        if n is not None:
            out[n] = v
    return out

def _parse_cookie_header(cookie_header_value: str):
    out = {}
    if not cookie_header_value:
        return out
    for part in cookie_header_value.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def _find_last_submit_entry(entries):
    candidates = []
    for e in entries:
        req = e.get("request", {})
        url = req.get("url", "")
        method = (req.get("method") or "").upper()
        if method == "POST" and urlparse(url).path.endswith(TARGET_SUBMIT_PATH):
            candidates.append(e)
    return candidates[-1] if candidates else None

def _parse_multipart_from_text(raw_text: str):
    if not raw_text:
        return {}
    first_line_end = raw_text.find("\r\n")
    if first_line_end == -1:
        return {}
    boundary = raw_text[:first_line_end]
    if not boundary.startswith("----"):
        m = re.search(r"(-{4,}[^ \r\n]+)", raw_text)
        if not m:
            return {}
        boundary = m.group(1)

    parts = raw_text.split(boundary)
    out = {}
    for p in parts:
        p = p.strip()
        if not p or p == "--":
            continue
        p = p.lstrip("\r\n")
        if p.endswith("--"):
            p = p[:-2].rstrip()
        sep = "\r\n\r\n"
        if sep not in p:
            continue
        head, body = p.split(sep, 1)
        m = re.search(r'Content-Disposition:\s*form-data;\s*name="([^"]+)"', head, re.I)
        if not m:
            continue
        name = m.group(1)
        out[name] = body.rstrip("\r\n")
    return out

def _extract_from_har(wid: str, har_json: dict) -> dict:
    BUS.push(wid, "Parsing HAR...")
    entries = har_json.get("log", {}).get("entries", [])
    if not entries:
        raise RuntimeError("HAR tidak punya entries.")

    submit_entry = _find_last_submit_entry(entries)
    if not submit_entry:
        raise RuntimeError("Tidak ketemu request POST /chx/submit di HAR.")

    req = submit_entry.get("request", {})
    hdict = _headers_list_to_dict(req.get("headers", []))

    clean_headers = {}
    for k, v in hdict.items():
        if k.lower() in DROP_HEADER_KEYS:
            continue
        clean_headers[k] = v

    cookies = _cookies_list_to_dict(req.get("cookies", []))
    cookie_hdr = None
    for k, v in hdict.items():
        if k.lower() == "cookie":
            cookie_hdr = v
            break
    if cookie_hdr:
        cookies.update(_parse_cookie_header(cookie_hdr))

    postdata = req.get("postData", {}) or {}
    project_payload_raw = None
    user_choices_raw = None

    params = postdata.get("params")
    if isinstance(params, list):
        for p in params:
            if p.get("name") == "project_payload":
                project_payload_raw = p.get("value")
            if p.get("name") == "user_choices":
                user_choices_raw = p.get("value")

    if not project_payload_raw:
        fields = _parse_multipart_from_text(postdata.get("text", "") or "")
        project_payload_raw = fields.get("project_payload")
        user_choices_raw = fields.get("user_choices")

    if not project_payload_raw:
        raise RuntimeError("Tidak ketemu project_payload di HAR (params maupun text).")

    try:
        project_payload = json.loads(project_payload_raw)
    except Exception:
        project_payload = project_payload_raw

    user_choices = None
    if user_choices_raw:
        try:
            user_choices = json.loads(user_choices_raw)
        except Exception:
            user_choices = user_choices_raw

    BUS.push(wid, "HAR extracted: headers/cookies/payload siap.")
    return {
        "headers": clean_headers,
        "cookies": cookies,
        "submit_payload": project_payload,
        "user_choices": user_choices,
        "source_url": req.get("url"),
    }

@app.post("/api/step3/upload-har")
async def step3_upload_har(request: Request, har: UploadFile = File(...)):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)

    if not har.filename.lower().endswith(".har"):
        raise HTTPException(status_code=400, detail="FILE_MUST_BE_HAR")

    raw = await har.read()
    try:
        har_json = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        raise HTTPException(status_code=400, detail="INVALID_HAR_JSON")

    extracted = _extract_from_har(wid, har_json)

    _write_json(wid, "headers.json", extracted["headers"])
    _write_json(wid, "cookies.json", extracted["cookies"])
    _write_json(wid, "submit_payload.json", extracted["submit_payload"])
    if extracted.get("user_choices") is not None:
        _write_json(wid, "user_choices.json", extracted["user_choices"])

    BUS.push(wid, f"HAR OK dari: {extracted.get('source_url')}")
    return {"ok": True}

@app.post("/api/step3/upload-extracted")
async def step3_upload_extracted(
    request: Request,
    headers: UploadFile = File(...),
    cookies: UploadFile = File(...),
    submit_payload: UploadFile = File(...),
    user_choices: Optional[UploadFile] = File(None),
):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)

    import asyncio

    async def read_json_file(up: UploadFile, label: str):
        raw = await up.read()
        try:
            return json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            raise HTTPException(status_code=400, detail=f"INVALID_JSON_{label}")

    BUS.push(wid, "Upload extracted JSON...")

    h = await read_json_file(headers, "HEADERS")
    c = await read_json_file(cookies, "COOKIES")
    p = await read_json_file(submit_payload, "SUBMIT_PAYLOAD")

    if not isinstance(h, dict):
        raise HTTPException(status_code=400, detail="HEADERS_MUST_BE_OBJECT")
    if not isinstance(c, dict):
        raise HTTPException(status_code=400, detail="COOKIES_MUST_BE_OBJECT")
    if not isinstance(p, dict):
        raise HTTPException(status_code=400, detail="SUBMIT_PAYLOAD_MUST_BE_OBJECT")

    _write_json(wid, "headers.json", h)
    _write_json(wid, "cookies.json", c)
    _write_json(wid, "submit_payload.json", p)

    if user_choices is not None:
        uc = await read_json_file(user_choices, "USER_CHOICES")
        _write_json(wid, "user_choices.json", uc)

    BUS.push(wid, "Extracted JSON saved: headers/cookies/submit_payload/user_choices.")
    return {"ok": True}

def _build_adobe_session(headers: dict, cookies: dict) -> requests.Session:
    sess = requests.Session()
    sess.trust_env = False
    sess.proxies = {}

    h = dict(headers or {})
    for drop in ["Content-Type", "Content-Length", "Host", "Connection"]:
        h.pop(drop, None)

    h.setdefault("Origin", "https://new.express.adobe.com")
    h.setdefault("Referer", "https://new.express.adobe.com/")
    h.setdefault("Accept", "*/*")
    sess.headers.update(h)

    if isinstance(cookies, dict):
        for k, v in cookies.items():
            sess.cookies.set(k, v, domain=".adobe.io")

    return sess

def _adobe_upload_asset(sess: requests.Session, wid: str, file_path: str, filename: str, mime: str) -> str:
    with open(file_path, "rb") as fp:
        files = {"file": (filename, fp, mime)}
        r = sess.post(ADOBE_UPLOAD_URL, files=files, timeout=(20, 180), headers=CLOSE_HEADERS)
    BUS.push(wid, f"assets/upload => HTTP {r.status_code}")
    r.raise_for_status()
    js = r.json()
    items = js.get("items") or []
    if not items:
        raise RuntimeError(f"Upload sukses tapi items kosong: {_mask_tokens(js)}")
    return items[0]

def _adobe_get_download_url(sess: requests.Session, wid: str, resource_token: str) -> str:
    r = sess.post(
        ADOBE_DOWNLOAD_URLS_API,
        json={"assets": [{"resource_token": resource_token}]},
        timeout=(20, 120),
        headers=CLOSE_HEADERS,
    )
    BUS.push(wid, f"assets/download_urls => HTTP {r.status_code}")
    r.raise_for_status()
    js = r.json()
    items = js.get("items") or []
    if not items or not items[0].get("download_url"):
        raise RuntimeError(f"download_urls invalid: {_mask_tokens(js)}")
    return items[0]["download_url"]

def _adobe_submit_job(sess: requests.Session, wid: str, project_payload: dict, user_choices: dict) -> Tuple[str, dict]:
    BUS.push(wid, "Submit job ke Adobe /chx/submit (multipart)...")
    files = {
        "project_payload": (None, json.dumps(project_payload), "application/json"),
        "user_choices": (None, json.dumps(user_choices), "application/json"),
    }
    r = sess.post(ADOBE_SUBMIT_URL, files=files, timeout=(20, 180), headers=CLOSE_HEADERS)
    BUS.push(wid, f"submit => HTTP {r.status_code}")
    r.raise_for_status()
    j = r.json()
    job_id = j.get("job_id") or j.get("jobId")
    if not job_id:
        raise RuntimeError(f"job_id tidak ada: {_mask_tokens(j)}")
    return job_id, j

def _adobe_polling_url(job_id: str, job_json: dict) -> str:
    polling_url = f"{ADOBE_BASE}/chx/{job_id}"
    uri = job_json.get("_links", {}).get("job_status", {}).get("uri")
    if uri:
        polling_url = ADOBE_BASE + uri
    return polling_url

def _adobe_poll_until_success(sess: requests.Session, wid: str, polling_url: str, max_minutes: int = 30) -> str:
    BUS.push(wid, f"Polling: {polling_url}")
    start = _now()
    backoff = 3.0

    while True:
        if (_now() - start) > max_minutes * 60:
            raise RuntimeError("Timeout polling > max_minutes.")
        time.sleep(random.uniform(backoff, backoff + 1.5))

        r = sess.get(polling_url, timeout=(20, 25), headers=CLOSE_HEADERS)
        BUS.push(wid, f"poll => HTTP {r.status_code}")

        if r.status_code in (502, 503, 504):
            backoff = min(backoff * 1.2, 20.0)
            continue
        if r.status_code == 429:
            time.sleep(15)
            continue

        r.raise_for_status()
        j = r.json()
        status = j.get("status")
        BUS.push(wid, f"Status: {status}")

        if status in ("SUCCESS", "COMPLETED"):
            out = j.get("output", {})
            locs = out.get("output_location") or []
            if locs and locs[0].get("path"):
                return locs[0]["path"]
            raise RuntimeError(f"SUCCESS tapi output_location kosong: {_mask_tokens(j)}")

        if status in ("FAILED", "ERROR"):
            raise RuntimeError(f"Render gagal: {_mask_tokens(j)}")

        backoff = 3.0

def _download_signed(url: str, out_path: str) -> None:
    dl = requests.Session()
    dl.trust_env = False
    dl.proxies = {}
    with dl.get(url, stream=True, timeout=(20, 300)) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

def _run_render_flow(wid: str, mp3_relpath: str) -> dict:
    headers = _read_json(wid, "headers.json")
    cookies = _read_json(wid, "cookies.json")
    submit_payload = _read_json(wid, "submit_payload.json")
    try:
        user_choices = _read_json(wid, "user_choices.json")
    except Exception:
        user_choices = None

    mp3_abs = _safe_join(wid, mp3_relpath)
    if not os.path.isfile(mp3_abs):
        raise RuntimeError("MP3 tidak ditemukan di workspace.")

    mp3_filename = os.path.basename(mp3_abs)
    base_name = mp3_filename.rsplit(".", 1)[0]

    sess = _build_adobe_session(headers, cookies)

    BUS.push(wid, "Upload MP3 ke Adobe...")
    token = _adobe_upload_asset(sess, wid, mp3_abs, mp3_filename, "audio/mpeg")
    mp3_remote = _adobe_get_download_url(sess, wid, token)

    payload = submit_payload
    if isinstance(payload, dict) and "input_media" in payload:
        if "downloaded_file_name" in payload:
            payload["downloaded_file_name"] = base_name
        media_datas = payload["input_media"][0]["media_data"]
        for md in media_datas:
            if (md.get("mime_type") or "").lower() == "audio/mpeg":
                md["path"] = {"local": mp3_filename, "remote": mp3_remote}
                if "name" in md:
                    md["name"] = mp3_filename
    else:
        raise RuntimeError("submit_payload.json format tidak sesuai (bukan dict payload Adobe).")

    uc = user_choices if isinstance(user_choices, dict) else {"version": 1, "output_file_name": base_name}
    if "output_file_name" in uc:
        uc["output_file_name"] = base_name

    job_id, job_json = _adobe_submit_job(sess, wid, payload, uc)
    polling_url = _adobe_polling_url(job_id, job_json)

    signed_url = _adobe_poll_until_success(sess, wid, polling_url, max_minutes=30)

    out_dir = os.path.join(_ws_dir(wid), "animasi")
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"{base_name}.mp4")
    BUS.push(wid, "Download MP4 dari signed URL...")
    _download_signed(signed_url, out_file)
    BUS.push(wid, f"Render selesai: animasi/{os.path.basename(out_file)}")

    mp4_rel = f"animasi/{os.path.basename(out_file)}"

    _write_text(
        wid,
        "animasi.txt",
        f"OUTPUT_MP4={mp4_rel}\nSOURCE_MP3={mp3_relpath}\nSIGNED_URL={signed_url}\n"
    )

    # simpan juga json kecil agar UI bisa ambil signed_url setelah reload
    _write_json(
        wid,
        "render_last.json",
        {"mp4_rel": mp4_rel, "signed_url": signed_url, "source_mp3": mp3_relpath, "ts": int(_now())}
    )

    return {"mp4_rel": mp4_rel, "signed_url": signed_url}

@app.post("/api/step4/render")
async def step4_render(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)

    missing = []
    for reqf in ("headers.json", "cookies.json", "submit_payload.json"):
        if not os.path.isfile(_safe_join(wid, reqf)):
            missing.append(reqf)

    if missing:
        BUS.push(wid, f"HAR_NOT_READY missing={missing} files_now={_list_files(wid)}")
        raise HTTPException(status_code=400, detail={"code": "HAR_NOT_READY", "missing": missing})

    body = await request.json()
    mp3_rel = (body.get("mp3_relpath") or "").strip()
    if not mp3_rel:
        raise HTTPException(status_code=400, detail="MP3_NOT_SELECTED")

    try:
        res = _run_render_flow(wid, mp3_rel)
    except requests.HTTPError as e:
        BUS.push(wid, f"ERROR Adobe HTTP: {str(e)}")
        raise HTTPException(status_code=400, detail="ADOBE_HTTP_ERROR")
    except Exception as e:
        BUS.push(wid, f"ERROR: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    return {"ok": True, "mp4": res["mp4_rel"], "signed_url": res["signed_url"]}

@app.get("/api/step4/last")
async def step4_last(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)
    try:
        j = _read_json(wid, "render_last.json")
        return {"ok": True, **j}
    except Exception:
        return {"ok": True, "mp4_rel": None, "signed_url": None, "source_mp3": None}

@app.get("/api/artifacts")
async def artifacts(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)
    return {"ok": True, "workspace_id": wid, "files": _list_files(wid)}

@app.post("/api/workspace/reset")
async def workspace_reset(request: Request):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)

    BUS.push(wid, "Reset workspace diminta (Generate Lagi).")
    _reset_workspace(wid)
    BUS.push(wid, "Workspace direset: kembali fresh seperti baru login.")
    return {"ok": True, "workspace_id": wid}

def _safe_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    name = name.replace("\r", "").replace("\n", "")
    name = name.replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9_\-\,\.\(\)\[\]\s]+", "_", name).strip()
    return name[:120]

@app.get("/api/download")
async def download(request: Request, path: str, name: str = ""):
    sess = _get_session(request)
    _require_license(sess)
    wid = _ensure_workspace(sess)

    full = _safe_join(wid, path)
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="NOT_FOUND")

    safe = _safe_filename(name)
    return FileResponse(full, filename=(safe or os.path.basename(full)))
