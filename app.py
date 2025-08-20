import os, json, time, math, hashlib, threading, random
from secrets import token_hex
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, make_response
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
import numpy as np

# =================== CARGA DE CONFIG ===================
def load_config(file_path="config.txt"):
    print(f"[CONFIG] Cargando configuración desde {file_path}...")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
        print("[CONFIG] Configuración cargada correctamente (archivo).")
    else:
        print("[CONFIG] No se encontró config.txt; usando variables de entorno de Azure/App Service.")

load_config("config.txt")

# =================== CONFIG GLOBAL ===================
DATA_DIR   = os.path.join(os.getcwd(), "data")
INDEX_NPZ  = os.path.join(DATA_DIR, "index.npz")
META_JSON  = os.path.join(DATA_DIR, "meta.json")
LOCK       = threading.Lock()
os.makedirs(DATA_DIR, exist_ok=True)

# ====== Ventanas y límites ======
WINDOW_SECS = 60
MAX_CALLS_PER_WINDOW = 10

# ====== Umbral de similitud ======
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.28"))

# ====== Límite de tokens por sesión ======
TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT_PER_SESSION", "10000"))
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "martindelgado@bcp.com.pe")

# ====== Mensajes y follow-up ======
GREETING_MSG = "Hola Soy el Asistente de Mapas de Carrera y estoy aqui para ayudarte en las consultas que tengas"
NOINFO_MSG = "Disculpa no tengo esta informacion pero puedes escribir al correo martindelgado@bcp.com.pe para que pueda aprender mas"
FOLLOWUP_DELAY_SECONDS = int(os.getenv("FOLLOWUP_DELAY_SECONDS", "15"))
FOLLOWUP_PROMPT = "¿Tienes alguna otra consulta? Marca Sí o No."
FINAL_CHAT_MSG = "Espero haberte ayudado, si tienes otra consulta solo refresca la pagina. Cuidate :)"

# =================== CLIENTES AZURE OPENAI ===================
client_chat = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_CHAT", os.getenv("AZURE_OPENAI_ENDPOINT")),
    api_key=os.getenv("AZURE_OPENAI_KEY_CHAT", os.getenv("AZURE_OPENAI_KEY")),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_CHAT", os.getenv("AZURE_OPENAI_API_VERSION")),
)
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

client_emb = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EMB", os.getenv("AZURE_OPENAI_ENDPOINT")),
    api_key=os.getenv("AZURE_OPENAI_KEY_EMB", os.getenv("AZURE_OPENAI_KEY")),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMB", os.getenv("AZURE_OPENAI_API_VERSION")),
)
EMB_DEPLOYMENT  = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

if not CHAT_DEPLOYMENT or not EMB_DEPLOYMENT:
    print("[STARTUP][WARN] Revisa deployment names: AZURE_OPENAI_DEPLOYMENT y AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

# =================== AZURE BLOB STORAGE ===================
blob_service = BlobServiceClient(
    account_url=f"https://{os.getenv('AZURE_STORAGE_ACCOUNT')}.blob.core.windows.net",
    credential=os.getenv("AZURE_STORAGE_KEY")
)
# Contenedor de documentos .txt
container = blob_service.get_container_client(os.getenv("AZURE_CONTAINER_NAME"))

# Contenedor de logs (puede ser el mismo que documentos)
LOGS_CONTAINER_NAME = os.getenv("AZURE_LOGS_CONTAINER_NAME", "").strip()
LOGS_PREFIX = os.getenv("LOGS_PREFIX", "logs").strip().strip("/")
# Carpeta separada para feedback
FEEDBACK_PREFIX = os.getenv("FEEDBACK_PREFIX", "feedback").strip().strip("/")

if LOGS_CONTAINER_NAME:
    logs_container = blob_service.get_container_client(LOGS_CONTAINER_NAME)
    try:
        logs_container.create_container()
        print(f"[LOGS] Contenedor '{LOGS_CONTAINER_NAME}' creado.")
    except Exception:
        print(f"[LOGS] Usando contenedor existente '{LOGS_CONTAINER_NAME}'.")
else:
    logs_container = container  # mismo contenedor

# =================== CIRCUIT BREAKER (anti 403) ===================
BLOCK_UNTIL_TS = 0
DEFAULT_BLOCK_SECONDS = 900

def is_block_active():
    return time.time() < BLOCK_UNTIL_TS

def activate_block(seconds=DEFAULT_BLOCK_SECONDS):
    global BLOCK_UNTIL_TS
    BLOCK_UNTIL_TS = max(BLOCK_UNTIL_TS, time.time() + seconds)
    print(f"[CIRCUIT] Activado bloqueo local por {seconds}s (hasta {int(BLOCK_UNTIL_TS)}).")

def parse_block_seconds_from_error(e):
    try:
        resp = getattr(e, "response", None)
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                return int(retry_after)
    except Exception:
        pass
    return DEFAULT_BLOCK_SECONDS

def is_unusual_behavior_403(e):
    msg = repr(e)
    return ("403" in msg) and ("unusual behavior" in msg.lower() or "temporarily blocked" in msg.lower())

# =================== SESIONES Y TOKENS ===================
SESS_TOKENS = {}  # {sid: used_tokens}

class TokenLimitError(Exception):
    pass

def _token_len(txt: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(txt))
    except Exception:
        return max(1, math.ceil(len(txt) / 4))

def get_or_create_sid():
    sid = request.cookies.get("SID")
    new = False
    if not sid:
        sid = token_hex(16)
        new = True
    if sid not in SESS_TOKENS:
        SESS_TOKENS[sid] = 0
    return sid, new

def tokens_used(sid):
    return int(SESS_TOKENS.get(sid, 0))

def ensure_tokens_available(sid, tokens_needed, stage=""):
    used = tokens_used(sid)
    if used + tokens_needed > TOKEN_LIMIT:
        print(f"[TOKENS] Límite por sesión alcanzado en stage='{stage}'. used={used}, need={tokens_needed}, limit={TOKEN_LIMIT}")
        raise TokenLimitError
    SESS_TOKENS[sid] = used + tokens_needed  # reserva lógica

def add_tokens(sid, n):
    SESS_TOKENS[sid] = tokens_used(sid) + int(n)

def human_limit_message():
    return f"Por cada conversación solo se dispone de {TOKEN_LIMIT} tokens. Para consultas adicionales, escríbenos a {SUPPORT_EMAIL}."

# =================== UTILIDADES RAG ===================
def chunk_text(text, max_tokens=350, overlap=60):
    words = text.split()
    chunks, cur, cur_tokens = [], [], 0
    for w in words:
        t = _token_len(w) + 1
        if cur_tokens + t > max_tokens and cur:
            chunk = " ".join(cur)
            chunks.append(chunk)
            take_back = max(0, len(cur) - overlap)
            cur = cur[take_back:]
            cur_tokens = _token_len(" ".join(cur))
        cur.append(w)
        cur_tokens += t
    if cur:
        chunks.append(" ".join(cur))
    return [c.strip() for c in chunks if c.strip()]

def read_txt_bytes(file_bytes: bytes) -> str:
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = file_bytes.decode("latin-1", errors="ignore")
    return text.strip()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return np.dot(a_norm, b_norm)

def embed_texts(texts, sid=None):
    print(f"[EMB] Generando embeddings para {len(texts)} textos...")
    if is_block_active():
        raise RuntimeError("Circuit breaker activo; omitiendo llamadas a embeddings.")

    total_input_tokens = sum(_token_len(t) for t in texts)
    if sid is not None:
        ensure_tokens_available(sid, total_input_tokens, stage="embeddings")

    out = []
    batch = 16
    for i in range(0, len(texts), batch):
        print(f"[EMB] Batch {i//batch + 1} ({i}-{min(i+batch, len(texts))})")
        for attempt in range(6):
            try:
                resp = client_emb.embeddings.create(
                    model=EMB_DEPLOYMENT,
                    input=texts[i:i+batch]
                )
                out.extend([d.embedding for d in resp.data])
                time.sleep(1.0)
                break
            except Exception as e:
                if is_unusual_behavior_403(e):
                    secs = parse_block_seconds_from_error(e)
                    print(f"[EMB][BLOCK] 403 unusual behavior. Activando bloqueo {secs}s.")
                    activate_block(secs)
                    raise
                wait = (2 ** attempt) + random.random()
                print(f"[EMB][ERROR] {repr(e)}; retry en {wait:.2f}s (intento {attempt+1}/6)")
                if attempt == 5:
                    print("[EMB][FATAL] Falló embeddings tras reintentos.")
                    raise
                time.sleep(wait)
    print("[EMB] Embeddings generados correctamente.")
    return np.array(out, dtype=np.float32)

def list_doc_blobs():
    return [b for b in container.list_blobs() if b.name.lower().endswith(".txt")]

def hash_blob_meta(name, etag, size):
    return hashlib.sha256(f"{name}:{etag}:{size}".encode()).hexdigest()

def load_index():
    if os.path.exists(INDEX_NPZ) and os.path.exists(META_JSON):
        try:
            data = np.load(INDEX_NPZ)
            with open(META_JSON, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"[INDEX] Índice cargado: {data['X'].shape[0]} vectores.")
            return data["X"], meta["chunks"], meta["sources"], meta["blob_meta"]
        except Exception as e:
            print(f"[INDEX][WARN] No se pudo cargar índice existente: {repr(e)}")
            return None
    print("[INDEX] No hay índice previo en disco.")
    return None

def save_index(X: np.ndarray, chunks, sources, blob_meta):
    np.savez_compressed(INDEX_NPZ, X=X)
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "sources": sources, "blob_meta": blob_meta}, f, ensure_ascii=False)
    print(f"[INDEX] Índice guardado ({len(chunks)} chunks / {X.shape[0]} vectores).")

def build_or_update_index(force=False):
    print(f"[INDEX] Iniciando proceso de indexado (force={force})...")
    if is_block_active():
        raise RuntimeError("Circuit breaker activo; no se puede indexar por ahora.")
    with LOCK:
        existing = load_index()
        X_old, chunks_old, sources_old, blob_meta_old = (None, [], [], {})
        if existing:
            X_old, chunks_old, sources_old, blob_meta_old = existing
            print(f"[INDEX] Índice existente con {len(chunks_old)} chunks.")

        blobs = list_doc_blobs()
        print(f"[INDEX] {len(blobs)} archivos .txt encontrados en el contenedor.")

        new_chunks, new_sources, new_blob_meta = [], [], {}

        for b in blobs:
            meta_hash = hash_blob_meta(b.name, b.etag, b.size)
            new_blob_meta[b.name] = {"etag": b.etag, "size": b.size, "hash": meta_hash}

        unchanged = (not force) and (blob_meta_old == new_blob_meta) and len(chunks_old) > 0
        if unchanged:
            print("[INDEX] Los documentos no han cambiado. Reutilizando índice existente.")
            return X_old, chunks_old, sources_old

        for b in blobs:
            print(f"[INDEX] Descargando: {b.name} ({b.size} bytes)...")
            blob = container.get_blob_client(b.name)
            content = blob.download_blob().readall()
            text = read_txt_bytes(content)
            if not text.strip():
                print(f"[INDEX][WARN] {b.name} sin texto utilizable.")
                continue
            chunks = chunk_text(text, max_tokens=350, overlap=60)
            print(f"[INDEX] {len(chunks)} chunks generados para {b.name}.")
            for i, c in enumerate(chunks):
                new_chunks.append(c)
                new_sources.append({"file": b.name, "chunk": i})

        if not new_chunks:
            raise RuntimeError("No se encontraron textos en los .txt")

        print("[INDEX] Creando embeddings para todos los chunks (indexación no consume tokens de sesión)...")
        X = embed_texts(new_chunks, sid=None)
        save_index(X, new_chunks, new_sources, new_blob_meta)
        print("[INDEX] Indexación finalizada.")
        return X, new_chunks, new_sources

def retrieve(query, k=6, sid=None):
    print(f"[RETRIEVE] Consulta: {query}")
    idx = load_index()
    if not idx:
        print("[RETRIEVE] No había índice, construyendo...")
        X, chunks, sources = build_or_update_index()
    else:
        X, _, _, _ = idx
        with open(META_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        chunks = meta["chunks"]
        sources = meta["sources"]

    print("[RETRIEVE] Generando embedding de la pregunta...")
    qv = embed_texts([query], sid=sid)[0]

    print("[RETRIEVE] Calculando similitudes coseno...")
    sims = cosine_sim(X, qv)
    top_idx = np.argsort(-sims)[:k]
    results = [{"text": chunks[i], "source": sources[i], "score": float(sims[i])} for i in top_idx]
    print(f"[RETRIEVE] Top-{k} chunks recuperados.")
    return results

def build_context(passages, max_context_tokens=1500, min_score=MIN_SIMILARITY):
    used, total = [], 0
    for p in passages:
        if p.get("score", 0.0) < min_score:
            continue
        t = _token_len(p["text"])
        if total + t > max_context_tokens:
            continue
        used.append(p)
        total += t
    ctx_lines = []
    for p in used:
        s = p["source"]
        tag = f"{s['file']}#chunk{str(s['chunk']).zfill(3)}"
        ctx_lines.append(f"[{tag}]\n{p['text']}")
    context = "\n\n---\n\n".join(ctx_lines)
    print(f"[CONTEXT] Contexto construido con {len(used)} pasajes (tokens aprox: {total}).")
    return context, used

def answer(question, sid, temperature=0.3, max_output_tokens=500):
    print(f"[ASK] Pregunta recibida: {question}")
    passages = retrieve(question, k=8, sid=sid)
    print(f"[ASK] {len(passages)} pasajes recuperados, construyendo contexto...")
    context, used = build_context(passages, max_context_tokens=1500)

    if not used or not context.strip():
        print("[ASK] Sin contexto relevante -> no se llama al modelo.")
        return NOINFO_MSG, {"no_context": True}

    system_msg = (
        "Imagina que eres un asistente humano que responde solo teniendo en cuenta el contexto proporcionado. "
        "Si no hay información suficiente, menciona que no puedes encontrar alguna respuesta y que lo vas a revisar... "
        "Sé amigable, responde sin mensajes de odio o discriminación, sin mencionar que eres un modelo de IA, y sin hacer preguntas al usuario."
    )
    user_msg = f"Contexto:\n{context}\n\nPregunta: {question}\nResponde de forma breve y precisa en español."

    prompt_tokens_est = _token_len(system_msg) + _token_len(user_msg)
    ensure_tokens_available(sid, prompt_tokens_est, stage="chat_prompt")

    print("[ASK] Llamando al modelo de chat...")
    for attempt in range(6):
        try:
            if is_block_active():
                raise RuntimeError("Circuit breaker activo; omitiendo llamadas a chat.")

            resp = client_chat.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            time.sleep(1.0)

            text = resp.choices[0].message.content or ""

            usage_total = None
            try:
                usage = getattr(resp, "usage", None)
                if usage and hasattr(usage, "total_tokens"):
                    usage_total = int(usage.total_tokens)
            except Exception:
                usage_total = None

            if usage_total is not None:
                extra = max(0, usage_total - prompt_tokens_est)
                add_tokens(sid, extra)
            else:
                add_tokens(sid, _token_len(text))

            print("[ASK] Respuesta recibida correctamente del modelo.")
            return text, {"no_context": False}
        except TokenLimitError:
            raise
        except Exception as e:
            if is_unusual_behavior_403(e):
                secs = parse_block_seconds_from_error(e)
                print(f"[ASK][BLOCK] 403 unusual behavior. Activando bloqueo {secs}s.")
                activate_block(secs)
                raise
            wait = (2 ** attempt) + random.random()
            print(f"[ASK][ERROR] {repr(e)}, retry en {wait:.2f}s (intento {attempt+1}/6)")
            if attempt == 5:
                print("[ASK][FATAL] Falló la llamada al modelo tras reintentos.")
                raise
            time.sleep(wait)

# =================== LOGS ===================
def _blob_ts_keys():
    ts = datetime.now(timezone.utc)
    ts_iso = ts.isoformat()
    ts_key = ts.strftime("%Y%m%d-%H%M%S-%f")
    return ts, ts_iso, ts_key

def save_conversation_to_blob(sid, question, answer, meta):
    try:
        _, ts_iso, ts_key = _blob_ts_keys()
        blob_name = f"{LOGS_PREFIX}/sessions/{sid}/{ts_key}.json" if not LOGS_CONTAINER_NAME else f"sessions/{sid}/{ts_key}.json"
        payload = {
            "timestamp_utc": ts_iso,
            "session_id": sid,
            "question": question,
            "answer": answer,
            "tokens_used_total": tokens_used(sid),
            "token_limit": TOKEN_LIMIT,
            "meta": meta or {}
        }
        logs_container.upload_blob(
            name=blob_name,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            overwrite=True
        )
        print(f"[LOGS] Conversación guardada en '{blob_name}'.")
    except Exception as e:
        print(f"[LOGS][WARN] No se pudo guardar la conversación: {repr(e)}")

def save_system_message_to_blob(sid, message, meta):
    try:
        _, ts_iso, ts_key = _blob_ts_keys()
        blob_name = f"{LOGS_PREFIX}/sessions/{sid}/system/{ts_key}.json" if not LOGS_CONTAINER_NAME else f"sessions/{sid}/system/{ts_key}.json"
        payload = {
            "timestamp_utc": ts_iso,
            "session_id": sid,
            "system_message": message,
            "tokens_used_total": tokens_used(sid),
            "token_limit": TOKEN_LIMIT,
            "meta": meta or {}
        }
        logs_container.upload_blob(
            name=blob_name,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            overwrite=True
        )
        print(f"[LOGS] Mensaje de sistema guardado en '{blob_name}'.")
    except Exception as e:
        print(f"[LOGS][WARN] No se pudo guardar el mensaje de sistema: {repr(e)}")

def save_feedback_to_blob(sid, feedback_text, meta):
    """Guarda el feedback en una carpeta distinta del log de conversación."""
    try:
        _, ts_iso, ts_key = _blob_ts_keys()
        blob_name = f"{FEEDBACK_PREFIX}/sessions/{sid}/{ts_key}.json"
        payload = {
            "timestamp_utc": ts_iso,
            "session_id": sid,
            "feedback": feedback_text,
            "meta": meta or {}
        }
        logs_container.upload_blob(
            name=blob_name,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            overwrite=True
        )
        print(f"[LOGS] Feedback guardado en '{blob_name}'.")
    except Exception as e:
        print(f"[LOGS][WARN] No se pudo guardar el feedback: {repr(e)}")

# =================== RATE LIMIT ===================
CALL_LOG = {}  # {ip: [timestamps]}

def check_rate(ip):
    now = time.time()
    CALL_LOG.setdefault(ip, [])
    CALL_LOG[ip] = [t for t in CALL_LOG[ip] if now - t < WINDOW_SECS]
    if len(CALL_LOG[ip]) >= MAX_CALLS_PER_WINDOW:
        retry = int(WINDOW_SECS - (now - CALL_LOG[ip][0]))
        print(f"[RATE] Límite alcanzado para {ip}. Retry en {retry}s.")
        return False, retry
    CALL_LOG[ip].append(now)
    return True, 0

# =================== FLASK APP ===================
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.get("/")
def home():
    print("[HTTP] GET / -> index.html")
    return render_template("index.html")

@app.get("/login")
def login_get():
    print("[HTTP] GET /login -> login.html")
    return render_template("login.html")

@app.route("/ask", methods=["POST"])
def ask():
    print("[HTTP] POST /ask - inicio")

    sid, new_sid = get_or_create_sid()

    if is_block_active():
        print("[HTTP] /ask - circuit breaker activo, devolviendo 503")
        resp = make_response(jsonify({
            "error": "service_temporarily_throttled",
            "message": "El servicio está regulando temporalmente el tráfico. Intenta de nuevo en unos minutos."
        }), 503)
        if new_sid:
            resp.set_cookie("SID", sid, httponly=True, samesite="Lax", max_age=60*60*24*7)
        return resp

    # Mejor IP detrás de proxy: usa X-Forwarded-For si viene
    ip = (request.headers.get("X-Forwarded-For", "") or "").split(",")[0].strip() or (request.remote_addr or "unknown")
    ok, wait_s = check_rate(ip)
    if not ok:
        print(f"[HTTP] /ask rate-limited, retry_in_seconds={wait_s}")
        resp = make_response(jsonify({"error": "rate_limited", "retry_in_seconds": wait_s}), 429)
        if new_sid:
            resp.set_cookie("SID", sid, httponly=True, samesite="Lax", max_age=60*60*24*7)
        return resp

    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        print("[HTTP] /ask - missing 'question'")
        resp = make_response(jsonify({"error": "missing 'question'"}), 400)
        if new_sid:
            resp.set_cookie("SID", sid, httponly=True, samesite="Lax", max_age=60*60*24*7)
        return resp

    if tokens_used(sid) >= TOKEN_LIMIT:
        msg = human_limit_message()
        output = msg
        save_conversation_to_blob(sid, question, output, {"limit_reached": True, "stage": "precheck"})
        resp = make_response(jsonify({
            "answer": output,
            "followup_suggested": True,
            "followup_after_seconds": FOLLOWUP_DELAY_SECONDS,
            "followup_prompt": FOLLOWUP_PROMPT,
            "followup_options": ["Sí", "No"]
        }), 200)
        if new_sid:
            resp.set_cookie("SID", sid, httponly=True, samesite="Lax", max_age=60*60*24*7)
        return resp

    try:
        ans, meta = answer(question, sid=sid)
        output = ans

        save_conversation_to_blob(sid, question, output, meta)
        resp = make_response(jsonify({
            "answer": output,
            "followup_suggested": True,
            "followup_after_seconds": FOLLOWUP_DELAY_SECONDS,
            "followup_prompt": FOLLOWUP_PROMPT,
            "followup_options": ["Sí", "No"]
        }), 200)
        if new_sid:
            resp.set_cookie("SID", sid, httponly=True, samesite="Lax", max_age=60*60*24*7)
        return resp
    except TokenLimitError:
        msg = human_limit_message()
        output = msg
        save_conversation_to_blob(sid, question, output, {"limit_reached": True, "stage": "processing"})
        resp = make_response(jsonify({
            "answer": output,
            "followup_suggested": True,
            "followup_after_seconds": FOLLOWUP_DELAY_SECONDS,
            "followup_prompt": FOLLOWUP_PROMPT,
            "followup_options": ["Sí", "No"]
        }), 200)
        if new_sid:
            resp.set_cookie("SID", sid, httponly=True, samesite="Lax", max_age=60*60*24*7)
        return resp
    except Exception as e:
        if is_unusual_behavior_403(e):
            secs = parse_block_seconds_from_error(e)
            activate_block(secs)
            print("[HTTP] /ask - 403 detectado, devolviendo 503")
            resp = make_response(jsonify({
                "error": "service_temporarily_throttled",
                "message": "El servicio está regulando temporalmente el tráfico. Intenta de nuevo más tarde."
            }), 503)
            if new_sid:
                resp.set_cookie("SID", sid, httponly=True, samesite="Lax", max_age=60*60*24*7)
            return resp
        print(f"[HTTP][ERROR] /ask - {repr(e)}")
        err_txt = "Ocurrió un error al procesar tu solicitud."
        output = err_txt
        save_conversation_to_blob(sid, question, output, {"error": str(e)})
        resp = make_response(jsonify({"error": str(e)}), 500)
        if new_sid:
            resp.set_cookie("SID", sid, httponly=True, samesite="Lax", max_age=60*60*24*7)
        return resp

@app.route("/followup-reply", methods=["POST"])
def followup_reply():
    """
    Recibe: { "reply": "Sí" | "No", "feedback": "texto opcional" }
    - Si reply == "No":
        * Devuelve 200 con final_message para que el frontend lo muestre en el chat.
        * Guarda el final_message en los logs de conversación.
        * Guarda el feedback en carpeta separada (feedback/...).
        * Cierra la sesión (borra cookie SID).
    - Si reply == "Sí": responde 200 {"ok": true}.
    """
    sid, _ = get_or_create_sid()
    data = request.get_json(force=True, silent=True) or {}
    reply = (data.get("reply") or "").strip().lower()
    feedback = (data.get("feedback") or "").strip()

    if reply in ("no", "n", "nope", "nah", "nop"):
        # 1) Guardar el mensaje final en logs de conversación
        save_system_message_to_blob(sid, FINAL_CHAT_MSG, {"farewell": True, "type": "final_message"})

        # 2) Guardar feedback en carpeta separada
        if feedback:
            save_feedback_to_blob(sid, feedback, {"source": "followup"})

        # 3) Responder mensaje final y cerrar la sesión (cookie expirada)
        resp = make_response(jsonify({
            "final_message": FINAL_CHAT_MSG,
            "conversation_closed": True
        }), 200)
        resp.set_cookie("SID", "", expires=0, samesite="Lax")
        return resp

    # Cualquier otra cosa (incluye "sí"): OK genérico
    return jsonify({"ok": True})

@app.route("/reindex", methods=["POST"])
def reindex():
    print("[HTTP] POST /reindex - inicio")
    if is_block_active():
        print("[HTTP] /reindex - circuit breaker activo, devolviendo 503")
        return jsonify({
            "error": "service_temporarily_throttled",
            "message": "El servicio está regulando temporalmente el tráfico. Intenta de nuevo más tarde."
        }), 503

    token = request.headers.get("X-APP-SECRET")
    if not token or token != os.getenv("APP_SECRET"):
        print("[HTTP] /reindex - unauthorized")
        return jsonify({"error": "unauthorized"}), 401

    force = request.args.get("force", "false").lower() == "true"
    try:
        X, chunks, sources = build_or_update_index(force=force)
        print(f"[HTTP] /reindex - completado. chunks_indexados={len(chunks)}")
        return jsonify({"indexed_chunks": len(chunks)})
    except Exception as e:
        print(f"[HTTP][ERROR] /reindex - {repr(e)}")
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    print("[HTTP] GET /health")
    return {"ok": True, "block_active": is_block_active()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"[STARTUP] Iniciando servidor en 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
