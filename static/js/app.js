// ====== Texto de bienvenida al abrir la página (solo frontend) ======
const GREETING = "Hola Soy el Asistente de Mapas de Carrera y estoy aqui para ayudarte en las consultas que tengas";

// ====== Referencias del DOM ======
const chatBox   = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const btnSend   = document.getElementById("btn-send");

const followupBox   = document.getElementById("followup");
const followupText  = document.getElementById("followup-text");
const feedbackInput = document.getElementById("feedback");
const btnYes = document.getElementById("btn-yes");
const btnNo  = document.getElementById("btn-no");

// ====== Estado ======
let followupTimer = null;
let busy = false;  // evita doble envío

// ====== Utilidades ======
function appendMessage(text, who = "bot") {
  const div = document.createElement("div");
  div.className = `message ${who}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function scheduleFollowup(delaySec = 15, prompt = "¿Tienes alguna otra consulta? Marca Sí o No.") {
  clearTimeout(followupTimer);
  followupTimer = setTimeout(() => {
    followupText.textContent = prompt;
    followupBox.style.display = "block";
  }, delaySec * 1000);
}

function disableConversation(reason = "Conversación cerrada. Refresca la página para iniciar otra.") {
  userInput.disabled = true;
  btnSend.disabled = true;
  userInput.placeholder = reason;
  clearTimeout(followupTimer);
  followupBox.style.display = "none";
}

// ====== Lógica principal ======
async function sendMessage() {
  if (busy) return;
  const message = (userInput.value || "").trim();
  if (!message) return;

  appendMessage(message, "user");
  userInput.value = "";
  busy = true;

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: message })
    });

    if (!response.ok) {
      const errText = await response.text();
      if (response.status === 429) {
        appendMessage("⚠️ Has alcanzado el límite de uso. Intenta de nuevo en unos segundos.", "bot");
      } else if (response.status === 503) {
        appendMessage("⏳ Servicio regulando temporalmente el tráfico. Intenta nuevamente en unos minutos.", "bot");
      } else {
        appendMessage(`❌ Error ${response.status}: ${errText || "Error desconocido"}`, "bot");
      }
      return;
    }

    const data = await response.json();
    appendMessage(data.answer || "", "bot");

    if (data.followup_suggested) {
      const delay = Number(data.followup_after_seconds ?? 15);
      const prompt = data.followup_prompt || "¿Tienes alguna otra consulta? Marca Sí o No.";
      scheduleFollowup(delay, prompt);
    } else {
      clearTimeout(followupTimer);
      followupBox.style.display = "none";
    }
  } catch (err) {
    appendMessage(`❌ Error de red: ${String(err)}`, "bot");
  } finally {
    busy = false;
  }
}

// ====== Eventos ======
btnSend.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });

// Follow-up: Sí -> notifica y sigue
btnYes.addEventListener("click", async () => {
  followupBox.style.display = "none";
  feedbackInput.value = "";
  try {
    await fetch("/followup-reply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reply: "Sí" })
    });
  } catch (e) { console.warn(e); }
});

// Follow-up: No -> muestra mensaje final, guarda feedback y cierra la sesión/conversación
btnNo.addEventListener("click", async () => {
  followupBox.style.display = "none";
  const feedback = feedbackInput.value || "";
  feedbackInput.value = "";

  try {
    const res = await fetch("/followup-reply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reply: "No", feedback })
    });

    if (res.ok) {
      const data = await res.json(); // { final_message, conversation_closed }
      if (data.final_message) {
        appendMessage(data.final_message, "bot");
      }
      if (data.conversation_closed) {
        disableConversation(); // "ya no responde más" en esta sesión
      }
    } else {
      disableConversation("Sesión finalizada.");
    }
  } catch (e) {
    console.warn(e);
    disableConversation("Sesión finalizada.");
  }
});

// ====== Saludo al cargar la página ======
window.addEventListener("DOMContentLoaded", () => {
  appendMessage(GREETING, "bot");
  userInput?.focus();
});
