"use strict";

const SESSION_LIMIT = 5;
const ENDPOINT = "/api/v1/chat";

const chat = document.getElementById("chat");
const counter = document.getElementById("counter");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

// ---- Session state ---------------------------------------------------------

function getSessionId() {
  let sid = sessionStorage.getItem("sid");
  if (!sid) {
    sid =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : "sid-" + Date.now() + "-" + Math.random().toString(36).slice(2);
    sessionStorage.setItem("sid", sid);
  }
  return sid;
}

function getCount() {
  const raw = sessionStorage.getItem("count");
  const n = raw ? parseInt(raw, 10) : 0;
  return Number.isFinite(n) && n >= 0 ? n : 0;
}

function setCount(n) {
  sessionStorage.setItem("count", String(n));
}

function remaining() {
  return Math.max(0, SESSION_LIMIT - getCount());
}

function refreshCounter() {
  const r = remaining();
  counter.textContent = r + " quer" + (r === 1 ? "y" : "ies") + " remaining";
  counter.classList.toggle("empty", r === 0);
  const locked = r <= 0;
  input.disabled = locked;
  sendBtn.disabled = locked;
  if (locked) {
    input.placeholder = "Session limit reached. Close the tab to reset.";
  }
}

// ---- Rendering ------------------------------------------------------------

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderMarkdownLite(text) {
  // Strip residual HTML tags before escaping so they never show as &lt;tag&gt; text.
  const clean = text.replace(/<[^>]*>/g, "");
  const escaped = escapeHtml(clean);
  return escaped
    .replace(/\*(.+?)\*/g, "<strong>$1</strong>")
    .replace(/_(.+?)_/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

function addBubble(kind, text, opts) {
  const el = document.createElement("div");
  el.className = "bubble " + kind + (opts && opts.typing ? " typing" : "");
  if (opts && opts.html) {
    el.innerHTML = renderMarkdownLite(text);
  } else {
    el.textContent = text;
  }
  if (opts && opts.route) {
    const tag = document.createElement("span");
    tag.className = "route-tag";
    tag.textContent = opts.route;
    el.appendChild(tag);
  }
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

// ---- Send handler ---------------------------------------------------------

async function send(message) {
  addBubble("user", message);
  const typing = addBubble("bot", "…", { typing: true });

  try {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        session_id: getSessionId(),
      }),
    });

    if (!res.ok) {
      typing.remove();
      addBubble(
        "bot",
        "Sorry, the server returned " + res.status + ". Please try again."
      );
      return;
    }

    const data = await res.json();
    typing.remove();
    addBubble("bot", data.reply || "(empty reply)", { html: true, route: data.route });
  } catch (err) {
    typing.remove();
    addBubble("bot", "Network error: " + (err.message || err));
  }
}

// ---- Event wiring ---------------------------------------------------------

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  if (remaining() <= 0) return;

  input.value = "";
  input.style.height = "auto";

  setCount(getCount() + 1);
  refreshCounter();

  send(text);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
});

// ---- Chips ----------------------------------------------------------------

document.getElementById("chips").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  if (remaining() <= 0) return;
  const query = chip.dataset.query;
  input.value = "";
  setCount(getCount() + 1);
  refreshCounter();
  send(query);
});

// ---- Init -----------------------------------------------------------------

getSessionId();
refreshCounter();
