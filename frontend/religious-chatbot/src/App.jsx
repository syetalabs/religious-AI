import { useState, useRef, useEffect } from "react";

const LANGUAGES = ["English", "Sinhala", "Tamil"];

const RECENT_CHATS = [
  { id: 1, title: "Understanding Karma", time: "2h ago" },
];

// ─── Config ───────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL || "https://religious-ai.onrender.com";

// ─── SVG Watermark ────────────────────────────────────────
const DharmaWheelSVG = () => (
  <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ width: "100%", height: "100%" }}>
    <circle cx="100" cy="100" r="90" stroke="#c9a96e" strokeWidth="6" fill="none" opacity="0.35" />
    <circle cx="100" cy="100" r="18" stroke="#c9a96e" strokeWidth="5" fill="none" opacity="0.45" />
    <circle cx="100" cy="100" r="6" fill="#c9a96e" opacity="0.4" />
    {[0,30,60,90,120,150,180,210,240,270,300,330].map((angle, i) => {
      const rad = (angle * Math.PI) / 180;
      const x1 = 100 + 18 * Math.cos(rad);
      const y1 = 100 + 18 * Math.sin(rad);
      const x2 = 100 + 90 * Math.cos(rad);
      const y2 = 100 + 90 * Math.sin(rad);
      return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#c9a96e" strokeWidth="2.5" opacity="0.3" />;
    })}
    {[0,40,80,120,160,200,240,280,320].map((angle, i) => {
      const rad = (angle * Math.PI) / 180;
      const cx = 100 + 55 * Math.cos(rad);
      const cy = 100 + 55 * Math.sin(rad);
      return (
        <g key={i} transform={`rotate(${angle}, ${cx}, ${cy})`}>
          <ellipse cx={cx} cy={cy} rx="10" ry="5" stroke="#c9a96e" strokeWidth="2" fill="none" opacity="0.3" />
        </g>
      );
    })}
  </svg>
);

// ─── Avatars ──────────────────────────────────────────────
const UserAvatar = ({ name = "User", size = 32 }) => (
  <div style={{
    width: size, height: size, borderRadius: "50%",
    background: "linear-gradient(135deg, #c9a96e, #8b6914)",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: size * 0.38, color: "#fff", fontWeight: "700",
    fontFamily: "'Cinzel', serif", flexShrink: 0
  }}>
    {name[0].toUpperCase()}
  </div>
);

const BotAvatar = ({ size = 32 }) => (
  <div style={{
    width: size, height: size, borderRadius: "50%",
    background: "linear-gradient(135deg, #e8d5b0, #c9a96e)",
    display: "flex", alignItems: "center", justifyContent: "center",
    flexShrink: 0, fontSize: size * 0.55
  }}>
    🪷
  </div>
);

// ─── Source Pills ─────────────────────────────────────────
const SourcePills = ({ sources, palette }) => {
  if (!sources || sources.length === 0) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
      {sources.map((src, i) => (
        <span key={i} style={{
          fontSize: 10.5, padding: "3px 10px", borderRadius: 20,
          background: palette.hover, color: palette.accentDark,
          border: `1px solid ${palette.sidebarBorder}`,
          fontFamily: "'Cinzel', serif", letterSpacing: 0.5
        }}>
          📖 {src}
        </span>
      ))}
    </div>
  );
};

// ─── Confidence Warning ───────────────────────────────────
const ConfidenceWarning = ({ palette }) => (
  <div style={{
    marginTop: 8, fontSize: 11, color: "#a0622a",
    background: "#fff3e0", border: "1px solid #f5c78e",
    borderRadius: 8, padding: "5px 10px", display: "flex", alignItems: "center", gap: 5
  }}>
    ⚠️ Low confidence — please verify this with a religious scholar.
  </div>
);

// ─── English Translation Box ──────────────────────────────
const EnglishTranslation = ({ text, palette }) => {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{
      marginTop: 8, borderRadius: 8,
      border: `1px dashed ${palette.sidebarBorder}`,
      overflow: "hidden"
    }}>
      <button
        onClick={() => setExpanded(e => !e)}
        style={{
          width: "100%", padding: "5px 10px",
          background: palette.hover, border: "none",
          cursor: "pointer", display: "flex", alignItems: "center",
          justifyContent: "space-between",
          fontFamily: "'Cinzel', serif", fontSize: 10.5,
          color: palette.accentDark, letterSpacing: 0.5
        }}>
        <span> English translation</span>
        <span style={{ fontSize: 10 }}>{expanded ? "▲" : "▼"}</span>
      </button>
      {expanded && (
        <div style={{
          padding: "8px 12px", fontSize: 12.5,
          color: palette.textMuted, lineHeight: 1.6,
          background: palette.inputBg,
          fontStyle: "italic"
        }}>
          {text}
        </div>
      )}
    </div>
  );
};

const ErrorToast = ({ message, onClose }) => (
  <div style={{
    position: "fixed", bottom: 90, left: "50%", transform: "translateX(-50%)",
    background: "#3d0f0f", color: "#ffd5d5", padding: "12px 20px",
    borderRadius: 12, fontSize: 13, zIndex: 999,
    boxShadow: "0 4px 20px #0004", display: "flex", alignItems: "center", gap: 10
  }}>
    ❌ {message}
    <button onClick={onClose} style={{ background: "none", border: "none", color: "#ffd5d5", cursor: "pointer", fontSize: 16 }}>×</button>
  </div>
);

// ─── Main Component ───────────────────────────────────────
export default function ReligiousChatbot() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [language, setLanguage] = useState("English");
  const [showLangDropdown, setShowLangDropdown] = useState(false);
  const [activeChat, setActiveChat] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Check backend health on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => setIsConnected(data.status === "ready"))
      .catch(() => setIsConnected(false));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // ── Send message to FastAPI ──────────────────────────────
  const handleSend = async () => {
    if (!input.trim() || isTyping) return;

    const userText = input.trim();
    setMessages(prev => [...prev, { id: Date.now(), role: "user", text: userText }]);
    setInput("");
    setIsTyping(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userText, language }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();

      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: "bot",
        text: data.answer,                   // translated answer
        textEnglish: data.is_english         // if English selected, no secondary box needed
          ? null
          : data.answer_english,
        sources: data.sources,
        warning: data.confidence_warning,
      }]);

      setIsConnected(true);

    } catch (err) {
      setError(err.message || "Could not reach the server.");
      setIsConnected(false);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleNewChat = () => {
    setMessages([]);
    setActiveChat(null);
    setError(null);
  };

  const palette = {
    bg: "#f5edd8",
    sidebar: "#ede0c4",
    sidebarBorder: "#d4bc94",
    header: "#faf4e8",
    accent: "#c9a96e",
    accentDark: "#8b6914",
    text: "#3d2e0f",
    textMuted: "#7a6040",
    botBubble: "#fff8ee",
    inputBg: "#faf4e8",
    hover: "#e8d5b0",
  };

  const connColor = isConnected === null ? "#aaa" : isConnected ? "#4caf50" : "#e53935";
  const connLabel = isConnected === null ? "Checking..." : isConnected ? "Connected" : "Offline";

  return (
    <div style={{
      display: "flex", height: "100vh", width: "100vw",
      fontFamily: "'Lora', Georgia, serif",
      background: palette.bg, color: palette.text,
      overflow: "hidden"
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #c9a96e55; border-radius: 4px; }
        .chat-input:focus { outline: none; }
        .sidebar-btn:hover { background: #e8d5b0 !important; }
        .recent-item:hover { background: #e8d5b0 !important; }
        .lang-opt:hover { background: #e8d5b0 !important; }
        .icon-btn:hover { background: #e8d5b088 !important; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }
        @keyframes connPulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        .msg-anim { animation: fadeUp 0.35s ease forwards; }
        .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #c9a96e; animation: pulse 1.2s infinite; margin: 0 2px; }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .send-btn:not(:disabled):hover { opacity: 0.85; }
      `}</style>

      {/* Sidebar */}
      <div style={{
        width: sidebarOpen ? 220 : 0,
        minWidth: sidebarOpen ? 220 : 0,
        background: palette.sidebar,
        borderRight: `1px solid ${palette.sidebarBorder}`,
        display: "flex", flexDirection: "column",
        transition: "width 0.3s ease, min-width 0.3s ease",
        overflow: "hidden",
        position: "relative", zIndex: 10
      }}>
        <div style={{ padding: "20px 14px 80px", opacity: sidebarOpen ? 1 : 0, transition: "opacity 0.2s", height: "100%" }}>
          <div style={{ textAlign: "center", marginBottom: 20 }}>
            <div style={{ fontSize: 28, marginBottom: 4 }}>☸️</div>
          </div>
          <button className="sidebar-btn" onClick={handleNewChat}
            style={{
            width: "100%", padding: "10px 14px", borderRadius: 10,
            border: `1.5px solid ${palette.accent}`, background: "transparent",
            color: palette.accentDark, fontFamily: "'Lora', serif", fontSize: 13,
            cursor: "pointer", display: "flex", alignItems: "center", gap: 8,
            marginBottom: 8, transition: "background 0.2s"
          }}>
            <span style={{ fontSize: 16 }}>✦</span> New Chat
          </button>
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: 10, letterSpacing: 2, textTransform: "uppercase", color: palette.textMuted, marginBottom: 8, paddingLeft: 4, fontFamily: "'Cinzel', serif" }}>
              Recent
            </div>
            {RECENT_CHATS.map(chat => (
              <div key={chat.id} className="recent-item"
                onClick={() => setActiveChat(chat.id)}
                style={{
                padding: "8px 10px", borderRadius: 8, cursor: "pointer",
                background: activeChat === chat.id ? palette.hover : "transparent",
                marginBottom: 2, transition: "background 0.15s",
                borderLeft: activeChat === chat.id ? `3px solid ${palette.accent}` : "3px solid transparent"
              }}>
                <div style={{ fontSize: 12.5, color: palette.text, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{chat.title}</div>
                <div style={{ fontSize: 10, color: palette.textMuted, marginTop: 2 }}>{chat.time}</div>
              </div>
            ))}
          </div>
        </div>
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          padding: "12px 14px",
          borderTop: `1px solid ${palette.sidebarBorder}`,
          background: palette.sidebar,
          display: "flex", alignItems: "center", gap: 10,
          opacity: sidebarOpen ? 1 : 0, transition: "opacity 0.2s"
        }}>
          <UserAvatar name="User" size={34} />
          <div style={{ flex: 1, overflow: "hidden" }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: palette.text }}>Seeker</div>
            <div style={{ fontSize: 10, color: connColor, display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{
                display: "inline-block", width: 6, height: 6, borderRadius: "50%",
                background: connColor,
                animation: isConnected === null ? "connPulse 1s infinite" : "none"
              }} />
              {connLabel}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {/* Header */}
        <div style={{
          height: 58, background: palette.header,
          borderBottom: `1px solid ${palette.sidebarBorder}`,
          display: "flex", alignItems: "center", padding: "0 20px",
          gap: 12, flexShrink: 0
        }}>
          <button onClick={() => setSidebarOpen(o => !o)}
            style={{
            width: 36, height: 36, border: `1px solid ${palette.sidebarBorder}`,
            borderRadius: 8, background: "transparent", cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: palette.textMuted, fontSize: 16, transition: "background 0.15s"
          }} className="icon-btn">
            {sidebarOpen ? "◁" : "▷"}
          </button>
          <BotAvatar size={34} />
          <div style={{ flex: 1 }} />
          <div style={{ position: "relative" }}>
            <button onClick={() => setShowLangDropdown(d => !d)}
              style={{
              padding: "7px 14px", border: `1.5px solid ${palette.accent}`,
              borderRadius: 20, background: "transparent",
              fontFamily: "'Lora', serif", fontSize: 12.5,
              color: palette.accentDark, cursor: "pointer",
                display: "flex", alignItems: "center", gap: 6,
                transition: "background 0.15s"
            }} className="icon-btn">
              🌐 {language} <span style={{ fontSize: 10 }}>▾</span>
            </button>
            {showLangDropdown && (
              <div style={{
                position: "absolute", top: "calc(100% + 6px)", right: 0,
                background: palette.header, border: `1px solid ${palette.sidebarBorder}`,
                borderRadius: 12, boxShadow: "0 8px 30px #0002",
                zIndex: 100, minWidth: 140, overflow: "hidden"
              }}>
                {LANGUAGES.map(lang => (
                  <div key={lang} className="lang-opt"
                    onClick={() => { setLanguage(lang); setShowLangDropdown(false); }}
                    style={{
                      padding: "9px 16px", fontSize: 13, cursor: "pointer",
                      color: lang === language ? palette.accentDark : palette.text,
                      fontWeight: lang === language ? 600 : 400,
                      background: lang === language ? palette.hover : "transparent",
                      transition: "background 0.15s"
                    }}>
                    {lang}
                  </div>
                ))}
              </div>
            )}
          </div>
          <button style={{
            padding: "7px 16px", border: `1.5px solid ${palette.sidebarBorder}`,
            borderRadius: 20, background: "transparent",
            fontFamily: "'Lora', serif", fontSize: 12.5,
            color: palette.textMuted, cursor: "pointer"
          }} className="icon-btn">Home</button>
        </div>

        {/* Chat Area */}
        <div style={{ flex: 1, overflowY: "auto", padding: "28px 10%", position: "relative" }}>
          <div style={{
            position: "absolute", top: "50%", left: "50%",
            transform: "translate(-50%, -50%)",
            width: 320, height: 320, pointerEvents: "none", opacity: 0.18, zIndex: 0
          }}>
            <DharmaWheelSVG />
          </div>
          {messages.length === 0 && !isTyping && (
            <div style={{
              position: "absolute", top: "50%", left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center", color: palette.textMuted, zIndex: 1,
              pointerEvents: "none"
            }}>
              <div style={{ fontSize: 38, marginBottom: 12 }}>🪷</div>
              <div style={{ fontFamily: "'Cinzel', serif", fontSize: 14, letterSpacing: 1 }}>Ask a question about Buddhism</div>
            </div>
          )}
          <div style={{ position: "relative", zIndex: 1, display: "flex", flexDirection: "column", gap: 18 }}>
            {messages.map((msg) => (
              <div key={msg.id} className="msg-anim"
                style={{
                display: "flex",
                flexDirection: msg.role === "user" ? "row-reverse" : "row",
                alignItems: "flex-end", gap: 10
              }}>
                {msg.role === "bot" ? <BotAvatar size={30} /> : <UserAvatar name="U" size={30} />}
                <div style={{ maxWidth: "58%", display: "flex", flexDirection: "column" }}>
                  <div style={{
                    background: msg.role === "user"
                      ? `linear-gradient(135deg, ${palette.accent}, ${palette.accentDark})`
                      : palette.botBubble,
                    color: msg.role === "user" ? "#fff" : palette.text,
                    borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
                    padding: "12px 16px",
                    fontSize: 13.5, lineHeight: 1.65,
                    boxShadow: msg.role === "user" ? "0 4px 16px #c9a96e44" : "0 2px 12px #0000000d",
                    border: msg.role === "bot" ? `1px solid ${palette.sidebarBorder}` : "none"
                  }}>
                    {msg.text}
                  </div>

                  {/* English translation box — only shown for Sinhala/Tamil */}
                  {msg.role === "bot" && msg.textEnglish && (
                    <EnglishTranslation text={msg.textEnglish} palette={palette} />
                  )}

                  {msg.role === "bot" && <SourcePills sources={msg.sources} palette={palette} />}
                  {msg.role === "bot" && msg.warning && <ConfidenceWarning palette={palette} />}
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="msg-anim" style={{ display: "flex", alignItems: "flex-end", gap: 10 }}>
                <BotAvatar size={30} />
                <div style={{
                  background: palette.botBubble, borderRadius: "18px 18px 18px 4px",
                  padding: "14px 18px", border: `1px solid ${palette.sidebarBorder}`,
                  boxShadow: "0 2px 12px #0000000d"
                }}>
                  <span className="dot" /><span className="dot" /><span className="dot" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div style={{
          padding: "14px 10%", background: palette.header,
          borderTop: `1px solid ${palette.sidebarBorder}`
        }}>
          {isConnected === false && (
            <div style={{
              background: "#fff3e0", border: "1px solid #f5c78e",
              borderRadius: 10, padding: "8px 14px", marginBottom: 10,
              fontSize: 12, color: "#a0622a", display: "flex", alignItems: "center", gap: 8
            }}>
              ⚠️ Backend offline — check <a href="https://religious-ai.onrender.com/health" target="_blank" rel="noreferrer" style={{ color: "#8b6914" }}>health status</a>
            </div>
          )}
          <div style={{
            display: "flex", alignItems: "center",
            background: palette.inputBg,
            border: `1.5px solid ${isConnected === false ? "#f5c78e" : palette.sidebarBorder}`,
            borderRadius: 16, padding: "10px 16px", gap: 10,
            boxShadow: "0 2px 8px #0000000a",
            transition: "border-color 0.2s"
          }}>
            <textarea
              ref={inputRef}
              className="chat-input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isConnected === false
                  ? "Server offline..."
                  : language === "Sinhala"
                  ? "ඔබේ පණිවිඩය මෙහි ටයිප් කරන්න..."
                  : language === "Tamil"
                  ? "உங்கள் செய்தியை இங்கே தட்டச்சு செய்யுங்கள்..."
                  : "Type a new message here"
              }
              rows={1}
              disabled={isTyping}
              style={{
                flex: 1, background: "transparent", border: "none",
                fontFamily: "'Lora', serif", fontSize: 13.5, color: palette.text,
                resize: "none", lineHeight: 1.5,
                maxHeight: 100, overflowY: "auto",
                opacity: isTyping ? 0.6 : 1
              }}
            />
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={!input.trim() || isTyping}
              style={{
                width: 36, height: 36,
                background: `linear-gradient(135deg, ${palette.accent}, ${palette.accentDark})`,
                border: "none", borderRadius: 10,
                cursor: "pointer", color: "#fff", fontSize: 16,
                display: "flex", alignItems: "center", justifyContent: "center",
                boxShadow: "0 3px 12px #c9a96e55",
                transition: "opacity 0.15s",
              }}>
              {isTyping ? "⏳" : "➤"}
            </button>
          </div>
        </div>
      </div>

      {error && <ErrorToast message={error} onClose={() => setError(null)} />}
      {showLangDropdown && (
        <div onClick={() => setShowLangDropdown(false)}
          style={{ position: "fixed", inset: 0, zIndex: 50 }} />
      )}
    </div>
  );
}