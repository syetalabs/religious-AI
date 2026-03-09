import { useState, useRef, useEffect } from "react";
import { RELIGIONS, WatermarkSVG } from "./Landingpage";

const API_BASE = import.meta.env.VITE_API_URL || "https://religious-ai.onrender.com";
// Christianity is English-only for now
const LANGUAGES = ["English"];

// ─── Avatars ──────────────────────────────────────────────────
const UserAvatar = ({ name = "User", size = 32, palette }) => (
  <div style={{
    width: size, height: size, borderRadius: "50%",
    background: `linear-gradient(135deg, ${palette.accentColor}, ${palette.accentDark})`,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: size * 0.38, color: "#fff", fontWeight: "700",
    fontFamily: "'Cinzel', serif", flexShrink: 0,
  }}>
    {name[0].toUpperCase()}
  </div>
);

const BotAvatar = ({ size = 32, emoji }) => (
  <div style={{
    width: size, height: size, borderRadius: "50%",
    background: "linear-gradient(135deg, #e8e8e8, #c0c0c0)",
    display: "flex", alignItems: "center", justifyContent: "center",
    flexShrink: 0, fontSize: size * 0.55,
  }}>
    {emoji}
  </div>
);

// ─── UI helpers ───────────────────────────────────────────────
const SourcePills = ({ sources, palette }) => {
  if (!sources?.length) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
      {sources.map((src, i) => (
        <span key={i} style={{
          fontSize: 10.5, padding: "3px 10px", borderRadius: 20,
          background: palette.hover, color: palette.accentDark,
          border: `1px solid ${palette.border}`,
          fontFamily: "'Cinzel', serif", letterSpacing: 0.5,
        }}>📖 {src}</span>
      ))}
    </div>
  );
};

const ConfidenceWarning = () => (
  <div style={{
    marginTop: 8, fontSize: 11, color: "#a0622a",
    background: "#fff3e0", border: "1px solid #f5c78e",
    borderRadius: 8, padding: "5px 10px", display: "flex", alignItems: "center", gap: 5,
  }}>
    ⚠️ Low confidence — please verify this with a religious scholar.
  </div>
);

const ErrorToast = ({ message, onClose }) => (
  <div style={{
    position: "fixed", bottom: 90, left: "50%", transform: "translateX(-50%)",
    background: "#3d0f0f", color: "#ffd5d5", padding: "12px 20px",
    borderRadius: 12, fontSize: 13, zIndex: 999,
    boxShadow: "0 4px 20px #0004", display: "flex", alignItems: "center", gap: 10,
  }}>
    ❌ {message}
    <button onClick={onClose} style={{ background: "none", border: "none", color: "#ffd5d5", cursor: "pointer", fontSize: 16 }}>×</button>
  </div>
);

// ─── Main Chatbot Component ───────────────────────────────────
export default function Chatbot({ religion, onSwitchReligion }) {
  const cfg = RELIGIONS[religion] || RELIGIONS.Buddhism;

  // Auto-close sidebar on mobile (viewport < 640px)
  const isMobile = () => typeof window !== "undefined" && window.innerWidth < 640;
  const [sidebarOpen, setSidebarOpen] = useState(() => !isMobile());
  const [messages, setMessages]                 = useState([]);
  const [input, setInput]                       = useState("");
  const [language, setLanguage]                 = useState("English");
  const [showLangDropdown, setShowLangDropdown] = useState(false);
  const [activeChat, setActiveChat]             = useState(null);
  const [isTyping, setIsTyping]                 = useState(false);
  const [error, setError]                       = useState(null);
  const [isConnected, setIsConnected]           = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);

  // Language options: Buddhism has Sinhala/Tamil; Christianity English only
  const availableLanguages = religion === "Buddhism" ? ["English", "Sinhala", "Tamil"] : LANGUAGES;

  // Ping /health on mount so the connection indicator updates immediately
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => setIsConnected(d.status === "ready"))
      .catch(() => setIsConnected(false));
  }, []);

  // Close sidebar when window resizes to mobile
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 640) setSidebarOpen(false);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;
    const userText = input.trim();
    setMessages(prev => [...prev, { id: Date.now(), role: "user", text: userText }]);
    setInput("");
    setIsTyping(true);
    setError(null);

    try {
      const langCode = language === "Sinhala" ? "si" : language === "Tamil" ? "ta" : "en";
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userText, religion, language: langCode }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setMessages(prev => [...prev, {
        id:      Date.now() + 1,
        role:    "bot",
        text:    data.answer,
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

  const handleKeyDown = e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleNewChat = () => { setMessages([]); setActiveChat(null); setError(null); };

  const connColor = isConnected === null ? "#aaa" : isConnected ? "#4caf50" : "#e53935";
  const connLabel = isConnected === null ? "Checking…" : isConnected ? "Connected" : "Offline";

  const inputPlaceholder =
    language === "Sinhala" ? "ඔබේ පණිවිඩය මෙහි ටයිප් කරන්න..." :
    language === "Tamil"   ? "உங்கள் செய்தியை இங்கே தட்டச்சு செய்யுங்கள்..." :
    cfg.placeholder;

  return (
    <div style={{
      display: "flex", height: "100dvh", width: "100vw",
      fontFamily: "'Lora', Georgia, serif",
      background: cfg.bgColor, color: cfg.text, overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; overflow: hidden; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${cfg.accentColor}55; border-radius: 4px; }
        .chat-input:focus { outline: none; }
        .sidebar-btn:hover { background: ${cfg.hover} !important; }
        .recent-item:hover { background: ${cfg.hover} !important; }
        .lang-opt:hover    { background: ${cfg.hover} !important; }
        .icon-btn:hover    { background: ${cfg.hover}88 !important; }
        @keyframes fadeUp  { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse   { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }
        @keyframes connPulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        .msg-anim { animation: fadeUp 0.35s ease forwards; }
        .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: ${cfg.accentColor}; animation: pulse 1.2s infinite; margin: 0 2px; }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .send-btn:not(:disabled):hover { opacity: 0.85; }
      `}</style>

      {/* Sidebar — overlays on mobile, pushes content on desktop */}
      {sidebarOpen && isMobile() && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{
            position: "fixed", inset: 0, background: "#0005",
            zIndex: 20, backdropFilter: "blur(1px)",
          }}
        />
      )}
      <div style={{
        width: sidebarOpen ? 220 : 0, minWidth: sidebarOpen ? 220 : 0,
        background: cfg.sidebarBg, borderRight: `1px solid ${cfg.border}`,
        display: "flex", flexDirection: "column",
        transition: "width 0.3s ease, min-width 0.3s ease",
        overflow: "hidden", position: isMobile() ? "fixed" : "relative",
        top: isMobile() ? 0 : "auto", left: isMobile() ? 0 : "auto",
        height: isMobile() ? "100vh" : "auto",
        zIndex: isMobile() ? 30 : 10,
      }}>
        <div style={{ padding: "20px 14px 80px", opacity: sidebarOpen ? 1 : 0, transition: "opacity 0.2s", height: "100%" }}>
          <div style={{ textAlign: "center", marginBottom: 20 }}>
            <div style={{ fontSize: 28, marginBottom: 4 }}>{cfg.emoji}</div>
            <div style={{ fontSize: 11, fontFamily: "'Cinzel', serif", letterSpacing: 1, color: cfg.textMuted }}>{cfg.label}</div>
          </div>

          <button className="sidebar-btn" onClick={handleNewChat} style={{
            width: "100%", padding: "10px 14px", borderRadius: 10,
            border: `1.5px solid ${cfg.accentColor}`, background: "transparent",
            color: cfg.accentDark, fontFamily: "'Lora', serif", fontSize: 13,
            cursor: "pointer", display: "flex", alignItems: "center", gap: 8,
            marginBottom: 8, transition: "background 0.2s",
          }}>
            <span style={{ fontSize: 16 }}>✦</span> New Chat
          </button>

          <button className="sidebar-btn" onClick={onSwitchReligion} style={{
            width: "100%", padding: "10px 14px", borderRadius: 10,
            border: `1.5px solid ${cfg.border}`, background: "transparent",
            color: cfg.textMuted, fontFamily: "'Lora', serif", fontSize: 13,
            cursor: "pointer", display: "flex", alignItems: "center", gap: 8,
            marginBottom: 16, transition: "background 0.2s",
          }}>
            <span style={{ fontSize: 14 }}>⇄</span> Switch Religion
          </button>
        </div>

        {/* Sidebar footer */}
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          padding: "12px 14px", borderTop: `1px solid ${cfg.border}`,
          background: cfg.sidebarBg, display: "flex", alignItems: "center", gap: 10,
          opacity: sidebarOpen ? 1 : 0, transition: "opacity 0.2s",
        }}>
          <UserAvatar name="User" size={34} palette={cfg} />
          <div style={{ flex: 1, overflow: "hidden" }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: cfg.text }}>Seeker</div>
            <div style={{ fontSize: 10, color: connColor, display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{
                display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: connColor,
                animation: isConnected === null ? "connPulse 1s infinite" : "none",
              }} />
              {connLabel}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, height: "100dvh", overflow: "hidden" }}>

        {/* Header */}
        <div style={{
          height: 58, background: cfg.headerBg, borderBottom: `1px solid ${cfg.border}`,
          display: "flex", alignItems: "center", padding: "0 20px", gap: 12, flexShrink: 0,
        }}>
          <button onClick={() => setSidebarOpen(o => !o)} className="icon-btn" style={{
            width: 36, height: 36, border: `1px solid ${cfg.border}`,
            borderRadius: 8, background: "transparent", cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: cfg.textMuted, fontSize: 16, transition: "background 0.15s",
          }}>
            {sidebarOpen ? "◁" : "▷"}
          </button>

          <BotAvatar size={34} emoji={cfg.botEmoji} />

          <div style={{ flex: 1 }} />

          {/* Language selector */}
          <div style={{ position: "relative" }}>
            <button onClick={() => setShowLangDropdown(d => !d)} className="icon-btn" style={{
              padding: "7px 14px", border: `1.5px solid ${cfg.accentColor}`,
              borderRadius: 20, background: "transparent",
              fontFamily: "'Lora', serif", fontSize: 12.5,
              color: cfg.accentDark, cursor: "pointer",
              display: "flex", alignItems: "center", gap: 6, transition: "background 0.15s",
            }}>
              🌐 {language} <span style={{ fontSize: 10 }}>▾</span>
            </button>
            {showLangDropdown && (
              <div style={{
                position: "absolute", top: "calc(100% + 6px)", right: 0,
                background: cfg.headerBg, border: `1px solid ${cfg.border}`,
                borderRadius: 12, boxShadow: "0 8px 30px #0002",
                zIndex: 100, minWidth: 140, overflow: "hidden",
              }}>
                {availableLanguages.map(lang => (
                  <div key={lang} className="lang-opt"
                    onClick={() => { setLanguage(lang); setShowLangDropdown(false); }}
                    style={{
                      padding: "9px 16px", fontSize: 13, cursor: "pointer",
                      color: lang === language ? cfg.accentDark : cfg.text,
                      fontWeight: lang === language ? 600 : 400,
                      background: lang === language ? cfg.hover : "transparent",
                      transition: "background 0.15s",
                    }}>
                    {lang}
                  </div>
                ))}
              </div>
            )}
          </div>

          <button className="icon-btn" onClick={onSwitchReligion} style={{
            padding: "7px 16px", border: `1.5px solid ${cfg.border}`,
            borderRadius: 20, background: "transparent",
            fontFamily: "'Lora', serif", fontSize: 12.5,
            color: cfg.textMuted, cursor: "pointer",
          }}>
            ⇄ Religion
          </button>
        </div>

        {/* Chat Area */}
        <div style={{ flex: 1, overflowY: "auto", padding: "28px clamp(12px, 5%, 10%)", position: "relative" }}>
          <div style={{
            position: "absolute", top: "50%", left: "50%",
            transform: "translate(-50%, -50%)",
            width: 320, height: 320, pointerEvents: "none", opacity: 0.18, zIndex: 0,
          }}>
            <WatermarkSVG type={cfg.watermark} color={cfg.accentColor} />
          </div>

          {messages.length === 0 && !isTyping && (
            <div style={{
              position: "absolute", top: "50%", left: "50%",
              transform: "translate(-50%, -50%)",
              textAlign: "center", color: cfg.textMuted, zIndex: 1, pointerEvents: "none",
            }}>
              <div style={{ fontSize: 38, marginBottom: 12 }}>{cfg.botEmoji}</div>
              <div style={{ fontFamily: "'Cinzel', serif", fontSize: 14, letterSpacing: 1 }}>{cfg.placeholder}</div>
            </div>
          )}

          <div style={{ position: "relative", zIndex: 1, display: "flex", flexDirection: "column", gap: 18 }}>
            {messages.map(msg => (
              <div key={msg.id} className="msg-anim" style={{
                display: "flex",
                flexDirection: msg.role === "user" ? "row-reverse" : "row",
                alignItems: "flex-end", gap: 10,
              }}>
                {msg.role === "bot"
                  ? <BotAvatar size={30} emoji={cfg.botEmoji} />
                  : <UserAvatar name="U" size={30} palette={cfg} />}
                <div style={{ maxWidth: "min(58%, 480px)", display: "flex", flexDirection: "column" }}>
                  <div style={{
                    background: msg.role === "user"
                      ? `linear-gradient(135deg, ${cfg.accentColor}, ${cfg.accentDark})`
                      : cfg.botBubble,
                    color:        msg.role === "user" ? "#fff" : cfg.text,
                    borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
                    padding: "12px 16px", fontSize: 13.5, lineHeight: 1.65,
                    boxShadow:   msg.role === "user" ? `0 4px 16px ${cfg.cardGlow}` : "0 2px 12px #0000000d",
                    border:      msg.role === "bot" ? `1px solid ${cfg.border}` : "none",
                  }}>
                    {msg.text}
                  </div>
                  {msg.role === "bot" && <SourcePills sources={msg.sources} palette={cfg} />}
                  {msg.role === "bot" && msg.warning && <ConfidenceWarning />}
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="msg-anim" style={{ display: "flex", alignItems: "flex-end", gap: 10 }}>
                <BotAvatar size={30} emoji={cfg.botEmoji} />
                <div style={{
                  background: cfg.botBubble, borderRadius: "18px 18px 18px 4px",
                  padding: "14px 18px", border: `1px solid ${cfg.border}`,
                  boxShadow: "0 2px 12px #0000000d",
                }}>
                  <span className="dot" /><span className="dot" /><span className="dot" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div style={{ padding: "14px clamp(12px, 5%, 10%)", background: cfg.headerBg, borderTop: `1px solid ${cfg.border}`, flexShrink: 0 }}>
          {isConnected === false && (
            <div style={{
              background: "#fff3e0", border: "1px solid #f5c78e", borderRadius: 10,
              padding: "8px 14px", marginBottom: 10, fontSize: 12,
              color: "#a0622a", display: "flex", alignItems: "center", gap: 8,
            }}>
              ⚠️ Backend offline — check{" "}
              <a href={`${API_BASE}/health`} target="_blank" rel="noreferrer" style={{ color: "#8b6914" }}>health status</a>
            </div>
          )}
          <div style={{
            display: "flex", alignItems: "center", background: cfg.inputBg,
            border: `1.5px solid ${isConnected === false ? "#f5c78e" : cfg.border}`,
            borderRadius: 16, padding: "10px 16px", gap: 10,
            boxShadow: "0 2px 8px #0000000a", transition: "border-color 0.2s",
          }}>
            <textarea
              ref={inputRef}
              className="chat-input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={inputPlaceholder}
              rows={1}
              disabled={isTyping}
              style={{
                flex: 1, background: "transparent", border: "none",
                fontFamily: "'Lora', serif", fontSize: 13.5, color: cfg.text,
                resize: "none", lineHeight: 1.5, maxHeight: 100, overflowY: "auto",
                opacity: isTyping ? 0.6 : 1,
              }}
            />
            <button className="send-btn" onClick={handleSend} disabled={!input.trim() || isTyping} style={{
              width: 36, height: 36,
              background: `linear-gradient(135deg, ${cfg.accentColor}, ${cfg.accentDark})`,
              border: "none", borderRadius: 10, cursor: "pointer", color: "#fff", fontSize: 16,
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: `0 3px 12px ${cfg.cardGlow}`, transition: "opacity 0.15s",
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