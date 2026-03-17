import { useState, useRef, useEffect } from "react";
import {
  FaDharmachakra, FaCross, FaOm, FaStarAndCrescent,
  FaGlobe, FaBookOpen, FaHeart, FaStar, FaCircleCheck, FaChevronLeft,
} from "react-icons/fa6";
import { MdError } from "react-icons/md";
import { IoArrowForward } from "react-icons/io5";

const API_BASE        = import.meta.env.VITE_API_URL || "https://religious-ai.onrender.com";
const POLL_INTERVAL_MS = 2500;

// Map symbol key → FA6 icon component
const SYMBOL_ICONS = {
  DharmaWheel: FaDharmachakra,
  Cross:       FaCross,
  Om:          FaOm,
  Crescent:    FaStarAndCrescent,
};

export const RELIGIONS = {
  Buddhism: {
    label:       "Buddhism",
    symbolKey:   "DharmaWheel",
    accentColor: "#c9a96e",
    accentDark:  "#8b6914",
    bgColor:     "#f5edd8",
    sidebarBg:   "#ede0c4",
    border:      "#d4bc94",
    headerBg:    "#faf4e8",
    botBubble:   "#fff8ee",
    inputBg:     "#faf4e8",
    hover:       "#e8d5b0",
    text:        "#3d2e0f",
    textMuted:   "#7a6040",
    gradient:    "linear-gradient(135deg, #f5edd8 0%, #e8d5b0 100%)",
    cardGlow:    "#c9a96e44",
    description: "Explore the teachings of the Buddha — the path to peace, wisdom, and liberation.",
    placeholder: "Ask about the Dhamma, meditation, or Buddhist practice…",
    loadingMsg:  "Preparing sacred texts from the Pali Canon…",
  },
  Christianity: {
    label:       "Christianity",
    symbolKey:   "Cross",
    accentColor: "#4a7fcb",
    accentDark:  "#1e4e8c",
    bgColor:     "#f0f4fb",
    sidebarBg:   "#dce6f5",
    border:      "#a8c0e8",
    headerBg:    "#f7f9fe",
    botBubble:   "#f0f6ff",
    inputBg:     "#f7f9fe",
    hover:       "#cfe0f5",
    text:        "#1a2d4a",
    textMuted:   "#4a6a9a",
    gradient:    "linear-gradient(135deg, #f0f4fb 0%, #dce6f5 100%)",
    cardGlow:    "#4a7fcb44",
    description: "Discover the Word of God — teachings of love, grace, faith, and salvation.",
    placeholder: "Ask about the Gospels, prayer, or Christian teaching…",
    loadingMsg:  "Preparing scripture from the Holy Bible…",
  },
  Hinduism: {
    label:       "Hinduism",
    symbolKey:   "Om",
    accentColor: "#e85d26",
    accentDark:  "#9c3210",
    bgColor:     "#fdf1e8",
    sidebarBg:   "#f5dfc8",
    border:      "#e8c49a",
    headerBg:    "#fef8f2",
    botBubble:   "#fff5ed",
    inputBg:     "#fef8f2",
    hover:       "#f5d9b8",
    text:        "#3d1a08",
    textMuted:   "#8a4a22",
    gradient:    "linear-gradient(135deg, #fdf1e8 0%, #f5dfc8 100%)",
    cardGlow:    "#e85d2644",
    description: "Explore the eternal wisdom of the Vedas, Upanishads, and Bhagavad Gita.",
    placeholder: "Ask about dharma, karma, yoga, or Hindu philosophy…",
    loadingMsg:  "Preparing sacred texts from the Vedic scriptures…",

  },
  Islam: {
    label:       "Islam",
    symbolKey:   "Crescent",
    accentColor: "#2d8a5e",
    accentDark:  "#1a5438",
    bgColor:     "#edf6f1",
    sidebarBg:   "#cfe8da",
    border:      "#9ecfb8",
    headerBg:    "#f5fbf8",
    botBubble:   "#edfaf3",
    inputBg:     "#f5fbf8",
    hover:       "#bde4cf",
    text:        "#0e2d1e",
    textMuted:   "#3a7058",
    gradient:    "linear-gradient(135deg, #edf6f1 0%, #cfe8da 100%)",
    cardGlow:    "#2d8a5e44",
    description: "Discover the teachings of the Quran — guidance, mercy, and the path of Islam.",
    placeholder: "Ask about the Quran, prayer, or Islamic teaching…",
    loadingMsg:  "Preparing scripture from the Holy Quran…",
    comingSoon:  true,
  },
};

// ─── Reusable symbol icon ─────────────────────────────────────
export const ReligionSymbol = ({ symbolKey, size = 48, color = "currentColor", opacity = 1 }) => {
  const Icon = SYMBOL_ICONS[symbolKey];
  if (!Icon) return null;
  return <Icon size={size} color={color} style={{ opacity, display: "block", flexShrink: 0 }} />;
};

// ─── Dev Banner ───────────────────────────────────────────────
const DevBanner = ({ onClose }) => (
  <div style={{
    position: "fixed", top: 0, left: 0, right: 0, zIndex: 9999,
    background: "linear-gradient(90deg, #7f0000 0%, #b71c1c 40%, #c62828 60%, #b71c1c 80%, #7f0000 100%)",
    borderBottom: "1px solid rgba(255,100,100,0.25)",
    padding: "9px 52px",
    display: "flex", alignItems: "center", justifyContent: "center",
    boxShadow: "0 2px 16px rgba(183,28,28,0.55)",
    fontFamily: "'Lora', Georgia, serif",
  }}>
    <span style={{
      fontSize: 12.5, color: "#ffcdd2", letterSpacing: 0.4,
      display: "flex", alignItems: "center", gap: 9,
    }}>
      <MdError size={15} color="#ff8a80" style={{ flexShrink: 0 }} />
      <strong style={{
        fontFamily: "'Cinzel', serif", letterSpacing: 1.5,
        color: "#fff", fontSize: 11.5, textTransform: "uppercase",
      }}>
        Under Development
      </strong>
      <span style={{ color: "rgba(255,205,210,0.5)", fontSize: 11 }}>·</span>
      This platform is still being built.
    </span>
    <button
      onClick={onClose}
      aria-label="Dismiss banner"
      style={{
        position: "absolute", right: 14,
        background: "none", border: "none", cursor: "pointer",
        color: "#ffcdd2", fontSize: 16, lineHeight: 1,
        padding: "4px 8px", borderRadius: 6,
        transition: "background 0.15s",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}
      onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.18)"; }}
      onMouseLeave={e => { e.currentTarget.style.background = "none"; }}
    >
      ✕
    </button>
  </div>
);

// ─── Religion Selection Page ──────────────────────────────────
const ReligionSelectPage = ({ onSelect, bannerVisible }) => {
  const [hovered, setHovered] = useState(null);

  const features = [
    { Icon: FaGlobe,    label: "Multi-language support" },
    { Icon: FaBookOpen, label: "More scriptures" },
    { Icon: FaHeart,    label: "Free for all" },
  ];

  return (
    <div style={{
      minHeight: "100vh", width: "100vw",
      background: "linear-gradient(160deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "flex-start",
      fontFamily: "'Lora', Georgia, serif",
      padding: `${bannerVisible ? "98px" : "60px"} 24px 24px`,
      overflowX: "hidden",
      transition: "padding-top 0.2s ease",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .rel-card { transition: transform 0.25s ease, box-shadow 0.25s ease; }
        .rel-card:hover { transform: translateY(-6px); }
      `}</style>

      <div style={{ textAlign: "center", marginBottom: 56, animation: "fadeIn 0.6s ease" }}>
        <div style={{ fontSize: 14, letterSpacing: 5, textTransform: "uppercase", color: "#8899bb", fontFamily: "'Cinzel', serif", marginBottom: 16 }}>
          Welcome to
        </div>
        <h1 style={{
          fontSize: "clamp(28px, 5vw, 46px)", fontFamily: "'Cinzel', serif",
          fontWeight: 700, color: "#e8eaf6", letterSpacing: 2, marginBottom: 14,
          textShadow: "0 2px 20px #0008",
        }}>
          Multi-Religious AI Chatbot
        </h1>
        <p style={{ fontSize: 15, color: "#8899bb", maxWidth: 440, margin: "0 auto", lineHeight: 1.7 }}>
          Choose a spiritual tradition to explore. Your guide will answer questions grounded in authentic scripture.
        </p>
      </div>

      <div style={{ display: "flex", gap: 28, flexWrap: "wrap", justifyContent: "center", animation: "fadeIn 0.8s ease" }}>
        {Object.entries(RELIGIONS).map(([key, cfg]) => (
          <div key={key} className="rel-card"
            onMouseEnter={() => !cfg.comingSoon && setHovered(key)}
            onMouseLeave={() => setHovered(null)}
            onClick={() => !cfg.comingSoon && onSelect(key)}
            style={{
              width: 260, borderRadius: 20,
              background: hovered === key
                ? `linear-gradient(145deg, ${cfg.bgColor}, ${cfg.hover})`
                : "rgba(255,255,255,0.06)",
              border: `2px solid ${hovered === key ? cfg.accentColor : cfg.comingSoon ? "rgba(255,255,255,0.07)" : "rgba(255,255,255,0.12)"}`,
              padding: "36px 28px 28px",
              display: "flex", flexDirection: "column", alignItems: "center", gap: 14,
              boxShadow: hovered === key
                ? `0 20px 50px ${cfg.cardGlow}, 0 0 0 1px ${cfg.accentColor}33`
                : "0 8px 32px rgba(0,0,0,0.3)",
              backdropFilter: "blur(12px)",
              opacity: cfg.comingSoon ? 0.55 : 1,
              cursor: cfg.comingSoon ? "default" : "pointer",
              position: "relative", overflow: "hidden",
            }}
          >
            {cfg.comingSoon && (
              <div style={{
                position: "absolute", top: 14, right: -24,
                background: "linear-gradient(135deg, #667eea, #764ba2)",
                color: "#fff", fontSize: 9.5, fontFamily: "'Cinzel', serif",
                letterSpacing: 1.5, padding: "4px 32px",
                transform: "rotate(35deg)", transformOrigin: "center",
                boxShadow: "0 2px 8px #0004",
              }}>
                SOON
              </div>
            )}

            <ReligionSymbol
              symbolKey={cfg.symbolKey}
              size={56}
              color={hovered === key ? cfg.accentDark : cfg.accentColor}
              opacity={hovered === key ? 1 : 0.65}
            />

            <div style={{
              fontFamily: "'Cinzel', serif", fontSize: 20, fontWeight: 600,
              letterSpacing: 1, color: hovered === key ? cfg.text : "#e8eaf6",
              textAlign: "center",
            }}>
              {cfg.label}
            </div>
            <p style={{
              fontSize: 12.5, lineHeight: 1.65, textAlign: "center",
              color: hovered === key ? cfg.textMuted : "#8899bb", minHeight: 52,
            }}>
              {cfg.description}
            </p>
            <div style={{
              marginTop: 6, padding: "9px 24px", borderRadius: 30,
              background: cfg.comingSoon
                ? "rgba(255,255,255,0.06)"
                : hovered === key
                ? `linear-gradient(135deg, ${cfg.accentColor}, ${cfg.accentDark})`
                : "rgba(255,255,255,0.1)",
              color: cfg.comingSoon ? "#445566" : hovered === key ? "#fff" : "#aabbdd",
              fontSize: 12, fontFamily: "'Cinzel', serif", letterSpacing: 1,
              border: `1px solid ${hovered === key && !cfg.comingSoon ? "transparent" : "rgba(255,255,255,0.15)"}`,
              transition: "all 0.2s",
              display: "flex", alignItems: "center", gap: 6,
            }}>
              {cfg.comingSoon ? "Coming Soon" : <><span>Enter</span><IoArrowForward size={13} /></>}
            </div>
          </div>
        ))}
      </div>

      {/* ── Help Us Section ───────────────────────────────── */}
      <div style={{
        marginTop: 80, width: "100%", maxWidth: 700,
        borderTop: "1px solid rgba(255,255,255,0.08)",
        paddingTop: 60, paddingBottom: 40,
        display: "flex", flexDirection: "column", alignItems: "center", gap: 18,
        animation: "fadeIn 1s ease",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 22, letterSpacing: 3, color: "#667eea", fontFamily: "'Cinzel', serif", fontWeight: 600 }}>
          <FaStar size={14} color="#667eea" />
          Help Us Grow
          <FaStar size={14} color="#667eea" />
        </div>
        <h2 style={{
          fontSize: "clamp(20px, 3.5vw, 30px)", fontFamily: "'Cinzel', serif",
          fontWeight: 700, color: "#e8eaf6", letterSpacing: 1, textAlign: "center",
          textShadow: "0 2px 20px #0006",
        }}>
          Support Multi-Religious AI for Everyone
        </h2>
        <p style={{
          fontSize: 14.5, color: "#7788aa", maxWidth: 520, textAlign: "center",
          lineHeight: 1.8, fontFamily: "'Lora', Georgia, serif",
        }}>
          We're building scripture-grounded AI guides for all major world religions —
          Buddhism, Christianity, Islam, Hinduism, and beyond. Your support helps us expand
          scripture datasets, improve accuracy, and reach more people seeking spiritual guidance.
        </p>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", justifyContent: "center", marginTop: 8 }}>
          {features.map(({ Icon, label }, i) => (
            <div key={i} style={{
              padding: "6px 16px", borderRadius: 20,
              border: "1px solid rgba(102,126,234,0.3)",
              background: "rgba(102,126,234,0.08)",
              fontSize: 12, color: "#8899cc", fontFamily: "'Lora', serif",
              display: "flex", alignItems: "center", gap: 7,
            }}>
              <Icon size={12} color="#667eea" />
              {label}
            </div>
          ))}
        </div>
        <a
          href="https://fund.syetalabs.org/en/explore/multi-religious-ai-platform"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            marginTop: 12, padding: "14px 40px", borderRadius: 40,
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            color: "#fff", fontSize: 14, fontFamily: "'Cinzel', serif",
            letterSpacing: 1.5, fontWeight: 600, textDecoration: "none",
            boxShadow: "0 8px 32px rgba(102,126,234,0.4), 0 0 0 1px rgba(255,255,255,0.1)",
            transition: "transform 0.2s, box-shadow 0.2s",
            display: "inline-flex", alignItems: "center", gap: 10,
          }}
          onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-3px)"; e.currentTarget.style.boxShadow = "0 14px 40px rgba(102,126,234,0.55)"; }}
          onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)"; e.currentTarget.style.boxShadow = "0 8px 32px rgba(102,126,234,0.4)"; }}
        >
          <FaHeart size={14} color="rgba(255,255,255,0.8)" />
          Support Our Mission
        </a>
        <p style={{ fontSize: 11, color: "#334455", fontFamily: "'Cinzel', serif", letterSpacing: 1 }}>
          Every contribution makes a difference
        </p>
      </div>
    </div>
  );
};

// ─── Loading Screen ───────────────────────────────────────────
const LoadingScreen = ({ religion, onReady, onError }) => {
  const cfg = RELIGIONS[religion];
  const [dots, setDots]           = useState(0);
  const [statusMsg, setStatusMsg] = useState("Connecting to server…");
  const [phase, setPhase]         = useState("connecting");
  const pollRef                   = useRef(null);

  useEffect(() => {
    const id = setInterval(() => setDots(d => (d + 1) % 4), 500);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const resolveStatus = (data) => {
      if (data.status === "ready") {
        setPhase("indexing");
        setTimeout(() => {
          if (!cancelled) {
            setPhase("done");
            setStatusMsg("Ready!");
            setTimeout(() => { if (!cancelled) onReady(); }, 600);
          }
        }, 400);
        return true;
      }
      if (data.status === "error") {
        setPhase("error");
        setStatusMsg(data.error || "An error occurred while loading data.");
        onError(data.error || "Failed to load data.");
        return true;
      }
      return false;
    };

    const poll = async () => {
      // Step 1 — check server is alive
      let serverReachable = false;
      for (let attempt = 0; attempt < 5; attempt++) {
        try {
          const probe = await fetch(`${API_BASE}/health`);
          if (probe.ok) { serverReachable = true; break; }
        } catch { /* retry */ }
        await new Promise(r => setTimeout(r, 1500));
      }

      if (!serverReachable) {
        if (!cancelled) {
          setPhase("error");
          setStatusMsg("Could not reach the server. Is the backend running?");
          onError("Could not reach the server.");
        }
        return;
      }

      if (cancelled) return;

      // Step 2 — trigger religion data load on backend
      setPhase("downloading");
      setStatusMsg(cfg.loadingMsg);
      try {
        await fetch(`${API_BASE}/prepare/${religion}`, { method: "POST" });
      } catch { /* backend will still try */ }

      if (cancelled) return;

      // Step 3 — poll /status/:religion until ready
      pollRef.current = setInterval(async () => {
        if (cancelled) return;
        try {
          const res  = await fetch(`${API_BASE}/status/${religion}`);
          const data = await res.json();
          if (resolveStatus(data)) clearInterval(pollRef.current);
        } catch { /* network hiccup */ }
      }, POLL_INTERVAL_MS);
    };

    poll();
    return () => { cancelled = true; if (pollRef.current) clearInterval(pollRef.current); };
  }, [religion]);

  const dotStr = ".".repeat(dots + 1).padEnd(3, "\u00a0");

  const phaseSteps = [
    { key: "connecting",  label: "Connecting to server",      done: phase !== "connecting" },
    { key: "downloading", label: "Downloading scripture data", done: phase === "indexing" || phase === "done" },
    { key: "indexing",    label: "Loading search index",       done: phase === "done" },
  ];

  return (
    <div style={{
      minHeight: "100vh", width: "100vw", background: cfg.gradient,
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      fontFamily: "'Lora', Georgia, serif", padding: 24, position: "relative", overflow: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes spin   { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse  { 0%,100% { opacity: 0.5; transform: scale(1); } 50% { opacity: 1; transform: scale(1.04); } }
        .spin-ring  { animation: spin 1.4s linear infinite; }
        .pulse-icon { animation: pulse 2s ease-in-out infinite; }
        .fade-in    { animation: fadeIn 0.5s ease forwards; }
      `}</style>

      {/* Faint background watermark */}
      <div style={{
        position: "absolute", top: "50%", left: "50%",
        transform: "translate(-50%, -50%)",
        pointerEvents: "none", opacity: 0.05,
      }}>
        <ReligionSymbol symbolKey={cfg.symbolKey} size={320} color={cfg.accentColor} />
      </div>

      <div className="fade-in" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 28, position: "relative", zIndex: 1 }}>
        <div style={{ position: "relative", width: 110, height: 110 }}>
          {phase !== "error" && phase !== "done" && (
            <svg className="spin-ring" viewBox="0 0 110 110"
              style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
              <circle cx="55" cy="55" r="48"
                stroke={cfg.accentColor} strokeWidth="4" fill="none"
                strokeDasharray="200 100" strokeLinecap="round" opacity="0.8" />
            </svg>
          )}
          <div className="pulse-icon" style={{
            position: "absolute", inset: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            {phase === "done"
              ? <FaCircleCheck size={46} color={cfg.accentColor} />
              : phase === "error"
              ? <MdError size={52} color="#e53935" />
              : <ReligionSymbol symbolKey={cfg.symbolKey} size={48} color={cfg.accentDark} />
            }
          </div>
        </div>

        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: "'Cinzel', serif", fontSize: 22, fontWeight: 700, color: cfg.accentDark, letterSpacing: 1, marginBottom: 6 }}>
            {cfg.label}
          </div>
          <div style={{ fontSize: 13.5, color: cfg.textMuted, letterSpacing: 0.3 }}>
            {phase === "error" ? statusMsg : `${statusMsg}${dotStr}`}
          </div>
        </div>

        {phase !== "error" && (
          <div style={{
            background: cfg.headerBg, borderRadius: 16, border: `1px solid ${cfg.border}`,
            padding: "18px 28px", minWidth: 280,
            display: "flex", flexDirection: "column", gap: 12,
            boxShadow: `0 4px 24px ${cfg.cardGlow}`,
          }}>
            {phaseSteps.map(step => {
              const isActive = (
                (step.key === "connecting"  && phase === "connecting")  ||
                (step.key === "downloading" && phase === "downloading") ||
                (step.key === "indexing"    && phase === "indexing")
              );
              const isDone = step.done || phase === "done";
              return (
                <div key={step.key} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{
                    width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
                    background: isDone
                      ? `linear-gradient(135deg, ${cfg.accentColor}, ${cfg.accentDark})`
                      : isActive ? cfg.hover : "transparent",
                    border: `2px solid ${isDone ? "transparent" : isActive ? cfg.accentColor : cfg.border}`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    transition: "all 0.3s",
                  }}>
                    {isDone
                      ? <FaCircleCheck size={11} color="#fff" />
                      : isActive
                      ? <span style={{ width: 8, height: 8, borderRadius: "50%", background: cfg.accentColor, display: "block" }} />
                      : null}
                  </div>
                  <span style={{
                    fontSize: 13, color: isDone ? cfg.text : isActive ? cfg.accentDark : cfg.textMuted,
                    fontWeight: isActive || isDone ? 500 : 400, transition: "color 0.3s",
                  }}>
                    {step.label}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {phase === "error" && (
          <div style={{
            background: "#fff3e0", border: "1px solid #f5c78e", borderRadius: 10,
            padding: "10px 20px", fontSize: 13, color: "#a0622a",
            display: "flex", alignItems: "center", gap: 8,
          }}>
            <MdError size={16} />
            {statusMsg}
          </div>
        )}

        <button
          onClick={() => { if (pollRef.current) clearInterval(pollRef.current); onError(null); }}
          style={{
            marginTop: 4, padding: "8px 22px", borderRadius: 20,
            border: `1.5px solid ${cfg.border}`, background: "transparent",
            fontFamily: "'Lora', serif", fontSize: 12.5, color: cfg.textMuted, cursor: "pointer",
            display: "flex", alignItems: "center", gap: 6,
          }}>
          <FaChevronLeft size={11} /> Choose a different tradition
        </button>
      </div>
    </div>
  );
};

// ─── Main export ──────────────────────────────────────────────
export default function LandingPage({ onEnterChat }) {
  const [screen, setScreen]                     = useState("select");
  const [selectedReligion, setSelectedReligion] = useState(null);
  const [showBanner, setShowBanner]             = useState(true);

  const handleSelect = (religion) => { setSelectedReligion(religion); setScreen("loading"); };
  const handleReady  = () => { onEnterChat(selectedReligion); };
  const handleError  = () => { setScreen("select"); setSelectedReligion(null); };

  if (screen === "loading") {
    return (
      <>
        {showBanner && <DevBanner onClose={() => setShowBanner(false)} />}
        <LoadingScreen religion={selectedReligion} onReady={handleReady} onError={handleError} />
      </>
    );
  }

  return (
    <>
      {showBanner && <DevBanner onClose={() => setShowBanner(false)} />}
      <ReligionSelectPage onSelect={handleSelect} bannerVisible={showBanner} />
    </>
  );
}