import { useState, useRef, useEffect } from "react";

// ─── API URL ─────────────────────────────────────────────────────────────────
// Change this to your Render URL when deploying:
// const API_URL = "https://svara-vaidya.onrender.com";
const API_URL = "https://mannatgupta512-svara-vaidya.hf.space";

// ─── Helpers ──────────────────────────────────────────────────────────────────
const gradeColor = (score) => {
  if (score >= 85) return "#22c55e";
  if (score >= 70) return "#f59e0b";
  if (score >= 50) return "#f97316";
  return "#ef4444";
};

const gradeEmoji = (score) => {
  if (score >= 85) return "✅";
  if (score >= 70) return "🔶";
  if (score >= 50) return "⚠️";
  return "❌";
};

// ─── Score bar component ──────────────────────────────────────────────────────
const ScoreBar = ({ label, score, sublabel }) => (
  <div style={{ marginBottom: 14 }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
      <span style={{ fontSize: 13, color: "#cbd5e1" }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 700, color: gradeColor(score) }}>
        {score}% {gradeEmoji(score)}
      </span>
    </div>
    <div style={{ background: "#0f172a", borderRadius: 6, height: 8 }}>
      <div style={{
        width: `${Math.min(score, 100)}%`, height: 8, borderRadius: 6,
        background: gradeColor(score), transition: "width 1s ease"
      }} />
    </div>
    {sublabel && (
      <div style={{ fontSize: 11, color: "#64748b", marginTop: 3 }}>{sublabel}</div>
    )}
  </div>
);

// ─── Word svara card ──────────────────────────────────────────────────────────
// Matches the word_svara_feedback list from compare_word_svara_maps()
// Each item has: word, word_num, ref_accent, user_accent, ref_shape,
//                user_shape, accent_ok, shape_ok, accent_fix, shape_fix
const WordCard = ({ item }) => {
  const ok = item.accent_ok && item.shape_ok;
  return (
    <div style={{
      background: ok ? "#052e16" : "#2d0f0f",
      border: `1px solid ${ok ? "#16a34a" : "#991b1b"}`,
      borderRadius: 10, padding: "12px 16px", marginBottom: 10
    }}>
      {/* Word header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: ok ? 0 : 10 }}>
        <span style={{
          background: ok ? "#16a34a" : "#991b1b",
          color: "#fff", borderRadius: 20, padding: "2px 10px",
          fontSize: 12, fontWeight: 700
        }}>
          Word {item.word_num}
        </span>
        <span style={{ fontSize: 17, fontWeight: 800, color: "#f1f5f9" }}>
          '{item.word}'
        </span>
        <span style={{ marginLeft: "auto", fontSize: 18 }}>{ok ? "✅" : "❌"}</span>
      </div>

      {/* Accent error */}
      {!item.accent_ok && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4 }}>
            <span style={{ color: "#f87171", fontWeight: 700 }}>Accent</span>
            {"  "}Expected{" "}
            <span style={{
              background: "#14532d", color: "#86efac",
              padding: "1px 8px", borderRadius: 4, fontWeight: 700
            }}>{item.ref_accent}</span>
            {"  "}You sang{" "}
            <span style={{
              background: "#450a0a", color: "#fca5a5",
              padding: "1px 8px", borderRadius: 4, fontWeight: 700
            }}>{item.user_accent}</span>
          </div>
          {item.accent_fix && (
            <div style={{
              background: "#1e3a5f", borderLeft: "3px solid #3b82f6",
              padding: "8px 12px", borderRadius: 6,
              fontSize: 12, color: "#bfdbfe", lineHeight: 1.6
            }}>
              💡 {item.accent_fix}
            </div>
          )}
        </div>
      )}

      {/* Shape error */}
      {!item.shape_ok && (
        <div>
          <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 4 }}>
            <span style={{ color: "#f87171", fontWeight: 700 }}>Shape</span>
            {"  "}Expected{" "}
            <span style={{
              background: "#14532d", color: "#86efac",
              padding: "1px 8px", borderRadius: 4, fontWeight: 700
            }}>{item.ref_shape}</span>
            {"  "}You sang{" "}
            <span style={{
              background: "#450a0a", color: "#fca5a5",
              padding: "1px 8px", borderRadius: 4, fontWeight: 700
            }}>{item.user_shape}</span>
          </div>
          {item.shape_fix && (
            <div style={{
              background: "#1e3a5f", borderLeft: "3px solid #3b82f6",
              padding: "8px 12px", borderRadius: 6,
              fontSize: 12, color: "#bfdbfe", lineHeight: 1.6
            }}>
              💡 {item.shape_fix}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ─── Feedback bullet list ─────────────────────────────────────────────────────
const FeedbackList = ({ items, color = "#bfdbfe", bg = "#1e3a5f", border = "#3b82f6" }) => (
  <>
    {items.map((item, i) => (
      <div key={i} style={{
        background: bg, borderLeft: `3px solid ${border}`,
        padding: "8px 12px", borderRadius: 6,
        fontSize: 13, color, lineHeight: 1.6, marginBottom: 6
      }}>
        • {item}
      </div>
    ))}
  </>
);

// ─── Tab button ───────────────────────────────────────────────────────────────
const Tab = ({ label, active, onClick }) => (
  <button onClick={onClick} style={{
    padding: "8px 18px", borderRadius: 8, border: "none",
    cursor: "pointer", fontSize: 13, fontWeight: 600,
    background: active ? "#3b82f6" : "#1e293b",
    color: active ? "#fff" : "#94a3b8",
    transition: "all 0.2s"
  }}>
    {label}
  </button>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [recording,  setRecording]  = useState(false);
  const [audioBlob,  setAudioBlob]  = useState(null);
  const [audioURL,   setAudioURL]   = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [error,      setError]      = useState(null);
  const [activeTab,  setActiveTab]  = useState("svara");
  const [apiStatus,  setApiStatus]  = useState("checking"); // checking | ok | down

  const mediaRef    = useRef(null);
  const chunksRef   = useRef([]);
  const fileInputRef = useRef(null);

  // ── Keep backend alive + check status ──────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API_URL}/health`);
        setApiStatus(r.ok ? "ok" : "down");
      } catch {
        setApiStatus("down");
      }
    };
    check();
    const interval = setInterval(check, 600000); // ping every 10 min
    return () => clearInterval(interval);
  }, []);

  // ── Microphone recording ───────────────────────────────────────────────────
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        setAudioBlob(blob);
        setAudioURL(URL.createObjectURL(blob));
        stream.getTracks().forEach(t => t.stop());
        setResult(null);
        setError(null);
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
      setResult(null);
      setError(null);
    } catch {
      setError("Microphone access denied. Please allow microphone permission in your browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRef.current) { mediaRef.current.stop(); setRecording(false); }
  };

  // ── File upload ────────────────────────────────────────────────────────────
  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setAudioBlob(file);
    setAudioURL(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  // ── Analyze ────────────────────────────────────────────────────────────────
  // Sends audio to POST /analyze
  // Response structure matches generate_full_feedback() return dict:
  //   result.mantra, result.overall_score, result.overall_grade
  //   result.dimensions.uccharana / svara / laya / nada
  //   result.svara_word_detail   ← word-by-word breakdown
  //   result.scores              ← all numeric scores
  //   result.priority            ← ranked focus areas
  //   result.sadhana_tip
  //   result.all_praises
  const analyze = async () => {
    if (!audioBlob) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", audioBlob, "user_chant.wav");
      const resp = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: "Server error" }));
        throw new Error(err.detail || "Analysis failed");
      }
      const data = await resp.json();
      setResult(data);
      setActiveTab("svara");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // ── Shortcuts ──────────────────────────────────────────────────────────────
  const r  = result;
  // scores come from result.scores (added by analyze_chant())
  const sc = r?.scores || {};
  // dimensions come from generate_full_feedback() return dict
  const d  = r?.dimensions || {};
  // word-level svara breakdown
  const wordDetail = r?.svara_word_detail || [];

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0f172a",
        color: "#f1f5f9",
        fontFamily: "'Segoe UI', system-ui, sans-serif",
        padding: "24px 16px",
      }}
    >
      <div style={{ maxWidth: 740, margin: "0 auto" }}>
        {/* ── Header ── */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{ fontSize: 36, marginBottom: 6 }}>🪔</div>
          <h1
            style={{
              fontSize: 22,
              fontWeight: 900,
              margin: "0 0 6px",
              background: "linear-gradient(135deg, #f59e0b, #ef4444)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Svara Vaidya
          </h1>
          <p style={{ color: "#94a3b8", fontSize: 13, margin: "0 0 10px" }}>
            Vedic Mantra Chanting Analyzer · स्वर वैद्य
          </p>

          {/* API status indicator */}
          <div style={{ marginTop: 10 }}>
            <span
              style={{
                fontSize: 11,
                padding: "3px 10px",
                borderRadius: 20,
                background:
                  apiStatus === "ok"
                    ? "#14532d"
                    : apiStatus === "down"
                      ? "#450a0a"
                      : "#1e293b",
                color:
                  apiStatus === "ok"
                    ? "#86efac"
                    : apiStatus === "down"
                      ? "#fca5a5"
                      : "#94a3b8",
              }}
            >
              {apiStatus === "ok"
                ? "● API Connected"
                : apiStatus === "down"
                  ? "● API Offline — start uvicorn"
                  : "● Checking API..."}
            </span>
          </div>
        </div>

        {/* ── Record / Upload card ── */}
        <div
          style={{
            background: "#1e293b",
            borderRadius: 16,
            padding: 24,
            marginBottom: 20,
          }}
        >
          <h2
            style={{
              fontSize: 15,
              fontWeight: 700,
              marginBottom: 16,
              color: "#cbd5e1",
            }}
          >
            🎙️ Record or Upload Your Chant
          </h2>

          <div
            style={{
              display: "flex",
              gap: 10,
              flexWrap: "wrap",
              marginBottom: 16,
              justifyContent: "center",
            }}
          >
            {!recording ? (
              <button
                onClick={startRecording}
                disabled={loading}
                style={{
                  background: "#dc2626",
                  color: "#fff",
                  border: "none",
                  borderRadius: 10,
                  padding: "11px 22px",
                  fontSize: 14,
                  fontWeight: 700,
                  cursor: loading ? "not-allowed" : "pointer",
                  opacity: loading ? 0.5 : 1,
                }}
              >
                🔴 Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                style={{
                  background: "#7f1d1d",
                  color: "#fff",
                  border: "2px solid #dc2626",
                  borderRadius: 10,
                  padding: "11px 22px",
                  fontSize: 14,
                  fontWeight: 700,
                  cursor: "pointer",
                }}
              >
                ⏹ Stop Recording
              </button>
            )}

            <button
              onClick={() => fileInputRef.current.click()}
              disabled={loading}
              style={{
                background: "#1e40af",
                color: "#fff",
                border: "none",
                borderRadius: 10,
                padding: "11px 22px",
                fontSize: 14,
                fontWeight: 700,
                cursor: loading ? "not-allowed" : "pointer",
                opacity: loading ? 0.5 : 1,
              }}
            >
              📁 Upload Audio
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".wav,.mp3,.m4a,.ogg,.flac"
              style={{ display: "none" }}
              onChange={handleFile}
            />
          </div>

          {recording && (
            <div style={{ color: "#f87171", fontSize: 13, marginBottom: 10 }}>
              ● Recording... chant your mantra now, then press Stop
            </div>
          )}

          {audioURL && (
            <audio
              controls
              src={audioURL}
              style={{ width: "100%", borderRadius: 8, marginBottom: 14 }}
            />
          )}

          {audioBlob && !loading && !recording && (
            <button
              onClick={analyze}
              style={{
                background: "linear-gradient(135deg, #f59e0b, #ef4444)",
                color: "#fff",
                border: "none",
                borderRadius: 10,
                padding: "14px 0",
                fontSize: 16,
                fontWeight: 800,
                cursor: "pointer",
                width: "100%",
              }}
            >
              🔍 Analyze Chant
            </button>
          )}

          {loading && (
            <div
              style={{
                background: "#0f172a",
                borderRadius: 10,
                padding: 24,
                textAlign: "center",
                color: "#94a3b8",
              }}
            >
              <div style={{ fontSize: 36, marginBottom: 8 }}>🔄</div>
              <div style={{ fontSize: 14 }}>Analyzing your chant...</div>
              <div style={{ fontSize: 12, marginTop: 4, color: "#475569" }}>
                Whisper transcription + pitch analysis may take 15–30 seconds
              </div>
            </div>
          )}

          {error && (
            <div
              style={{
                background: "#450a0a",
                border: "1px solid #991b1b",
                borderRadius: 10,
                padding: 14,
                color: "#fca5a5",
                fontSize: 13,
                marginTop: 12,
              }}
            >
              ❌ {error}
            </div>
          )}
        </div>

        {/* ── Results ── */}
        {r && (
          <>
            {/* Overall score card */}
            <div
              style={{
                background: "#1e293b",
                borderRadius: 16,
                padding: 24,
                marginBottom: 16,
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 13, color: "#64748b", marginBottom: 4 }}>
                Mantra: <strong style={{ color: "#f1f5f9" }}>{r.mantra}</strong>
              </div>
              {r.user_transcript && (
                <div
                  style={{ fontSize: 13, color: "#64748b", marginBottom: 16 }}
                >
                  Whisper heard:{" "}
                  <em style={{ color: "#cbd5e1" }}>"{r.user_transcript}"</em>
                </div>
              )}
              <div
                style={{
                  fontSize: 72,
                  fontWeight: 900,
                  lineHeight: 1,
                  color: gradeColor(r.overall_score || sc.overall || 0),
                }}
              >
                {r.overall_score || sc.overall || 0}%
              </div>
              <div style={{ fontSize: 26, marginTop: 8 }}>
                <span style={{ color: "#f1f5f9", fontWeight: 800 }}>
                  {r.overall_grade}
                </span>
                <span style={{ fontSize: 13, color: "#64748b", marginLeft: 8 }}>
                  {r.overall_grade === "Uttama"
                    ? "Excellent"
                    : r.overall_grade === "Madhyama"
                      ? "Good — keep practicing"
                      : r.overall_grade === "Adi"
                        ? "Foundation level"
                        : "Just starting"}
                </span>
              </div>
            </div>

            {/* Dimension score bars */}
            <div
              style={{
                background: "#1e293b",
                borderRadius: 16,
                padding: 24,
                marginBottom: 16,
              }}
            >
              <h2
                style={{
                  fontSize: 15,
                  fontWeight: 700,
                  marginBottom: 18,
                  color: "#cbd5e1",
                }}
              >
                📊 Dimension Scores
              </h2>

              <ScoreBar
                label="🔤 Ucchāraṇa (Pronunciation)"
                score={d.uccharana?.score ?? sc.text_similarity ?? 0}
              />
              <ScoreBar
                label="🔔 Nāda (Voice Quality)"
                score={d.nada?.score ?? sc.mfcc_similarity ?? 0}
              />
              <ScoreBar
                label="🎵 Svara (Pitch / Accent)"
                score={d.svara?.score ?? sc.pitch_combined ?? 0}
              />
              {/* Svara sub-scores */}
              <div
                style={{
                  paddingLeft: 20,
                  borderLeft: "2px solid #334155",
                  marginBottom: 14,
                }}
              >
                <ScoreBar
                  label="↳ Contour shape"
                  score={
                    sc.pitch_breakdown?.contour_shape ?? d.svara?.score ?? 0
                  }
                />
                <ScoreBar
                  label="↳ Zone accuracy"
                  score={sc.pitch_breakdown?.zone_accuracy ?? 0}
                />
                <ScoreBar
                  label="↳ Accent placement"
                  score={sc.pitch_breakdown?.accent_placement ?? 0}
                />
                <ScoreBar
                  label="↳ Syllable shape"
                  score={sc.pitch_breakdown?.syllable_shape ?? 0}
                />
                <ScoreBar
                  label="↳ Slope/steepness"
                  score={sc.pitch_breakdown?.slope_steepness ?? 0}
                />
              </div>
              <ScoreBar
                label="🥁 Laya (Rhythm / Tempo)"
                score={d.laya?.score ?? sc.tempo_similarity ?? 0}
              />
            </div>

            {/* Priority focus areas */}
            {r.priority && r.priority.length > 0 && (
              <div
                style={{
                  background: "#1e293b",
                  borderRadius: 16,
                  padding: 24,
                  marginBottom: 16,
                }}
              >
                <h2
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    marginBottom: 16,
                    color: "#cbd5e1",
                  }}
                >
                  🎯 Focus Areas (fix these first)
                </h2>
                {r.priority.map((p, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "10px 0",
                      borderBottom:
                        i < r.priority.length - 1
                          ? "1px solid #1e293b"
                          : "none",
                    }}
                  >
                    <div
                      style={{ display: "flex", alignItems: "center", gap: 10 }}
                    >
                      <span
                        style={{
                          background:
                            i === 0
                              ? "#dc2626"
                              : i === 1
                                ? "#f59e0b"
                                : "#334155",
                          color: "#fff",
                          borderRadius: 20,
                          width: 24,
                          height: 24,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 12,
                          fontWeight: 800,
                          flexShrink: 0,
                        }}
                      >
                        {i + 1}
                      </span>
                      <span style={{ fontSize: 14 }}>{p.dimension}</span>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div
                        style={{
                          fontSize: 14,
                          fontWeight: 700,
                          color: gradeColor(p.score),
                        }}
                      >
                        {p.score}%
                      </div>
                      <div style={{ fontSize: 11, color: "#475569" }}>
                        impact: {p.impact}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* What you did well */}
            {r.all_praises && r.all_praises.length > 0 && (
              <div
                style={{
                  background: "#052e16",
                  border: "1px solid #16a34a",
                  borderRadius: 16,
                  padding: 20,
                  marginBottom: 16,
                }}
              >
                <h2
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    marginBottom: 12,
                    color: "#86efac",
                  }}
                >
                  ✨ Sādhu! — What You Did Well
                </h2>
                {r.all_praises.map((p, i) => (
                  <div
                    key={i}
                    style={{ fontSize: 13, color: "#bbf7d0", marginBottom: 6 }}
                  >
                    ✓ {p}
                  </div>
                ))}
              </div>
            )}

            {/* ── Detail tabs ── */}
            <div
              style={{
                background: "#1e293b",
                borderRadius: 16,
                padding: 24,
                marginBottom: 16,
              }}
            >
              <div
                style={{
                  display: "flex",
                  gap: 8,
                  marginBottom: 20,
                  flexWrap: "wrap",
                }}
              >
                <Tab
                  label="🎵 Svara"
                  active={activeTab === "svara"}
                  onClick={() => setActiveTab("svara")}
                />
                <Tab
                  label="🔤 Ucchāraṇa"
                  active={activeTab === "uccharana"}
                  onClick={() => setActiveTab("uccharana")}
                />
                <Tab
                  label="🥁 Laya"
                  active={activeTab === "laya"}
                  onClick={() => setActiveTab("laya")}
                />
                <Tab
                  label="🔔 Nāda"
                  active={activeTab === "nada"}
                  onClick={() => setActiveTab("nada")}
                />
              </div>

              {/* ── SVARA TAB ── */}
              {activeTab === "svara" && (
                <div>
                  {/* Summary issues from pitch_feedback() */}
                  {d.svara?.issues && d.svara.issues.length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                      <div
                        style={{
                          fontSize: 13,
                          color: "#94a3b8",
                          marginBottom: 8,
                        }}
                      >
                        Summary:
                      </div>
                      <FeedbackList items={d.svara.issues} />
                    </div>
                  )}

                  {/* Word-by-word breakdown — from svara_word_detail */}
                  <div
                    style={{ fontSize: 13, color: "#94a3b8", marginBottom: 10 }}
                  >
                    Word-by-word Svara Breakdown:
                  </div>
                  {wordDetail.length === 0 ? (
                    <div
                      style={{
                        background: "#052e16",
                        border: "1px solid #16a34a",
                        borderRadius: 10,
                        padding: 14,
                        color: "#86efac",
                        fontSize: 14,
                      }}
                    >
                      ✅ All words have correct svara accent and shape!
                    </div>
                  ) : (
                    wordDetail.map((item, i) => (
                      <WordCard key={i} item={item} />
                    ))
                  )}

                  {/* Zone feedback */}
                  {d.svara?.fixes && d.svara.fixes.length > 0 && (
                    <div style={{ marginTop: 16 }}>
                      <div
                        style={{
                          fontSize: 13,
                          color: "#94a3b8",
                          marginBottom: 8,
                        }}
                      >
                        General practice techniques:
                      </div>
                      <FeedbackList items={d.svara.fixes} />
                    </div>
                  )}
                </div>
              )}

              {/* ── UCCHARANA TAB ── */}
              {activeTab === "uccharana" && (
                <div>
                  {/* Sub-scores */}
                  <div
                    style={{
                      display: "flex",
                      gap: 10,
                      marginBottom: 16,
                      flexWrap: "wrap",
                    }}
                  >
                    {[
                      ["Char", sc.text_similarity],
                      ["Word", d.uccharana?.word_sim],
                      ["Syllable", d.uccharana?.syl_sim],
                    ].map(
                      ([label, val]) =>
                        val != null && (
                          <div
                            key={label}
                            style={{
                              background: "#0f172a",
                              borderRadius: 10,
                              padding: "12px 16px",
                              flex: 1,
                              minWidth: 80,
                            }}
                          >
                            <div style={{ fontSize: 11, color: "#64748b" }}>
                              {label}
                            </div>
                            <div
                              style={{
                                fontSize: 22,
                                fontWeight: 800,
                                color: gradeColor(val),
                              }}
                            >
                              {Math.round(val)}%
                            </div>
                          </div>
                        ),
                    )}
                  </div>

                  {/* Issues */}
                  {d.uccharana?.issues && d.uccharana.issues.length > 0 ? (
                    <div style={{ marginBottom: 14 }}>
                      <div
                        style={{
                          fontSize: 13,
                          color: "#94a3b8",
                          marginBottom: 8,
                        }}
                      >
                        Issues:
                      </div>
                      <FeedbackList
                        items={d.uccharana.issues}
                        bg="#2d0f0f"
                        border="#dc2626"
                        color="#fca5a5"
                      />
                    </div>
                  ) : (
                    <div
                      style={{
                        color: "#22c55e",
                        fontSize: 14,
                        marginBottom: 14,
                      }}
                    >
                      ✅ No pronunciation issues detected
                    </div>
                  )}

                  {/* Fixes */}
                  {d.uccharana?.fixes && d.uccharana.fixes.length > 0 && (
                    <div>
                      <div
                        style={{
                          fontSize: 13,
                          color: "#94a3b8",
                          marginBottom: 8,
                        }}
                      >
                        How to improve:
                      </div>
                      <FeedbackList items={d.uccharana.fixes} />
                    </div>
                  )}
                </div>
              )}

              {/* ── LAYA TAB ── */}
              {activeTab === "laya" && (
                <div>
                  <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
                    <div
                      style={{
                        background: "#0f172a",
                        borderRadius: 10,
                        padding: "14px 20px",
                        flex: 1,
                        textAlign: "center",
                      }}
                    >
                      <div
                        style={{
                          fontSize: 11,
                          color: "#64748b",
                          marginBottom: 4,
                        }}
                      >
                        Rhythm Score
                      </div>
                      <div
                        style={{
                          fontSize: 36,
                          fontWeight: 900,
                          color: gradeColor(d.laya?.score ?? 0),
                        }}
                      >
                        {d.laya?.score ?? 0}%
                      </div>
                    </div>
                    {d.laya?.duration_ratio != null && (
                      <div
                        style={{
                          background: "#0f172a",
                          borderRadius: 10,
                          padding: "14px 20px",
                          flex: 1,
                          textAlign: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 11,
                            color: "#64748b",
                            marginBottom: 4,
                          }}
                        >
                          Duration vs Reference
                        </div>
                        <div
                          style={{
                            fontSize: 36,
                            fontWeight: 900,
                            color:
                              Math.abs(d.laya.duration_ratio - 1) < 0.15
                                ? "#22c55e"
                                : "#f97316",
                          }}
                        >
                          {d.laya.duration_ratio}×
                        </div>
                        <div style={{ fontSize: 11, color: "#64748b" }}>
                          {d.laya.duration_ratio > 1.15
                            ? "Too slow — speed up"
                            : d.laya.duration_ratio < 0.85
                              ? "Too fast — slow down"
                              : "Good pace ✓"}
                        </div>
                      </div>
                    )}
                  </div>

                  {d.laya?.issues && d.laya.issues.length > 0 && (
                    <div style={{ marginBottom: 14 }}>
                      <div
                        style={{
                          fontSize: 13,
                          color: "#94a3b8",
                          marginBottom: 8,
                        }}
                      >
                        Issues:
                      </div>
                      <FeedbackList
                        items={d.laya.issues}
                        bg="#2d0f0f"
                        border="#dc2626"
                        color="#fca5a5"
                      />
                    </div>
                  )}
                  {d.laya?.fixes && d.laya.fixes.length > 0 && (
                    <div>
                      <div
                        style={{
                          fontSize: 13,
                          color: "#94a3b8",
                          marginBottom: 8,
                        }}
                      >
                        How to improve:
                      </div>
                      <FeedbackList items={d.laya.fixes} />
                    </div>
                  )}
                  {(!d.laya?.issues || d.laya.issues.length === 0) && (
                    <div style={{ color: "#22c55e", fontSize: 14 }}>
                      ✅ Laya is steady — good rhythmic flow
                    </div>
                  )}
                </div>
              )}

              {/* ── NADA TAB ── */}
              {activeTab === "nada" && (
                <div>
                  <div
                    style={{
                      background: "#0f172a",
                      borderRadius: 10,
                      padding: 20,
                      textAlign: "center",
                      marginBottom: 16,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 11,
                        color: "#64748b",
                        marginBottom: 4,
                      }}
                    >
                      Voice Quality Score
                    </div>
                    <div
                      style={{
                        fontSize: 56,
                        fontWeight: 900,
                        color: gradeColor(d.nada?.score ?? 0),
                      }}
                    >
                      {d.nada?.score ?? 0}%
                    </div>
                  </div>

                  {d.nada?.issues && d.nada.issues.length > 0 ? (
                    <div style={{ marginBottom: 14 }}>
                      <div
                        style={{
                          fontSize: 13,
                          color: "#94a3b8",
                          marginBottom: 8,
                        }}
                      >
                        Issues:
                      </div>
                      <FeedbackList
                        items={d.nada.issues}
                        bg="#2d0f0f"
                        border="#dc2626"
                        color="#fca5a5"
                      />
                    </div>
                  ) : (
                    <div
                      style={{
                        color: "#22c55e",
                        fontSize: 14,
                        marginBottom: 14,
                      }}
                    >
                      ✅ Nāda quality is authentic
                    </div>
                  )}

                  {d.nada?.fixes && d.nada.fixes.length > 0 && (
                    <div>
                      <div
                        style={{
                          fontSize: 13,
                          color: "#94a3b8",
                          marginBottom: 8,
                        }}
                      >
                        How to improve:
                      </div>
                      <FeedbackList items={d.nada.fixes} />
                    </div>
                  )}

                  {d.nada?.praises && d.nada.praises.length > 0 && (
                    <div style={{ marginTop: 12 }}>
                      {d.nada.praises.map((p, i) => (
                        <div key={i} style={{ fontSize: 13, color: "#86efac" }}>
                          ✓ {p}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Sadhana tip */}
            <div
              style={{
                background: "linear-gradient(135deg, #1c1917, #292524)",
                border: "1px solid #92400e",
                borderRadius: 16,
                padding: 24,
                marginBottom: 32,
              }}
            >
              <div style={{ fontSize: 24, marginBottom: 8 }}>🪔</div>
              <h3
                style={{
                  fontSize: 14,
                  fontWeight: 800,
                  color: "#fbbf24",
                  marginBottom: 8,
                }}
              >
                Daily Sādhana Recommendation
              </h3>
              <p
                style={{
                  fontSize: 14,
                  color: "#d6d3d1",
                  lineHeight: 1.7,
                  margin: 0,
                }}
              >
                {r.sadhana_tip}
              </p>
            </div>
          </>
        )}

        <div
          style={{
            textAlign: "center",
            color: "#1e293b",
            fontSize: 13,
            paddingBottom: 16,
          }}
        >
          ॐ शान्तिः शान्तिः शान्तिः
        </div>
      </div>
    </div>
  );
}