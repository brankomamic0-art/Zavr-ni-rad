import React, { useCallback, useEffect, useMemo, useState } from "react";

// ═══════════════════════════════════════════════════════════════
// Prediktivni sustav za analizu sportskih događaja — sučelje
//
// Bez prijave i bez početne stranice: pri otvaranju se odmah dohvaća
// GET /api/matches i prikazuju se predikcije za današnji dan.
// ═══════════════════════════════════════════════════════════════

const API_BASE = process.env.REACT_APP_API_URL || "";

const MARKETS = [
  { key: "over25", label: "Over 2.5", color: "#10b981" },
  { key: "under25", label: "Under 2.5", color: "#3b82f6" },
  { key: "btts", label: "BTTS Da", color: "#f59e0b" },
  { key: "btts_no", label: "BTTS Ne", color: "#8b5cf6" },
];

// Zastavice po zemlji — čisto kozmetika u zaglavlju kartice.
const FLAGS = {
  Albania: "🇦🇱", Andorra: "🇦🇩", Armenia: "🇦🇲", Austria: "🇦🇹", Azerbaijan: "🇦🇿",
  Belarus: "🇧🇾", Belgium: "🇧🇪", Bulgaria: "🇧🇬", Croatia: "🇭🇷", Cyprus: "🇨🇾",
  "Czech-Republic": "🇨🇿", Denmark: "🇩🇰", England: "🏴󠁧󠁢󠁥󠁮󠁧󠁿", Estonia: "🇪🇪", Finland: "🇫🇮",
  France: "🇫🇷", Georgia: "🇬🇪", Germany: "🇩🇪", Greece: "🇬🇷", Hungary: "🇭🇺",
  Iceland: "🇮🇸", Ireland: "🇮🇪", Israel: "🇮🇱", Italy: "🇮🇹", Kazakhstan: "🇰🇿",
  Latvia: "🇱🇻", Lithuania: "🇱🇹", Luxembourg: "🇱🇺", Malta: "🇲🇹", Moldova: "🇲🇩",
  Montenegro: "🇲🇪", Netherlands: "🇳🇱", "North-Macedonia": "🇲🇰", Norway: "🇳🇴",
  Poland: "🇵🇱", Portugal: "🇵🇹", Romania: "🇷🇴", Russia: "🇷🇺", Scotland: "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
  Serbia: "🇷🇸", Slovakia: "🇸🇰", Slovenia: "🇸🇮", Spain: "🇪🇸", Sweden: "🇸🇪",
  Switzerland: "🇨🇭", Turkey: "🇹🇷", Ukraine: "🇺🇦", Wales: "🏴󠁧󠁢󠁷󠁬󠁳󠁿", World: "🌍",
};

const MONO = "'JetBrains Mono',ui-monospace,monospace";
const C = {
  bg: "#0a0e1a", panel: "rgba(15,23,42,0.55)", line: "rgba(255,255,255,0.06)",
  text: "#e2e8f0", dim: "#94a3b8", faint: "#64748b", green: "#10b981",
  amber: "#f59e0b", red: "#ef4444",
};

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:${C.bg}}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:3px}
input,select{font-family:inherit}
input:focus,select:focus{outline:none;border-color:rgba(16,185,129,0.5)}
`;

// ─── Pomoćne funkcije ───

function useIsMobile(breakpoint = 768) {
  const [isMobile, setIsMobile] = useState(
    typeof window !== "undefined" && window.innerWidth < breakpoint
  );
  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < breakpoint);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [breakpoint]);
  return isMobile;
}

const pct = (p) => (p === null || p === undefined ? "–" : `${Math.round(p * 100)}%`);

/** Današnji datum u LOKALNOJ zoni.
 *  `toISOString()` pretvara u UTC, pa bi u Hrvatskoj (UTC+2) između ponoći i
 *  2 ujutro vratio jučerašnji datum i stranica bi se otvorila na krivom danu. */
function todayLocal() {
  const d = new Date();
  const offsetMs = d.getTimezoneOffset() * 60000;
  return new Date(d.getTime() - offsetMs).toISOString().slice(0, 10);
}

/** Stvaran ishod odigrane utakmice za zadano tržište. */
function actualOutcome(match, market) {
  const { home_goals: hg, away_goals: ag } = match;
  if (hg === null || hg === undefined || ag === null || ag === undefined) return null;
  const total = hg + ag;
  const btts = hg > 0 && ag > 0;
  switch (market) {
    case "over25": return total > 2.5;
    case "under25": return total < 2.5;
    case "btts": return btts;
    case "btts_no": return !btts;
    default: return null;
  }
}

function probColor(p) {
  if (p === null || p === undefined) return C.faint;
  if (p >= 0.7) return C.green;
  if (p >= 0.5) return C.amber;
  return C.red;
}

/** Broj pozitivnih ishoda u prozoru forme za zadano tržište.
 *  Under 2.5 i BTTS Ne su komplementi, pa se izvode oduzimanjem. */
function formCount(block, market) {
  if (!block || !block.n) return null;
  switch (market) {
    case "over25": return block.over25;
    case "under25": return block.over25 === null ? null : block.n - block.over25;
    case "btts": return block.btts;
    case "btts_no": return block.btts === null ? null : block.n - block.btts;
    default: return null;
  }
}

// ═══════════════════════════════════════════════════════════════
// Sastavnice sučelja
// ═══════════════════════════════════════════════════════════════

function FormBar({ label, count, total }) {
  if (count === null || count === undefined || !total) return null;
  const ratio = count / total;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 10, color: C.faint, fontFamily: MONO }}>{label}</span>
        <span style={{ fontSize: 11, fontWeight: 700, color: probColor(ratio), fontFamily: MONO }}>
          {count}/{total}
        </span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${ratio * 100}%`, background: probColor(ratio), transition: "width .35s" }} />
      </div>
    </div>
  );
}

function TeamPanel({ title, overall, venue, venueLabel, market, injuries }) {
  const stats = [
    { l: "Zabija", v: overall?.gf, c: C.green },
    { l: "Prima", v: overall?.ga, c: C.red },
    { l: "Ukupno", v: overall?.tot, c: "#3b82f6" },
  ];
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color: C.dim, marginBottom: 8, fontFamily: MONO }}>
        {title}
      </div>
      {injuries?.length > 0 && (
        <div style={{ marginBottom: 8, padding: "6px 8px", borderRadius: 6, background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.15)" }}>
          <div style={{ fontSize: 8, fontWeight: 700, color: C.red, fontFamily: MONO, marginBottom: 4 }}>
            IZOSTANCI ({injuries.length})
          </div>
          {injuries.slice(0, 6).map((inj, i) => (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", gap: 8, padding: "2px 0" }}>
              <span style={{ fontSize: 9, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{inj.name}</span>
              <span style={{ fontSize: 8, color: C.faint, fontFamily: MONO, flexShrink: 0 }}>{inj.reason}</span>
            </div>
          ))}
        </div>
      )}
      <FormBar label={`Zadnjih ${overall?.n ?? 0}`} count={formCount(overall, market)} total={overall?.n} />
      <FormBar label={`Zadnjih ${venue?.n ?? 0} ${venueLabel}`} count={formCount(venue, market)} total={venue?.n} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4, marginTop: 8 }}>
        {stats.map((s, i) => (
          <div key={i} style={{ background: "rgba(255,255,255,0.02)", borderRadius: 5, padding: "5px 4px", textAlign: "center" }}>
            <div style={{ fontSize: 8, color: C.faint, fontFamily: MONO }}>{s.l}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: s.c }}>
              {s.v === null || s.v === undefined ? "–" : s.v.toFixed(2)}
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4, marginTop: 4 }}>
        <div style={{ background: "rgba(255,255,255,0.02)", borderRadius: 5, padding: 5, textAlign: "center" }}>
          <div style={{ fontSize: 8, color: C.faint, fontFamily: MONO }}>Bez primljenog</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: C.green }}>{overall?.cs ?? "–"}/{overall?.n ?? 0}</div>
        </div>
        <div style={{ background: "rgba(255,255,255,0.02)", borderRadius: 5, padding: 5, textAlign: "center" }}>
          <div style={{ fontSize: 8, color: C.faint, fontFamily: MONO }}>Bez zabijenog</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: C.red }}>{overall?.fts ?? "–"}/{overall?.n ?? 0}</div>
        </div>
      </div>
    </div>
  );
}

function MatchCard({ match, market }) {
  const [open, setOpen] = useState(false);
  const isMobile = useIsMobile();

  const prediction = match.predictions?.[market];
  const p = prediction?.p ?? null;
  // Utakmica je odigrana ako ima upisan rezultat. Tada postotak nije prognoza
  // nego procjena koju bi model dao PRIJE te utakmice, pa se uz nju prikazuje
  // i je li se ostvarila.
  const outcome = actualOutcome(match, market);
  const played = outcome !== null;
  const an = match.analysis;

  const kickoff = new Date(match.kickoff);
  const timeStr = kickoff.toLocaleTimeString("hr-HR", { hour: "2-digit", minute: "2-digit" });
  const flag = FLAGS[match.league.country] || "🏳️";
  const accent = probColor(p);

  return (
    <div
      onClick={() => setOpen((v) => !v)}
      style={{
        background: C.panel,
        border: `1px solid ${p >= 0.7 ? "rgba(16,185,129,0.2)" : C.line}`,
        borderRadius: 12,
        padding: isMobile ? "12px" : "14px 18px",
        marginBottom: 8,
        cursor: "pointer",
        transition: "border-color .2s",
      }}
    >
      {/* Zaglavlje */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontSize: isMobile ? 9 : 10, color: C.faint, fontFamily: MONO }}>
          {flag} {match.league.name} · {timeStr}
          {played && (
            <span style={{ marginLeft: 6, padding: "1px 6px", borderRadius: 6, background: "rgba(148,163,184,0.14)", color: C.dim, fontSize: 8 }}>
              odigrano
            </span>
          )}
          {match.category && match.category !== "senior" && (
            <span style={{ marginLeft: 6, padding: "1px 6px", borderRadius: 6, background: "rgba(148,163,184,0.12)", color: C.dim, fontSize: 8 }}>
              {{ youth: "juniori", women: "žene", reserve: "pričuve" }[match.category]}
            </span>
          )}
        </span>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {played && (
            <span
              title={outcome ? "Ishod se ostvario" : "Ishod se nije ostvario"}
              style={{
                fontSize: 9, fontWeight: 700, fontFamily: MONO, padding: "2px 7px", borderRadius: 8,
                background: outcome ? "rgba(16,185,129,0.12)" : "rgba(239,68,68,0.12)",
                color: outcome ? C.green : C.red,
              }}
            >
              {outcome ? "DA" : "NE"}
            </span>
          )}
          <span
            title={p === null ? "Premalo odigranih utakmica za pouzdanu procjenu" : undefined}
            style={{
              fontSize: 15, fontWeight: 800, fontFamily: MONO, color: accent,
              opacity: played ? 0.55 : 1,
            }}
          >
            {p === null ? "n/d" : pct(p)}
          </span>
        </div>
      </div>

      {/* Momčadi */}
      <div style={{ display: "flex", alignItems: "center", gap: isMobile ? 6 : 10 }}>
        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
          {match.home.logo && <img src={match.home.logo} alt="" style={{ width: 20, height: 20, flexShrink: 0 }} />}
          <span style={{ fontSize: isMobile ? 12 : 14, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {match.home.name}
          </span>
        </div>
        <span style={{ fontSize: 10, color: "#475569", fontFamily: MONO, fontWeight: 700, flexShrink: 0 }}>
          {match.home_goals !== null && match.home_goals !== undefined
            ? `${match.home_goals}-${match.away_goals}`
            : "vs"}
        </span>
        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 6, justifyContent: "flex-end", minWidth: 0 }}>
          <span style={{ fontSize: isMobile ? 12 : 14, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {match.away.name}
          </span>
          {match.away.logo && <img src={match.away.logo} alt="" style={{ width: 20, height: 20, flexShrink: 0 }} />}
        </div>
      </div>

      {/* Sažetak forme */}
      {an && (
        <div style={{ display: "flex", gap: 3, marginTop: 10 }}>
          {[
            { l: "Dom. ukupno", b: an.hO },
            { l: "Dom. kod kuće", b: an.hV },
            { l: "Gost ukupno", b: an.aO },
            { l: "Gost u gostima", b: an.aV },
          ].map((item, i) => {
            const count = formCount(item.b, market);
            const ratio = item.b?.n ? count / item.b.n : null;
            return (
              <div key={i} style={{ flex: 1, textAlign: "center", padding: "6px 2px", borderRadius: 6, background: "rgba(255,255,255,0.02)" }}>
                <div style={{ fontSize: isMobile ? 12 : 14, fontWeight: 800, fontFamily: MONO, color: probColor(ratio) }}>
                  {count ?? "–"}/{item.b?.n ?? 0}
                </div>
                <div style={{ fontSize: 7, color: C.faint, fontFamily: MONO, marginTop: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {item.l}
                </div>
              </div>
            );
          })}
          <div style={{ alignSelf: "center", fontSize: 9, color: "#475569", padding: "0 4px", transform: open ? "rotate(180deg)" : "none", transition: ".2s" }}>▼</div>
        </div>
      )}

      {/* Detalji */}
      {open && an && (
        <div style={{ borderTop: `1px solid ${C.line}`, marginTop: 14, paddingTop: 14 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 6, marginBottom: 14 }}>
            {MARKETS.map((m) => {
              const mp = match.predictions?.[m.key]?.p;
              const active = m.key === market;
              return (
                <div key={m.key} style={{ textAlign: "center", padding: "8px 4px", borderRadius: 7, background: active ? `${m.color}14` : "rgba(255,255,255,0.02)", border: `1px solid ${active ? `${m.color}44` : C.line}` }}>
                  <div style={{ fontSize: 7, color: active ? m.color : C.faint, fontFamily: MONO, marginBottom: 2 }}>{m.label}</div>
                  <div style={{ fontSize: 15, fontWeight: 800, fontFamily: MONO, color: active ? m.color : C.text }}>{pct(mp)}</div>
                </div>
              );
            })}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr", gap: 18 }}>
            <TeamPanel title={`🏠 ${match.home.name}`} overall={an.hO} venue={an.hV} venueLabel="kod kuće" market={market} injuries={match.injuries?.home} />
            <TeamPanel title={`✈️ ${match.away.name}`} overall={an.aO} venue={an.aV} venueLabel="u gostima" market={market} injuries={match.injuries?.away} />
          </div>
          {prediction && (
            <div style={{ marginTop: 12, fontSize: 9, color: C.faint, fontFamily: MONO, textAlign: "right" }}>
              izvor procjene: {prediction.mode === "model" ? `model ${prediction.source}` : "heuristika (model još nije istreniran)"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Glavna komponenta
// ═══════════════════════════════════════════════════════════════

export default function App() {
  const isMobile = useIsMobile();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [market, setMarket] = useState("over25");
  const [minProb, setMinProb] = useState(0);
  const [search, setSearch] = useState("");
  const [day, setDay] = useState(todayLocal);
  const [scope, setScope] = useState("senior");

  const load = useCallback(async (targetDay, scopeValue) => {
    setLoading(true);
    setError("");
    try {
      // Poslužitelj po zadanom već izostavlja juniorska, ženska i pričuvna
      // natjecanja te utakmice bez dovoljno odigranih utakmica za formu.
      const params = new URLSearchParams({ date: targetDay });
      if (scopeValue === "all") {
        params.set("only_senior", "false");
        params.set("only_predicted", "false");
      }
      const res = await fetch(`${API_BASE}/api/matches?${params}`);
      if (!res.ok) throw new Error(`Poslužitelj je vratio ${res.status}`);
      setData(await res.json());
    } catch (err) {
      setError(`Podaci nisu dostupni: ${err.message}`);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Dohvat kreće odmah pri otvaranju stranice — bez prijave i bez međukoraka.
  useEffect(() => {
    load(day, scope);
  }, [day, scope, load]);

  const matches = useMemo(() => {
    if (!data?.matches) return [];
    const needle = search.trim().toLowerCase();
    return data.matches
      .filter((m) => {
        const p = m.predictions?.[market]?.p;
        if (minProb > 0 && (p === null || p === undefined || p < minProb)) return false;
        if (!needle) return true;
        return (
          m.home.name.toLowerCase().includes(needle) ||
          m.away.name.toLowerCase().includes(needle) ||
          m.league.name.toLowerCase().includes(needle)
        );
      })
      .sort((a, b) => (b.predictions?.[market]?.p ?? -1) - (a.predictions?.[market]?.p ?? -1));
  }, [data, market, minProb, search]);

  // Kad je odabrani dan vec odigran, procjene se mogu usporediti sa stvarnim
  // ishodima. To je najizravniji prikaz koliko model vrijedi.
  const scoreboard = useMemo(() => {
    const settled = matches
      .map((m) => ({ o: actualOutcome(m, market), p: m.predictions?.[market]?.p }))
      .filter((x) => x.o !== null && x.p !== null && x.p !== undefined);
    if (!settled.length) return null;
    const hits = settled.filter((x) => (x.p >= 0.5) === x.o).length;
    const avgP = settled.reduce((a, x) => a + x.p, 0) / settled.length;
    const rate = settled.filter((x) => x.o).length / settled.length;
    return { n: settled.length, hits, avgP, rate };
  }, [matches, market]);

  const modelInfo = data?.model?.markets?.[market === "under25" ? "over25" : market === "btts_no" ? "btts" : market];
  const isHeuristic = modelInfo && modelInfo.mode !== "model";

  return (
    <div style={{ minHeight: "100vh", background: `linear-gradient(170deg,${C.bg},#0f172a 45%,#0c1220)`, fontFamily: "'Outfit',sans-serif", color: C.text }}>
      <style>{CSS}</style>

      {/* Zaglavlje */}
      <header style={{ position: "sticky", top: 0, zIndex: 50, padding: isMobile ? "12px" : "14px 24px", borderBottom: `1px solid ${C.line}`, background: "rgba(10,14,26,0.92)", backdropFilter: "blur(14px)", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 20 }}>⚽</span>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700 }}>Predikcije utakmica</div>
            <div style={{ fontSize: 9, color: C.faint, fontFamily: MONO }}>
              {data ? `${data.count} utakmica · ${data.date}` : "učitavanje…"}
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="date"
            value={day}
            onChange={(e) => setDay(e.target.value)}
            style={{ background: "rgba(255,255,255,0.04)", border: `1px solid ${C.line}`, borderRadius: 7, padding: "6px 9px", color: C.text, fontSize: 11, colorScheme: "dark" }}
          />
          <button
            onClick={() => load(day, scope)}
            disabled={loading}
            style={{ background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.25)", borderRadius: 7, padding: "6px 12px", color: C.green, fontSize: 11, fontWeight: 600, cursor: loading ? "wait" : "pointer" }}
          >
            {loading ? "…" : "Osvježi"}
          </button>
        </div>
      </header>

      <main style={{ maxWidth: 1100, margin: "0 auto", padding: isMobile ? "14px 10px" : "22px 20px" }}>
        {/* Odabir tržišta */}
        <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr 1fr" : "repeat(4,1fr)", gap: 6, marginBottom: 12 }}>
          {MARKETS.map((m) => (
            <button
              key={m.key}
              onClick={() => setMarket(m.key)}
              style={{ padding: "10px", borderRadius: 8, cursor: "pointer", fontSize: 12, fontWeight: 600, background: market === m.key ? `${m.color}18` : "rgba(255,255,255,0.02)", border: `1px solid ${market === m.key ? `${m.color}55` : C.line}`, color: market === m.key ? m.color : C.faint }}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Filteri */}
        <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Traži momčad ili ligu…"
            style={{ flex: "1 1 200px", background: "rgba(255,255,255,0.04)", border: `1px solid ${C.line}`, borderRadius: 8, padding: "9px 12px", color: C.text, fontSize: 12 }}
          />
          <select
            value={scope}
            onChange={(e) => setScope(e.target.value)}
            title="Koja se natjecanja prikazuju"
            style={{ background: "rgba(255,255,255,0.04)", border: `1px solid ${C.line}`, borderRadius: 8, padding: "9px 12px", color: C.text, fontSize: 12 }}
          >
            <option value="senior">Seniorska natjecanja</option>
            <option value="all">Sve, uključujući juniorske</option>
          </select>
          <select
            value={minProb}
            onChange={(e) => setMinProb(Number(e.target.value))}
            style={{ background: "rgba(255,255,255,0.04)", border: `1px solid ${C.line}`, borderRadius: 8, padding: "9px 12px", color: C.text, fontSize: 12 }}
          >
            <option value={0}>Sve vjerojatnosti</option>
            <option value={0.5}>Iznad 50 %</option>
            <option value={0.6}>Iznad 60 %</option>
            <option value={0.7}>Iznad 70 %</option>
            <option value={0.8}>Iznad 80 %</option>
          </select>
        </div>

        {/* Usporedba sa stvarnim ishodima za odigrane dane */}
        {scoreboard && !loading && (
          <div style={{ marginBottom: 14, padding: "10px 14px", borderRadius: 8, background: "rgba(59,130,246,0.06)", border: "1px solid rgba(59,130,246,0.18)", fontSize: 11, color: "#93c5fd", lineHeight: 1.7 }}>
            <b>Odigrano {scoreboard.n} utakmica ovog dana.</b>{" "}
            Procjena je pogodila smjer u {scoreboard.hits} ({Math.round((scoreboard.hits / scoreboard.n) * 100)} %).
            Prosječna procijenjena vjerojatnost {Math.round(scoreboard.avgP * 100)} %, stvarno ostvareno {Math.round(scoreboard.rate * 100)} %.
            <div style={{ fontSize: 10, color: C.faint, marginTop: 4 }}>
              Postotci su izračunati isključivo iz podataka poznatih prije početka svake utakmice.
            </div>
          </div>
        )}

        {/* Način rada modela */}
        {isHeuristic && !loading && (
          <div style={{ marginBottom: 14, padding: "10px 14px", borderRadius: 8, background: "rgba(245,158,11,0.07)", border: "1px solid rgba(245,158,11,0.2)", fontSize: 11, color: C.amber, lineHeight: 1.6 }}>
            Model još nije istreniran — prikazane vrijednosti su prosjek stopa iz forme, ne izlaz modela.
            Pokreni <code style={{ fontFamily: MONO }}>python -m scripts.backfill</code> pa{" "}
            <code style={{ fontFamily: MONO }}>python -m app.ml.train</code>.
          </div>
        )}

        {/* Sadržaj */}
        {error && (
          <div style={{ padding: 14, borderRadius: 8, background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: C.red, fontSize: 12 }}>
            {error}
          </div>
        )}

        {loading && (
          <div style={{ textAlign: "center", padding: 60, color: C.faint, animation: "pulse 1.6s infinite" }}>
            <div style={{ fontSize: 32, marginBottom: 10 }}>⚽</div>
            <div style={{ fontSize: 13 }}>Učitavam predikcije…</div>
          </div>
        )}

        {!loading && !error && matches.length === 0 && (
          <div style={{ textAlign: "center", padding: 60, color: "#475569" }}>
            <div style={{ fontSize: 32, marginBottom: 10 }}>📭</div>
            <div style={{ fontSize: 13 }}>
              {data?.count ? "Nema utakmica za zadane filtere" : "Za odabrani dan nema podataka u bazi"}
            </div>
          </div>
        )}

        {!loading &&
          matches.map((m, i) => (
            <div key={m.id} style={{ animation: `fadeIn .25s ease ${Math.min(i * 0.015, 0.3)}s both` }}>
              <MatchCard match={m} market={market} />
            </div>
          ))}

        {data && (
          <footer style={{ marginTop: 24, paddingTop: 14, borderTop: `1px solid ${C.line}`, fontSize: 9, color: "#475569", fontFamily: MONO, textAlign: "center", lineHeight: 1.8 }}>
            Procjene su statističke i ne jamče ishod. Podaci: API-Football.
            <br />
            Aktivni model: {data.model?.active_model ?? "—"}
          </footer>
        )}
      </main>
    </div>
  );
}
