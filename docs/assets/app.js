// Renders docs/data/results.json (written by `python -m eval.export_site`).
// When the page is served by nba_rag.server, /api/health answers and the Ask box turns on.
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const OUTCOME_LABEL = { correct: "correct", wrong: "wrong", false_abstain: "abstained (shouldn't have)", hallucination: "answered (shouldn't have)" };
const KIND_LABEL = {
  player_game_stat: "Player stat in a game", game_result: "Game result", team_leader: "Team leader in a game",
  opener_points: "Season-opener points", month_total: "Monthly totals", extreme_margin: "Largest win / loss",
  starter_in_losses: "Top starter in blowout losses", threshold_games_against: "Most N+ point games vs a team",
  after_break: "Totals after the All-Star break", single_game_high: "Single-game high vs a team",
  go_ahead: "Go-ahead basket (recaps)", comeback: "Largest comeback (recaps)", why_missed: "Why a player missed (notes)",
};
const TABLE_LABEL = { games: "game", box_scores: "box score", recaps: "recap", availability_notes: "note" };

const SPLIT = {
  test: { title: "Test set", note: "written after all fixes, run once, never tuned on" },
  heldout: { title: "Held-out set", note: "run blind once; its misses then drove fixes" },
  dev: { title: "Dev set", note: "developed against" },
};

let DATA = null;
const state = { split: "all", tier: "all", outcome: "all", q: "" };

function fmtFields(obj, fields) {
  return Object.keys(fields).map((k) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${esc(obj?.[k])}</dd>`).join("");
}

function renderMetrics() {
  const sc = DATA.scorecards, s = DATA.dataset;
  const t = sc.test?.overall;
  const all = Object.values(sc).map((x) => x.overall);
  const items = [
    [pct(t?.accuracy), "accuracy on the untouched test set"],
    [`${all.reduce((n, b) => n + b.hallucination, 0)}`, "answers to unanswerable questions"],
    [`${all.reduce((n, b) => n + b.n, 0)}`, "evaluated questions"],
    [s.games.toLocaleString(), "games indexed"],
    [(s.recaps + s.notes).toLocaleString(), "recaps + notes embedded"],
  ];
  $("#metrics").innerHTML = items.map(([b, l]) => `<div class="metric"><b>${esc(b)}</b><span>${esc(l)}</span></div>`).join("");
  $("#generated").textContent = ` · Results generated ${DATA.generated} with ${DATA.models.llm} + ${DATA.models.embed}.`;
}

function scoreTable(summary) {
  const rows = [["Overall", summary.overall], ...["core", "extended", "stretch"].filter((t) => summary.by_tier[t]).map((t) => [t, summary.by_tier[t]])];
  return `<div class="table-scroll"><table><thead><tr><th>Slice</th><th class="num">n</th><th class="num">Acc.</th>
    <th class="num">Wrong</th><th class="num">Bad abstain</th><th class="num">Hallucin.</th><th class="num">Evidence R</th></tr></thead><tbody>
    ${rows.map(([n, b]) => `<tr><td>${esc(n)}</td><td class="num">${b.n}</td><td class="num"><b>${pct(b.accuracy)}</b></td>
      <td class="num">${b.wrong}</td><td class="num">${b.false_abstain}</td><td class="num">${b.hallucination}</td>
      <td class="num">${pct(b.evidence_recall)}</td></tr>`).join("")}</tbody></table></div>`;
}

function renderResults() {
  const sc = DATA.scorecards;
  $("#results-lede").textContent = `${DATA.questions.length} questions in three sets, each with its own players, teams, dates and phrasings, across ${Object.keys(KIND_LABEL).length} kinds and three tiers. ` +
    "Core questions are lookups with one reasoning step, extended ones filter and aggregate, and stretch facts exist only in recap or note text. " +
    "About one in nine questions can't be answered from the data, and the right response to those is to abstain.";
  $("#scorecards").innerHTML = ["test", "heldout", "dev"].filter((k) => sc[k]).map((k) => `
    <div class="card scorecard"><header><div><h3>${SPLIT[k].title}</h3>
      <div class="note">${SPLIT[k].note}</div></div>
      <div class="big">${pct(sc[k].overall.accuracy)}</div></header>${scoreTable(sc[k])}</div>`).join("");
  const kinds = Object.keys(KIND_LABEL).filter((k) => ["dev", "heldout", "test"].some((s) => sc[s]?.by_kind[k]));
  const bar = (b, cls) => b ? `<div class="bar ${cls}"><i style="width:${Math.round(b.accuracy * 100)}%"></i><span>${b.correct}/${b.n}</span></div>` : `<div></div>`;
  $("#kinds").innerHTML = `<div class="kind-head"><span class="label">Kind</span><span>Test</span><span>Held-out</span><span>Dev</span></div>` +
    kinds.map((k) => `<div class="kind-row"><div class="label">${esc(KIND_LABEL[k])}</div>${bar(sc.test?.by_kind[k], "test")}${bar(sc.heldout?.by_kind[k], "heldout")}${bar(sc.dev?.by_kind[k], "dev")}</div>`).join("");
}

function chipGroup(el, key, options) {
  el.innerHTML = options.map(([v, l]) => `<button class="chip" data-v="${v}" aria-pressed="${state[key] === v}">${esc(l)}</button>`).join("");
  el.onclick = (e) => {
    const b = e.target.closest(".chip");
    if (!b) return;
    state[key] = b.dataset.v;
    el.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", c.dataset.v === state[key]));
    renderCards();
  };
}

function evidenceList(ev) {
  if (!ev?.length) return `<p class="meta">No evidence cited.</p>`;
  return `<ul class="evidence">${ev.map((e) => `<li><span class="src">${esc(TABLE_LABEL[e.table] || e.table)}</span><span>${esc(e.text || JSON.stringify(e))}</span></li>`).join("")}</ul>`;
}

function card(q) {
  const ans = q.answer, gold = q.gold, fields = q.fields;
  const short = ans.answerable ? Object.keys(fields).map((k) => `${k.replace(/_/g, " ")}: ${ans[k]}`).join(" · ") : "abstained";
  const t = q.trace || {};
  return `<details class="card qcard"><summary>
    <div class="qhead"><span class="badge b-${q.outcome}">${esc(OUTCOME_LABEL[q.outcome])}</span>
      <span class="tag">${esc(q.split)} · ${esc(q.tier)} · ${esc(KIND_LABEL[q.kind] || q.kind)}</span></div>
    <div class="qtext">${esc(q.question)}</div><div class="qanswer">→ ${esc(short)}</div></summary>
    <div class="qbody">
      <div class="compare">
        <div><h4>Pipeline answer</h4>${ans.answerable ? `<dl class="kv">${fmtFields(ans, fields)}</dl>` : "<p>Abstained</p>"}</div>
        <div><h4>Gold</h4>${gold.answerable ? `<dl class="kv">${fmtFields(gold, fields)}</dl>` : "<p>Not answerable from the data</p>"}</div>
      </div>
      ${t.abstain_reason ? `<div class="reason">Abstention reason: ${esc(t.abstain_reason)}</div>` : ""}
      <div><h4>Evidence cited</h4>${evidenceList(q.evidence)}</div>
      ${t.context ? `<div><h4>Context the model saw</h4><pre class="context">${esc(t.context)}</pre></div>` : ""}
      <p class="meta">${t.llm_calls ?? 0} LLM call(s) · ${t.seconds ?? "?"}s${t.plan ? ` · plan: stat=${esc(t.plan.stat)}, result=${esc(t.plan.result)}, phase=${esc(t.plan.season_phase)}${t.plan.min_value != null ? `, min=${t.plan.min_value}` : ""}${t.plan.min_margin != null ? `, margin≥${t.plan.min_margin}` : ""}` : ""}${t.retrieval?.length ? ` · ${esc(t.retrieval.join("; "))}` : ""}</p>
    </div></details>`;
}

function renderCards() {
  const q = state.q.trim().toLowerCase();
  const shown = DATA.questions.filter((x) =>
    (state.split === "all" || x.split === state.split) && (state.tier === "all" || x.tier === state.tier) &&
    (state.outcome === "all" || (state.outcome === "miss" ? x.outcome !== "correct" : x.outcome === state.outcome)) &&
    (!q || x.question.toLowerCase().includes(q)));
  $("#count").textContent = `${shown.length} of ${DATA.questions.length} questions`;
  $("#cards").innerHTML = shown.map(card).join("");
}

function renderExplorer() {
  chipGroup($("#f-split"), "split", [["all", "All sets"], ["test", "Test"], ["heldout", "Held-out"], ["dev", "Dev"]]);
  chipGroup($("#f-tier"), "tier", [["all", "All tiers"], ["core", "Core"], ["extended", "Extended"], ["stretch", "Stretch"]]);
  chipGroup($("#f-outcome"), "outcome", [["all", "All"], ["correct", "Correct"], ["miss", "Misses"]]);
  $("#f-search").oninput = (e) => { state.q = e.target.value; renderCards(); };
  renderCards();
}

async function enableLive() {
  try {
    const r = await fetch("api/health", { cache: "no-store" });
    if (!r.ok) return;
    const h = await r.json();
    document.querySelectorAll(".live-only").forEach((el) => (el.hidden = false));
    $("#ask-kind").innerHTML += Object.entries(h.shapes).map(([k, d]) => `<option value="${k}">${esc(k)}: ${esc(d)}</option>`).join("");
    $("#ask-form").onsubmit = async (e) => {
      e.preventDefault();
      const question = $("#ask-input").value.trim();
      if (!question) return;
      const btn = e.target.querySelector("button");
      btn.disabled = true; btn.textContent = "Thinking…";
      try {
        const res = await fetch("api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, kind: $("#ask-kind").value || null }) }).then((x) => x.json());
        const t = res.trace || {};
        $("#ask-out").innerHTML = `<div class="card"><div class="qbody">
          <div class="qhead"><span class="badge ${res.answerable ? "b-correct" : "b-false_abstain"}">${res.answerable ? "answered" : "abstained"}</span>
          <span class="tag">kind: ${esc(res.kind)}</span></div>
          <div class="qtext">${esc(res.message)}</div>
          <div><h4>Evidence cited</h4>${evidenceList(res.evidence)}</div>
          ${t.context ? `<div><h4>Context the model saw</h4><pre class="context">${esc(t.context)}</pre></div>` : ""}
          <p class="meta">${t.llm_calls ?? 0} LLM call(s)</p></div></div>`;
      } finally { btn.disabled = false; btn.textContent = "Ask"; }
    };
  } catch { /* static hosting: no API */ }
}

fetch("data/results.json").then((r) => r.json()).then((d) => {
  DATA = d;
  if (d.repo_url) $("#repo-link").href = d.repo_url;
  renderMetrics(); renderResults(); renderExplorer();
}).catch(() => { $("#metrics").innerHTML = `<p class="meta">Run <code>python -m eval.export_site</code> to generate data/results.json.</p>`; });
enableLive();
