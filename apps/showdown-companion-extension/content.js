(function() {
  const PANEL_ID = "pokestrategist-panel";
  const STATE = {
    collapsed: false,
    lastSignature: "",
    lastRefreshAt: 0,
    config: null,
    drag: { active: false, startX: 0, startY: 0, panelX: 0, panelY: 0 }
  };

  /* ---- DOM parsing (unchanged core) ---- */
  function textOf(n) { return n?.textContent?.replace(/\s+/g, " ").trim() || ""; }
  function pickFirstText(sels) { for (const s of sels) { const v = textOf(document.querySelector(s)); if (v) return v; } return ""; }

  function parseSideStatbars() {
    const l = document.querySelector(".lstatbar,.trainer-near .statbar,.leftbar .statbar");
    const r = document.querySelector(".rstatbar,.trainer-far .statbar,.rightbar .statbar");
    return {
      self:  { name: textOf(l?.querySelector("strong,.pokemonname")) || textOf(l), hp: textOf(l?.querySelector(".hptext,.hp")) },
      opponent: { name: textOf(r?.querySelector("strong,.pokemonname")) || textOf(r), hp: textOf(r?.querySelector(".hptext,.hp")) }
    };
  }

  function parseButtons(sels) {
    const res = []; const seen = new Set();
    for (const sel of sels) {
      for (const b of document.querySelectorAll(sel)) {
        if (!(b instanceof HTMLButtonElement)) continue;
        const lb = textOf(b); if (!lb || b.disabled || seen.has(lb)) continue;
        seen.add(lb); res.push({ label: lb, disabled: b.disabled, tooltip: b.getAttribute("data-tooltip") || b.title || "" });
      }
    }
    return res;
  }

  function parseBattleLog() {
    const root = document.querySelector(".battle-log .inner,.battle-log,.battle-history");
    if (!root) return [];
    const vals = []; const seen = new Set();
    for (const n of root.querySelectorAll("div,p,li")) { const v = textOf(n); if (!v || seen.has(v)) continue; seen.add(v); vals.push(v); if (vals.length >= 20) break; }
    return vals.slice(-8);
  }

  function inferTurn() {
    const e = pickFirstText([".turn",".turnstatus",".battle .turn"]);
    if (e) return e;
    return parseBattleLog().find(l => /^turn\s+\d+/i.test(l)) || "Turn unknown";
  }

  function roomTitle() { return (document.title.replace(/\s*-\s*Pok[eé]mon Showdown!?/i,"").trim()) || "Pokemon Showdown"; }

  function buildSnapshot() {
    const sides = parseSideStatbars();
    const moves = parseButtons(["button[name='chooseMove']",".movemenu button",".choosemove button"]);
    const switches = parseButtons(["button[name='chooseSwitch']",".switchmenu button",".chooseswitch button","button[name='chooseTeamPreview']"]);
    const logs = parseBattleLog();
    return {
      source: "showdown-dom", pageTitle: roomTitle(), turn: inferTurn(),
      forcedSwitch: switches.length > 0 && moves.length === 0,
      self: sides.self, opponent: sides.opponent,
      legalMoves: moves, legalSwitches: switches, recentLog: logs,
      observedAt: new Date().toISOString(), url: location.href
    };
  }

  function fallbackSuggestions(snapshot) {
    const opts = [];
    const labels = snapshot.forcedSwitch ? snapshot.legalSwitches : snapshot.legalMoves;
    for (const e of labels.slice(0,3)) opts.push({ label: e.label, confidence: null, reason: "本地模型未连接，显示可用动作占位。" });
    return opts;
  }

  /* ---- Remote fetch ---- */
  async function getConfig() { const r = await chrome.runtime.sendMessage({type:"pokestrategist:get-config"}); return r?.result || null; }
  async function fetchRemote(snapshot) {
    const r = await chrome.runtime.sendMessage({type:"pokestrategist:fetch-suggestion",payload:{snapshot}});
    return r?.result || null;
  }

  /* ---- Panel: build, drag, render ---- */
  function ensurePanel() {
    let p = document.getElementById(PANEL_ID);
    if (p) { STATE.panel = p; return p; }
    p = document.createElement("aside"); p.id = PANEL_ID;
    p.innerHTML = '<div class="pokestrategist-drag-handle" id="pokestrategist-drag"><span class="pokestrategist-title">pokestrategist</span><button class="pokestrategist-toggle" id="pokestrategist-toggle">—</button></div><div class="pokestrategist-status-row"><span class="pokestrategist-status-dot offline" id="pokestrategist-dot"></span><span class="pokestrategist-status-text" id="pokestrategist-status">等待对战开始…</span></div><div class="pokestrategist-body" id="pokestrategist-body"></div>';
    document.body.appendChild(p);

    // Drag
    const drag = p.querySelector("#pokestrategist-drag");
    drag.addEventListener("mousedown", e => {
      STATE.drag.active = true; STATE.drag.startX = e.clientX; STATE.drag.startY = e.clientY;
      STATE.drag.panelX = p.offsetLeft; STATE.drag.panelY = p.offsetTop;
      p.classList.add("dragging");
    });
    document.addEventListener("mousemove", e => {
      if (!STATE.drag.active) return;
      p.style.left = (STATE.drag.panelX + e.clientX - STATE.drag.startX) + "px";
      p.style.top = (STATE.drag.panelY + e.clientY - STATE.drag.startY) + "px";
      p.style.right = "auto";
    });
    document.addEventListener("mouseup", () => { STATE.drag.active = false; p.classList.remove("dragging"); });

    // Toggle
    p.querySelector("#pokestrategist-toggle").addEventListener("click", () => {
      STATE.collapsed = !STATE.collapsed; p.classList.toggle("pokestrategist-collapsed", STATE.collapsed);
      p.querySelector("#pokestrategist-toggle").textContent = STATE.collapsed ? "+" : "—";
    });

    STATE.panel = p; return p;
  }

  function fmtPct(v) { return (typeof v==="number" && !Number.isNaN(v)) ? Math.round(v*100)+"%" : "—"; }
  function esc(v) { return String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }

  function renderReasonChain(chain) {
    if (!chain) return "";
    const rows = [];
    if (chain.is_lead) {
      // Lead phase: show team preview analysis
      rows.push('<span class="pokestrategist-chain-icon">🧭</span><span class="pokestrategist-chain-text">队伍预览分析 · 计划 <strong>'+esc(chain.plan)+'</strong></span>');
      if (chain.lead_hint) {
        rows.push('<span class="pokestrategist-chain-icon">📊</span><span class="pokestrategist-chain-text">环境首发率: <strong>'+esc(chain.lead_hint)+'</strong></span>');
      }
      if (chain.opp_team && chain.opp_team.length) {
        rows.push('<span class="pokestrategist-chain-icon">👁</span><span class="pokestrategist-chain-text">对手阵容: '+chain.opp_team.map(s=>'<strong>'+esc(s)+'</strong>').join(" · ")+'</span>');
      }
      rows.push('<span class="pokestrategist-chain-icon">🎯</span><span class="pokestrategist-chain-text">首发选择 · 模型基于隐藏配置先验和队伍匹配分析排序</span>');
    } else {
      rows.push('<span class="pokestrategist-chain-icon">🧭</span><span class="pokestrategist-chain-text">线路 <strong>'+esc(chain.line)+'</strong> · 计划 <strong>'+esc(chain.plan)+'</strong> · 阶段 <strong>'+esc(chain.phase)+'</strong></span>');
      if (chain.response && chain.response.length) {
        const rs = chain.response.map(r => '<strong>'+esc(r.label)+'</strong> <span class="up">'+fmtPct(r.prob)+'</span>').join(" / ");
        rows.push('<span class="pokestrategist-chain-icon">🔄</span><span class="pokestrategist-chain-text">对手回应: '+rs+'</span>');
      }
      if (chain.consequence && chain.consequence.length) {
        rows.push('<span class="pokestrategist-chain-icon">📊</span><span class="pokestrategist-chain-text">后果: '+chain.consequence.map(c=>esc(c)).join(" · ")+'</span>');
      }
      rows.push('<span class="pokestrategist-chain-icon">🎯</span><span class="pokestrategist-chain-text">家族: <strong>'+esc(chain.family||"unknown")+'</strong>'+ (chain.tera ? " (太晶)" : "") +'</span>');
    }
    return rows.map(r => '<div class="pokestrategist-chain-row">'+r+'</div>').join("");
  }

  function render(snapshot, result) {
    const panel = ensurePanel();
    const statusEl = panel.querySelector("#pokestrategist-status");
    const dot = panel.querySelector("#pokestrategist-dot");
    const body = panel.querySelector("#pokestrategist-body");
    if (!statusEl || !body) return;

    const suggestions = result?.ok ? (result.suggestions || []) : fallbackSuggestions(snapshot);
    const online = !!(result?.ok);
    dot.className = "pokestrategist-status-dot " + (online ? "online" : "offline");
    const isLead = !!(result?.metadata?.is_lead_phase);
    const statusLabel = isLead ? "🔍 首发分析 · 队伍预览" : (online ? "模型已连接 · "+esc(snapshot.turn) : "离线模式 · "+esc(snapshot.turn));
    statusEl.textContent = statusLabel;

    // Suggestion cards
    const ranks = ["❶","❷","❸"];
    const cards = suggestions.slice(0,3).map((item,i) => {
      const chain = item.reason_chain;
      return '<div class="pokestrategist-suggestion-card">'+
        '<div class="pokestrategist-suggestion-header">'+
          '<span class="pokestrategist-suggestion-rank">'+ranks[i]+'</span>'+
          '<span class="pokestrategist-suggestion-label">'+esc(item.label||"?")+'</span>'+
          '<span class="pokestrategist-confidence">'+fmtPct(item.confidence)+'</span>'+
        '</div>'+
        '<div class="pokestrategist-reason-chain">'+renderReasonChain(chain)+'</div>'+
      '</div>';
    }).join("");

    const moves = snapshot.legalMoves.map(m => '<span class="pokestrategist-pill">'+esc(m.label)+'</span>').join("");
    const switches = snapshot.legalSwitches.map(s => '<span class="pokestrategist-pill">'+esc(s.label)+'</span>').join("");

    body.innerHTML =
      '<div class="pokestrategist-suggestions">'+cards+'</div>'+
      '<div class="pokestrategist-section-title">对局快照</div>'+
      '<dl class="pokestrategist-snapshot">'+
        '<dt>我方</dt><dd>'+esc(snapshot.self.name||"?")+' '+esc(snapshot.self.hp||"")+'</dd>'+
        '<dt>对手</dt><dd>'+esc(snapshot.opponent.name||"?")+' '+esc(snapshot.opponent.hp||"")+'</dd>'+
        '<dt>回合</dt><dd>'+esc(snapshot.turn)+'</dd>'+
        '<dt>模式</dt><dd>'+(snapshot.forcedSwitch?"强制换人":"行动选择")+'</dd>'+
      '</dl>'+
      '<div class="pokestrategist-section-title">可用招式</div>'+
      '<div class="pokestrategist-legal">'+(moves||'<span class="pokestrategist-pill">无</span>')+'</div>'+
      '<div class="pokestrategist-section-title">可用换人</div>'+
      '<div class="pokestrategist-legal">'+(switches||'<span class="pokestrategist-pill">无</span>')+'</div>';
  }

  /* ---- Refresh loop ---- */
  async function refresh() {
    STATE.config = STATE.config || (await getConfig());
    if (!STATE.config?.enabled) return;
    const snap = buildSnapshot();
    const sig = JSON.stringify({turn:snap.turn,self:snap.self,opponent:snap.opponent,moves:snap.legalMoves.map(m=>m.label),switches:snap.legalSwitches.map(s=>s.label)});
    const now = Date.now();
    if (sig === STATE.lastSignature && now - STATE.lastRefreshAt < (STATE.config.refreshMs||1500)) return;
    STATE.lastSignature = sig; STATE.lastRefreshAt = now;
    const remote = await fetchRemote(snap);
    render(snap, remote);
  }

  function start() {
    ensurePanel();
    refresh().catch(console.error);
    new MutationObserver(() => refresh().catch(console.error)).observe(document.documentElement, {subtree:true,childList:true,characterData:true});
    setInterval(() => refresh().catch(console.error), 2000);
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start, {once:true}); }
  else { start(); }
})();
