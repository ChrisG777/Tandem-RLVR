"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  index: null,
  filtered: [],
  pid: null,
  problem: null,
  audit: null,
  machine: null,
  elegance: null,
  dirty: false,
  raw: false,
  shown: 80,
  panes: [],
  loading: 0,
};
const labels = new Map();
function el(tag, text = "", cls = "") {
  const n = document.createElement(tag);
  n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function error(e) {
  $("error").textContent = e.message || String(e);
  $("error").hidden = false;
  $("save-state").textContent = "Action failed — see message";
}
async function api(path, body) {
  const r = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const value = await r.json();
  if (!r.ok) throw Error(value.error || r.statusText);
  return value;
}
function math(n, text) {
  // Source layout commands have no meaning in a browser; raw view stays exact.
  n.textContent = state.raw ? text : text.replace(/\\noindent\b/g, "");
  if (!state.raw && window.renderMathInElement)
    renderMathInElement(n, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
      trust: false,
      strict: false,
    });
  n.classList.toggle("raw", state.raw);
}
function option(value, text) {
  const o = el("option", text);
  o.value = value;
  return o;
}
function markDirty() {
  state.dirty = true;
  $("save-state").textContent = "Unsaved changes";
  $("save-state").style.color = "#916623";
}
function effectiveTags(p = state.problem, a = state.audit) {
  const assessed = new Map(
    (p?.id === state.problem?.id
      ? state.machine?.route_assessments || []
      : []
    ).map((r) => [r.route_id, r.methods]),
  );
  return [
    ...new Set([
      ...p.routes.flatMap(
        (r) => a.methods?.[r.id] ?? assessed.get(r.id) ?? r.methods,
      ),
      ...(a.additional_methods || []),
    ]),
  ];
}
function splitFor(tags, splits = state.index.settings.splits) {
  const values = tags.map((m) => splits[m]);
  return values.includes("test")
    ? "test"
    : values.includes("validation")
      ? "validation"
      : tags.length
        ? "train"
        : "unassigned";
}
function readForm() {
  if (!state.audit) return;
  for (const k of ["elegant", "elementary", "capability", "notes"])
    state.audit[k] = $(k).value;
  for (const k of [
    "correct_elegant",
    "correct_elementary",
    "distinct",
    "methods_reviewed",
  ])
    state.audit[k] = $(k).checked;
  state.audit.answer = $("answer-correction").value;
  state.audit.status = $("verdict").value;
}
function blockers() {
  if (!state.problem) return [];
  const p = state.problem,
    a = state.audit,
    b = [];
  if (a.status !== "accepted") b.push("Set verdict to Accept pair");
  if (!a.elegant || !a.elementary || a.elegant === a.elementary)
    b.push("Assign two different routes");
  if (!a.correct_elegant || !a.correct_elementary)
    b.push("Confirm both arguments are valid");
  if (!a.distinct) b.push("Confirm distinct methods");
  if (!a.methods_reviewed) b.push("Review every route’s method labels");
  if (!a.answer.trim() && !p.answer.trim())
    b.push("Supply a standalone answer");
  if (p.routes.some((r) => !(a.methods?.[r.id] ?? r.methods).length))
    b.push("Classify all routes");
  if (
    p.routes.some(
      (r) =>
        [a.elegant, a.elementary].includes(r.id) &&
        r.kind === "summary" &&
        !a.route_text?.[r.id]?.trim(),
    )
  )
    b.push("Replace selected summaries with full derivations");
  if (
    p.flags.some((f) => f.includes("Conflicting reference")) &&
    !a.answer.trim()
  )
    b.push("Resolve conflicting answers");
  return b;
}
function readiness() {
  readForm();
  const b = blockers();
  $("export-readiness").textContent = b.length
    ? "Before export: " + b.join(" · ")
    : "Ready for " +
      splitFor(effectiveTags()) +
      " export. Capability status: " +
      state.audit.capability.replaceAll("_", " ") +
      ".";
  $("export-readiness").classList.toggle("good", !b.length);
}
function applyFilters() {
  const q = $("search").value.toLowerCase(),
    status = $("status-filter").value,
    source = $("source-filter").value,
    split = $("split-filter").value,
    method = $("method-filter").value,
    machine = $("machine-filter").value;
  state.filtered = state.index.problems.filter(
    (p) =>
      (!q ||
        [p.title, p.preview, ...p.methods.map((m) => labels.get(m))]
          .join(" ")
          .toLowerCase()
          .includes(q)) &&
      (!status || (status === "ready" ? p.eligible : p.status === status)) &&
      (!machine ||
        (machine === "ready"
          ? p.machine_eligible
          : p.machine_status === machine)) &&
      (!source || p.sources.includes(source)) &&
      (!split || p.split === split) &&
      (!method || p.methods.includes(method)) &&
      (!$("shortlist-filter").checked || p.shortlist) &&
      (!$("pair-filter").checked || p.full_routes >= 2),
  );
  state.shown = 80;
  renderList();
  $("empty").hidden = !!state.filtered.length;
}
function renderList() {
  const list = $("problem-list");
  list.replaceChildren();
  $("list-count").textContent =
    state.filtered.length.toLocaleString() + " matches";
  for (const p of state.filtered.slice(0, state.shown)) {
    const btn = el(
      "button",
      "",
      "problem-row" + (p.id === state.pid ? " active" : ""),
    );
    btn.append(
      el("strong", p.title),
      el("p", p.preview.replace(/\$|\\[a-z]+/g, "")),
    );
    const meta = el("div", "", "row-meta");
    meta.append(
      el("span", p.route_count + " routes · " + p.split),
      el(
        "span",
        p.status === "pending" ? "○ Unreviewed" : p.status.replace("_", " "),
      ),
    );
    meta.append(
      el(
        "span",
        "AI: " + (p.machine_status || "unreviewed").replaceAll("_", " "),
      ),
    );
    btn.append(meta);
    btn.onclick = () => select(p.id);
    list.append(btn);
  }
  if (state.shown < state.filtered.length) {
    const more = el("button", "Show 80 more", "load-more");
    more.onclick = () => {
      state.shown += 80;
      renderList();
    };
    list.append(more);
  }
  updatePosition();
}
function updatePosition() {
  const i = state.filtered.findIndex((p) => p.id === state.pid);
  $("position").textContent =
    i < 0 ? "Outside filter" : `${i + 1} / ${state.filtered.length}`;
  $("prev").disabled = i <= 0;
  $("next").disabled = i < 0 || i >= state.filtered.length - 1;
}
function updateStats() {
  const reviews = state.index.machine_summary || {};
  $("machine-progress").textContent =
    `AI reviewed ${reviews.reviewed || 0} / ${reviews.total || 819} · ${reviews.accepted || 0} accepted · ${reviews.export_ready || 0} answer-checked. All-path labels: ${state.index.elegance_summary?.complete || 0} / ${state.index.elegance_summary?.expected || 0}.`;
  $("total").textContent = (
    reviews.total || state.index.problems.length
  ).toLocaleString();
  $("reviewed").textContent = (reviews.reviewed || 0).toLocaleString();
  $("ready").textContent = (reviews.export_ready || 0).toLocaleString();
}
async function select(pid) {
  try {
    if (state.dirty) await save();
    const serial = ++state.loading;
    const data = await api("/api/problem/" + pid);
    if (serial !== state.loading) return;
    state.pid = pid;
    state.problem = data.problem;
    state.audit = data.audit;
    state.machine = data.machine_review || null;
    state.elegance = data.elegance_review || null;
    state.audit.methods ??= {};
    state.audit.route_text ??= {};
    state.audit.additional_methods ??= [];
    state.dirty = false;
    state.panes =
      state.problem.suggested_pair.length === 2
        ? [...state.problem.suggested_pair]
        : state.problem.routes.slice(0, 2).map((r) => r.id);
    if (state.machine?.status === "accepted")
      state.panes = [
        state.machine.elegant_route_id,
        state.machine.elementary_route_id,
      ];
    if (state.audit.elegant && state.audit.elementary)
      state.panes = [state.audit.elegant, state.audit.elementary];
    history.replaceState(null, "", "#" + pid);
    $("error").hidden = true;
    renderDetail();
    renderList();
    $("save-state").textContent = state.audit.revision
      ? "Saved review · revision " + state.audit.revision
      : "Unreviewed · nothing saved";
    $("save-state").style.color = "";
  } catch (e) {
    error(e);
  }
}
function renderDetail() {
  const p = state.problem,
    a = state.audit;
  $("detail").hidden = false;
  $("source-label").textContent = [
    ...new Set(p.sources.map((s) => s.dataset)),
  ].join(" / ");
  $("title").textContent = p.title;
  $("badges").replaceChildren();
  const partition = splitFor(effectiveTags());
  $("badges").append(
    el(
      "span",
      partition === "test"
        ? "Test · provisional method holdout"
        : partition + " · provisional",
      "badge " + partition,
    ),
    el("span", p.subject.replaceAll("_", " "), "badge"),
    el("span", "Not model-calibrated", "badge"),
  );
  math($("question"), p.question);
  math(
    $("answer"),
    p.answer || "No standalone answer in source. Supply one during review.",
  );
  $("provenance").replaceChildren();
  for (const s of p.sources) {
    const line = el("div"),
      link = el("a", s.dataset + " · " + s.key);
    link.href = s.url;
    link.target = "_blank";
    link.rel = "noopener";
    line.append(link);
    $("provenance").append(line);
  }
  $("curator-note").textContent = state.machine
    ? "All supplied routes remain available. The comparison panes show two paths at a time; the labeled export retains every path."
    : p.curator_note;
  $("flags").textContent = p.flags.length
    ? "Original screening flags: " + p.flags.join(" · ")
    : "";
  $("route-count").textContent = p.routes.length + " source routes · showing 2";
  renderMachineReview();
  renderPathReview();
  for (const k of ["elegant", "elementary"]) {
    $(k).replaceChildren(
      option("", "Choose a route…"),
      ...p.routes.map((r) => option(r.id, r.title + " · " + r.source)),
    );
    $(k).value = a[k] || "";
  }
  for (const k of [
    "correct_elegant",
    "correct_elementary",
    "distinct",
    "methods_reviewed",
  ])
    $(k).checked = !!a[k];
  $("answer-correction").value = a.answer || "";
  $("capability").value = a.capability || "untested";
  $("notes").value = a.notes || "";
  $("verdict").value = a.status;
  $("labels-count").textContent = "(" + p.routes.length + ")";
  renderRoutes();
  readiness();
}
function renderMachineReview() {
  const panel = $("machine-review"),
    r = state.machine;
  panel.replaceChildren();
  panel.hidden = !r;
  if (!r) return;
  panel.append(el("h3", "AI REVIEW · " + r.status.replaceAll("_", " ")));
  const evidence = el("details");
  evidence.append(
    el("summary", "Findability, methods & all route assessments"),
  );
  panel.append(
    el(
      "p",
      `Answer check: ${r.verification?.execution_status || "unexecuted"}. Junior-model capability remains untested.`,
      "muted",
    ),
  );
  evidence.append(
    el(
      "p",
      "Codex GPT 6.1 Sol review. Human audit stays separate; junior accessibility remains untested.",
      "muted",
    ),
  );
  for (const [label, value] of [
    ["Pair contrast", r.contrast_reason],
    ["Naturalness and findability", r.naturalness_reason],
    [
      "Elegant method",
      r.elegant_method
        ? `${labels.get(r.elegant_method.id) || r.elegant_method.label}: ${r.elegant_method.description}`
        : "",
    ],
    [
      "Elementary method",
      r.elementary_method
        ? `${labels.get(r.elementary_method.id) || r.elementary_method.label}: ${r.elementary_method.description}`
        : "",
    ],
    ["Verification", r.verification?.reasoning],
    ["Checker execution", r.verification?.execution_status],
    [
      "Source corrections",
      r.correction_note || Object.values(r.correction_notes || {}).join(" · "),
    ],
    ["Concerns", (r.concerns || []).join(" · ")],
    ["Coordinator follow-up", r.coordinator_followup?.scope],
    ["Export blockers", (r.export_blockers || []).join(" · ")],
  ]) {
    if (!value) continue;
    const line = el("p");
    line.append(el("strong", label + ": "), el("span", value));
    (label === "Pair contrast" || label === "Export blockers"
      ? panel
      : evidence
    ).append(line);
  }
  for (const a of r.route_assessments || []) {
    const route = state.problem.routes.find((x) => x.id === a.route_id);
    evidence.append(
      el("p", `${route?.title || a.route_id} — ${a.correctness}: ${a.reason}`),
    );
  }
  const detail = el("details");
  detail.append(el("summary", "Checker code, output & review provenance"));
  detail.append(
    el(
      "pre",
      JSON.stringify(
        {
          verification: r.verification,
          provenance: r.provenance,
          coordinator_followup: r.coordinator_followup,
          source_fingerprint: r.source_fingerprint,
        },
        null,
        2,
      ),
    ),
  );
  evidence.append(detail);
  panel.append(evidence);
}
function renderPathReview() {
  const panel = $("path-review");
  panel.replaceChildren(el("h3", "ALL SOLUTION PATHS"));
  const judgment = state.elegance;
  if (!judgment || judgment.error) {
    panel.append(
      el(
        "p",
        judgment?.error || "All-path elegance labeling is pending.",
        "muted",
      ),
    );
    return;
  }
  panel.append(
    el(
      "p",
      "Shortness supports a judgment only when the solution is complete. Several paths may be elegant.",
      "muted",
    ),
  );
  const validIds = new Set(
    (state.machine?.route_assessments || [])
      .filter((r) => r.correctness === "valid")
      .map((r) => r.route_id),
  );
  const validLabels = judgment.route_labels.filter((r) =>
    validIds.has(r.route_id),
  );
  const contrast =
    validLabels.some((r) => r.elegant) && validLabels.some((r) => !r.elegant);
  panel.append(
    el(
      "p",
      contrast
        ? "Training contrast available: valid elegant and non-elegant paths."
        : "Audit only: this problem lacks a valid path in one of the two elegance classes.",
      "curator-note",
    ),
  );
  const list = el("div", "", "path-list");
  for (const route of state.problem.routes) {
    const label = judgment.route_labels.find((r) => r.route_id === route.id);
    const text = state.machine?.route_text?.[route.id] || route.text;
    const row = el("div", "", "path-row");
    const heading = el("div", "", "path-heading");
    heading.append(
      el("strong", route.title),
      el(
        "span",
        label.elegant ? "Elegant" : "Not elegant",
        "badge " + (label.elegant ? "train" : "validation"),
      ),
    );
    row.append(
      heading,
      el("p", label.reason),
      el(
        "small",
        `${text.trim().split(/\s+/).length} words · shortness: ${label.shortness_role} · confidence: ${label.confidence}`,
        "muted",
      ),
    );
    const actions = el("div", "", "path-actions");
    for (const i of [0, 1]) {
      const button = el("button", "Show in " + (i ? "B" : "A"), "text-button");
      button.onclick = () => {
        state.panes[i] = route.id;
        renderRoutes();
        $("route-grid").scrollIntoView({ behavior: "smooth", block: "start" });
      };
      actions.append(button);
    }
    row.append(actions);
    list.append(row);
  }
  panel.append(list);
}
function renderRoutes() {
  const grid = $("route-grid");
  grid.replaceChildren();
  for (let i = 0; i < 2; i++) {
    const pane = el("section", "", "route"),
      top = el("div", "", "route-top");
    const style = state.elegance?.route_labels?.find(
      (r) => r.route_id === state.panes[i],
    );
    const role = style
      ? style.elegant
        ? "ELEGANT · AI JUDGMENT"
        : "NOT ELEGANT · AI JUDGMENT"
      : state.machine?.status === "accepted"
        ? state.panes[i] === state.machine.elegant_route_id
          ? "AI SELECTED · ELEGANT"
          : state.panes[i] === state.machine.elementary_route_id
            ? "AI SELECTED · ELEMENTARY"
            : "SOURCE ALTERNATIVE"
        : i
          ? "ROUTE B"
          : "ROUTE A";
    top.append(el("span", role, "section-label"));
    const chooser = el("select");
    chooser.setAttribute("aria-label", "Solution in route " + (i ? "B" : "A"));
    chooser.append(
      ...state.problem.routes.map((r) =>
        option(
          r.id,
          r.title +
            " · " +
            r.source +
            (state.elegance?.route_labels?.find((a) => a.route_id === r.id)
              ? state.elegance.route_labels.find((a) => a.route_id === r.id)
                  .elegant
                ? " · Elegant"
                : " · Not elegant"
              : ""),
        ),
      ),
    );
    chooser.value = state.panes[i] || state.problem.routes[0]?.id;
    state.panes[i] = chooser.value;
    chooser.onchange = () => {
      state.panes[i] = chooser.value;
      renderRoutes();
    };
    top.append(chooser);
    const r = state.problem.routes.find((r) => r.id === chooser.value);
    if (!r) {
      pane.append(top, el("p", "No second route available"));
      grid.append(pane);
      continue;
    }
    const meta = el("div", "", "route-meta");
    meta.append(
      el(
        "span",
        state.audit.route_text[r.id]
          ? "Your edited reference"
          : state.machine?.route_text?.[r.id]
            ? "Source with explicit AI correction"
            : r.kind === "summary"
              ? "Method summary only"
              : r.kind === "model"
                ? "Model-generated solution"
                : "Human-written solution",
      ),
    );
    const edit = el("button", "Correct text", "text-button");
    edit.onclick = () => openCorrection(r.id);
    meta.append(edit);
    top.append(meta);
    const tags = el("div", "", "route-tags");
    const reviewed = state.machine?.route_assessments?.find(
      (a) => a.route_id === r.id,
    );
    for (const m of state.audit.methods[r.id] ??
      reviewed?.methods ??
      r.methods) {
      const chip = el("span", labels.get(m), "method-chip");
      chip.title = r.evidence[m] || "Manually assigned";
      tags.append(chip);
    }
    if (!tags.childNodes.length)
      tags.append(el("span", "No method proposed", "method-chip"));
    top.append(tags);
    const body = el("div", "", "route-body"),
      text = el("div", "", "math-text");
    math(
      text,
      state.audit.route_text[r.id] ||
        state.machine?.route_text?.[r.id] ||
        r.text,
    );
    body.append(text);
    if (state.audit.route_text[r.id] || state.machine?.route_text?.[r.id]) {
      const original = el("details"),
        sourceText = el("div", "", "math-text");
      original.append(el("summary", "Original source text"));
      math(sourceText, r.text);
      original.append(sourceText);
      body.append(original);
    }
    pane.append(
      top,
      body,
      el(
        "div",
        (state.audit.route_text[r.id] ? "Your corrected derivation · " : "") +
          (state.machine?.route_text?.[r.id] ? "AI source correction · " : "") +
          (reviewed
            ? `AI route assessment: ${reviewed.correctness}`
            : r.verification),
        "verification",
      ),
    );
    grid.append(pane);
  }
}
async function save() {
  readForm();
  if (!state.pid) return;
  if (!state.dirty) return;
  $("save-state").textContent = "Saving…";
  state.audit = await api("/api/audit/" + state.pid, state.audit);
  state.dirty = false;
  $("save-state").textContent =
    "Saved to disk · revision " + state.audit.revision;
  $("save-state").style.color = "";
  const row = state.index.problems.find((p) => p.id === state.pid);
  row.status = state.audit.status;
  row.methods = effectiveTags();
  row.split = splitFor(row.methods);
  row.eligible = !blockers().length;
  updateStats();
  renderList();
  readiness();
}
async function step(delta) {
  const i = state.filtered.findIndex((p) => p.id === state.pid),
    p = state.filtered[i + delta];
  if (p) await select(p.id);
}
function openAtlas() {
  const table = $("method-table");
  table.replaceChildren();
  for (const m of state.index.methods) {
    const rows = state.index.problems.filter((p) => p.methods.includes(m.id)),
      tr = el("tr");
    const name = el("td");
    name.append(el("strong", m.label));
    if (m.classification)
      name.append(el("p", m.classification.replaceAll("_", " "), "muted"));
    if (m.definition) name.append(el("p", m.definition));
    if (m.related_holdout_groups?.length)
      name.append(
        el(
          "p",
          "Overlap: " +
            m.related_holdout_groups.join(", ").replaceAll("_", " "),
          "muted",
        ),
      );
    tr.append(
      name,
      el("td", m.domain),
      el("td", rows.length.toLocaleString()),
      el("td", rows.filter((p) => p.machine_eligible).length),
    );
    const td = el("td"),
      select = el("select");
    select.dataset.method = m.id;
    select.setAttribute("aria-label", "Partition for " + m.label);
    for (const v of ["train", "validation", "test"])
      select.append(option(v, v === "test" ? "Test holdout" : v));
    select.value = state.index.settings.splits[m.id];
    td.append(select);
    tr.append(td);
    table.append(tr);
  }
  $("partition-counts").replaceChildren(
    ...["train", "validation", "test"].map((s) =>
      el(
        "span",
        `${state.index.methods.filter((m) => state.index.settings.splits[m.id] === s).length} ${s} labels`,
      ),
    ),
  );
  $("holdout-message").textContent =
    "Partitions are provisional. Related mechanisms need a joint exclusion rule before training; broad tools are not independent families.";
  $("atlas").showModal();
}
function openLabels() {
  readForm();
  const wrap = $("route-label-editor");
  wrap.replaceChildren();
  const groups = state.problem.routes.map((r) => ({
    id: r.id,
    title: r.title + " · " + r.source,
    tags: state.audit.methods[r.id] ?? r.methods,
    text: r.text,
  }));
  groups.push({
    id: "additional",
    title: "Additional methods required by the problem",
    tags: state.audit.additional_methods,
    text: "",
  });
  for (const g of groups) {
    const box = el("section", "", "label-route");
    box.append(el("h3", g.title));
    if (g.text) {
      const details = el("details");
      details.append(el("summary", "Inspect original solution"));
      const content = el("div", "", "math-text");
      math(content, g.text);
      details.append(content);
      box.append(details);
    }
    const options = el("div", "", "method-options");
    for (const m of state.index.methods) {
      const label = el("label", "", "check"),
        input = document.createElement("input");
      input.type = "checkbox";
      input.dataset.route = g.id;
      input.value = m.id;
      input.checked = g.tags.includes(m.id);
      label.append(input, document.createTextNode(m.label));
      options.append(label);
    }
    box.append(options);
    wrap.append(box);
  }
  $("labels").showModal();
}
let correctionRoute = null;
function openCorrection(rid) {
  correctionRoute = rid;
  $("corrected-text").value = state.audit.route_text[rid] || "";
  $("correction").showModal();
}
async function downloadJson(path, filename) {
  try {
    if (state.dirty) await save();
    const data = await api(path),
      blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      }),
      a = el("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    if (path === "/api/export") {
      $("save-state").textContent =
        "Exported " +
        Object.values(data.partitions).reduce((n, x) => n + x.length, 0) +
        " approved pairs";
    }
  } catch (e) {
    error(e);
  }
}
async function init() {
  try {
    state.index = await api("/api/index");
    for (const m of state.index.methods) {
      labels.set(m.id, m.label);
      $("method-filter").append(option(m.id, m.label));
    }
    const sources = [
      ...new Set(state.index.problems.flatMap((p) => p.sources)),
    ];
    for (const s of sources) $("source-filter").append(option(s, s));
    for (const note of state.index.notes)
      $("collection-notes").append(el("p", note));
    updateStats();
    applyFilters();
    const hash = location.hash.slice(1);
    await select(
      state.index.problems.some((p) => p.id === hash)
        ? hash
        : state.filtered[0]?.id,
    );
    for (const id of [
      "search",
      "status-filter",
      "source-filter",
      "split-filter",
      "method-filter",
      "pair-filter",
      "shortlist-filter",
      "machine-filter",
    ])
      $(id).addEventListener(
        id === "search" ? "input" : "change",
        applyFilters,
      );
    for (const id of [
      "elegant",
      "elementary",
      "correct_elegant",
      "correct_elementary",
      "distinct",
      "methods_reviewed",
      "answer-correction",
      "capability",
      "notes",
      "verdict",
    ])
      $(id).addEventListener("input", () => {
        readForm();
        markDirty();
        readiness();
      });
    $("prev").onclick = () => step(-1);
    $("next").onclick = () => step(1);
    $("save").onclick = () => save().catch(error);
    $("save-next").onclick = async () => {
      try {
        await save();
        await step(1);
      } catch (e) {
        error(e);
      }
    };
    $("raw-toggle").onclick = () => {
      state.raw = !state.raw;
      $("raw-toggle").textContent = state.raw
        ? "Render mathematics"
        : "Show raw text";
      readForm();
      renderDetail();
    };
    $("atlas-button").onclick = openAtlas;
    $("labels-button").onclick = openLabels;
    $("apply-labels").onclick = () => {
      const groups = {};
      for (const input of $("route-label-editor").querySelectorAll("input")) {
        groups[input.dataset.route] ??= [];
        if (input.checked) groups[input.dataset.route].push(input.value);
      }
      state.audit.additional_methods = groups.additional;
      delete groups.additional;
      state.audit.methods = groups;
      state.audit.methods_reviewed = false;
      $("methods_reviewed").checked = false;
      markDirty();
      $("labels").close();
      renderRoutes();
      readiness();
    };
    $("apply-correction").onclick = () => {
      state.audit.route_text[correctionRoute] = $("corrected-text").value;
      state.audit.methods_reviewed = false;
      $("methods_reviewed").checked = false;
      for (const role of ["elegant", "elementary"]) {
        if (state.audit[role] === correctionRoute) {
          state.audit["correct_" + role] = false;
          $("correct_" + role).checked = false;
        }
      }
      markDirty();
      $("correction").close();
      renderRoutes();
      readiness();
    };
    $("save-holdouts").onclick = async () => {
      try {
        if (state.dirty) await save();
        const splits = {};
        for (const s of $("method-table").querySelectorAll("select"))
          splits[s.dataset.method] = s.value;
        state.index.settings = await api("/api/settings", {
          revision: state.index.settings.revision,
          splits,
        });
        for (const p of state.index.problems) p.split = splitFor(p.methods);
        $("atlas").close();
        applyFilters();
        renderDetail();
      } catch (e) {
        $("holdout-message").textContent = e.message;
      }
    };
    $("backup").onclick = () =>
      downloadJson("/api/audits", "method-atlas-audits.json");
    $("machine-export").onclick = () =>
      downloadJson("/api/elegance-export", "method-atlas-labeled-paths.json");
    $("export").onclick = () =>
      downloadJson("/api/export", "method-atlas-approved.json");
    for (const b of document.querySelectorAll("[data-close]"))
      b.onclick = () => $(b.dataset.close).close();
    setInterval(async () => {
      try {
        if (state.dirty || document.querySelector("dialog[open]")) return;
        state.index = await api("/api/index");
        updateStats();
        applyFilters();
        if (state.pid) {
          const data = await api("/api/problem/" + state.pid);
          const previous = state.machine?.status;
          state.machine = data.machine_review || null;
          const labelsChanged =
            JSON.stringify(state.elegance) !==
            JSON.stringify(data.elegance_review || null);
          state.elegance = data.elegance_review || null;
          if (
            previous !== "accepted" &&
            state.machine?.status === "accepted" &&
            !state.audit.elegant &&
            !state.audit.elementary
          ) {
            state.panes = [
              state.machine.elegant_route_id,
              state.machine.elementary_route_id,
            ];
            renderRoutes();
          }
          renderMachineReview();
          renderPathReview();
          if (labelsChanged) renderRoutes();
        }
      } catch (_) {
        /* Keep the current review visible during a server restart. */
      }
    }, 30000);
    window.addEventListener("hashchange", () => {
      const pid = location.hash.slice(1);
      if (pid !== state.pid && state.index.problems.some((p) => p.id === pid))
        select(pid);
    });
    window.addEventListener("beforeunload", (e) => {
      if (state.dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    });
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        save().catch(error);
        return;
      }
      if (
        document.querySelector("dialog[open]") ||
        ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)
      )
        return;
      if (e.key === "j") step(1);
      if (e.key === "k") step(-1);
    });
  } catch (e) {
    error(e);
  }
}
init();
