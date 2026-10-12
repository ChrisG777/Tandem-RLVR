/* Presentation only: consumes prevalidated index and per-condition sample files. */
"use strict";
const $ = (id) => document.getElementById(id);
let data,
  filtered = [],
  selected,
  requestVersion = 0;
const visible = new Set(),
  sampleIndices = new Map();
const node = (tag, text, cls) => {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
};
async function readJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
function updateFilters(preferred) {
  const q = $("search").value.toLowerCase(),
    mode = $("filter").value;
  filtered = data.problems.filter((p) => {
    const values = Object.values(p.conditions),
      rates = values.map((c) => c.correct / c.count);
    return (
      (!$("benchmark").value || p.benchmark === $("benchmark").value) &&
      (!q || `${p.id} ${p.prompt}`.toLowerCase().includes(q)) &&
      (mode === "all" ||
        (mode === "matched" && values.length === data.conditions.length) ||
        (mode === "disagree" && new Set(rates).size > 1) ||
        (mode === "split" &&
          rates.some((x) => x === 0) &&
          rates.some((x) => x > 0)))
    );
  });
  $("problem").replaceChildren(
    ...filtered.map((p) => {
      const o = node(
        "option",
        `${p.id} · ${p.prompt.replace(/\s+/g, " ").slice(0, 85)}`,
      );
      o.value = p.id;
      return o;
    }),
  );
  selected =
    filtered.find((p) => p.id === (preferred || selected?.id)) || filtered[0];
  if (selected) $("problem").value = selected.id;
  render();
}
async function render() {
  const version = ++requestVersion,
    pos = filtered.indexOf(selected);
  $("count").textContent = selected
    ? `${pos + 1} / ${filtered.length} problems`
    : "No matching problems";
  $("prev").disabled = pos <= 0;
  $("next").disabled = pos < 0 || pos === filtered.length - 1;
  $("panels").replaceChildren();
  $("status").textContent = "";
  for (const id of ["prompt", "answer", "problem-id"]) $(id).textContent = "";
  if (!selected) return;
  $("prompt").textContent = selected.prompt;
  $("answer").textContent = selected.answer;
  $("problem-id").textContent = selected.id;
  history.replaceState(null, "", `#${encodeURIComponent(selected.id)}`);
  const conditions = data.conditions.filter((c) => visible.has(c.id));
  $("panels").style.setProperty(
    "--columns",
    Math.min(conditions.length, 3) || 1,
  );
  $("status").textContent = conditions.length
    ? "Loading traces…"
    : "Select a condition to view.";
  const problem = selected;
  const results = await Promise.allSettled(
    conditions.map(async (c) => ({
      c,
      samples: problem.conditions[c.id]
        ? await readJSON(`problems/${problem.id}-${c.id}.json`)
        : null,
    })),
  );
  if (version !== requestVersion) return;
  $("status").textContent = "";
  results.forEach((result, i) => {
    if (result.status === "rejected") {
      const p = node(
        "section",
        `${conditions[i].label}: ${result.reason.message}`,
        "panel",
      );
      $("panels").append(p);
      return;
    }
    const { c, samples } = result.value,
      panel = node("section", undefined, "panel"),
      head = node("div", undefined, "panel-head");
    head.append(node("h2", c.label));
    panel.append(head);
    $("panels").append(panel);
    if (!samples) {
      head.append(node("p", "Not available in this snapshot."));
      return;
    }
    const picker = node("select");
    picker.setAttribute("aria-label", `${c.label} sample`);
    samples.forEach((s, i) => {
      const o = node(
        "option",
        `Sample ${i + 1} · ${s.correct ? "correct" : "incorrect"}`,
      );
      o.value = i;
      picker.append(o);
    });
    picker.value = Math.min(sampleIndices.get(c.id) || 0, samples.length - 1);
    const badge = node("span"),
      meta = node("div", undefined, "meta"),
      warning = node("p", undefined, "warning"),
      trace = node("div", undefined, "trace text");
    trace.tabIndex = 0;
    trace.setAttribute("aria-label", `${c.label} reasoning`);
    head.append(picker, node("br"), meta, badge, warning);
    panel.append(trace);
    function show() {
      const s = samples[Number(picker.value)];
      sampleIndices.set(c.id, Number(picker.value));
      badge.textContent = s.correct ? "Correct" : "Incorrect";
      badge.className = `badge ${s.correct ? "correct" : ""}`;
      meta.textContent = `${problem.conditions[c.id].correct}/${samples.length} saved samples correct${s.tokens !== undefined ? ` · ${s.tokens.toLocaleString()} tokens` : ""}${s.senior_fraction != null ? ` · ${Math.round(s.senior_fraction * 100)}% senior` : ""}${s.finish_reason ? ` · ${s.finish_reason === "length" ? "Token limit reached" : s.finish_reason}` : ""}`;
      warning.textContent = s.authorship_warning || "";
      trace.replaceChildren();
      if (s.spans)
        s.spans.forEach((span) => {
          const el = node("span", span.text, span.author);
          el.title =
            span.author === "senior"
              ? "Trained senior"
              : span.author === "base"
                ? "Frozen base"
                : "Character spans tokens from both authors";
          trace.append(el);
        });
      else trace.textContent = s.text;
      trace.scrollTop = 0;
    }
    picker.addEventListener("change", show);
    show();
  });
}
async function init() {
  data = await readJSON("index.json");
  data.conditions.forEach((c) => {
    visible.add(c.id);
    const label = node("label"),
      input = node("input");
    input.type = "checkbox";
    input.checked = true;
    input.addEventListener("change", () => {
      input.checked ? visible.add(c.id) : visible.delete(c.id);
      render();
    });
    label.append(input, document.createTextNode(c.label));
    $("condition-controls").append(label);
    const source = node("div", undefined, "source");
    source.append(
      node(
        "strong",
        `${c.label} · ${c.problems} problems × ${c.n} samples${c.partial ? " · PARTIAL SNAPSHOT" : ""}`,
      ),
      node("div", c.source),
    );
    $("sources").append(source);
  });
  [...new Set(data.problems.map((p) => p.benchmark))].forEach((b) => {
    const o = node("option", b);
    o.value = b;
    $("benchmark").append(o);
  });
  $("protocol").textContent = data.note;
  for (const id of ["benchmark", "filter"])
    $(id).addEventListener("change", () => updateFilters());
  $("search").addEventListener("input", () => updateFilters());
  $("problem").addEventListener("change", () => {
    selected = filtered.find((p) => p.id === $("problem").value);
    render();
  });
  for (const [id, step] of [
    ["prev", -1],
    ["next", 1],
  ])
    $(id).addEventListener("click", () => {
      selected = filtered[filtered.indexOf(selected) + step];
      $("problem").value = selected.id;
      render();
    });
  $("colors").addEventListener("change", () =>
    document.body.classList.toggle("no-colors", !$("colors").checked),
  );
  const initial = decodeURIComponent(location.hash.slice(1));
  if (
    initial &&
    data.problems.find(
      (p) =>
        p.id === initial &&
        Object.keys(p.conditions).length < data.conditions.length,
    )
  )
    $("filter").value = "all";
  updateFilters(initial);
}
init().catch((e) => {
  $("status").textContent =
    `Could not load browser: ${e.message}. Serve this directory over localhost; see docs/COMPARISON_BROWSER.md.`;
});
