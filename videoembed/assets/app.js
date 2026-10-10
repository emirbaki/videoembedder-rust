const API = { "X-VEI": "1", "Content-Type": "application/json" };
const $ = (id) => document.getElementById(id);

let key = "";

async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: API,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.status + " " + res.statusText);
  return data;
}

function note(text) {
  const el = $("message");
  el.textContent = text;
  el.hidden = text === "";
}

async function refreshState() {
  const state = await api("/api/state");
  key = state.key;
  $("status").textContent = state.records + " records, " + state.videos + " videos";
}

async function refreshVideos() {
  const answer = await api("/api/videos");
  const body = $("videos").querySelector("tbody");
  body.textContent = "";
  if (answer.videos.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.textContent = "No video is indexed yet.";
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  for (const video of answer.videos) {
    const row = document.createElement("tr");

    const name = document.createElement("td");
    name.textContent = video.path;

    const count = document.createElement("td");
    count.textContent = video.items + " items";

    const act = document.createElement("td");
    const drop = document.createElement("button");
    drop.type = "button";
    drop.textContent = "Forget";
    drop.addEventListener("click", () => forget(video.path));
    act.appendChild(drop);

    row.append(name, count, act);
    body.appendChild(row);
  }
}

async function forget(path) {
  try {
    await api("/api/forget", { path });
    await refreshVideos();
    await refreshState();
  } catch (err) {
    note(err.message);
  }
}

let folder = null;
let timer = null;

async function pickFolder() {
  try {
    const answer = await api("/api/pick-folder", {});
    if (answer.path === null) return;
    folder = answer.path;
    $("folder").textContent = folder;
    $("start").disabled = false;
    note("");
  } catch (err) {
    note(err.message);
  }
}

async function startIndex() {
  if (folder === null) return;
  $("start").disabled = true;
  $("stop").disabled = false;
  note("");
  try {
    await api("/api/index", { path: folder });
    startPolling();
  } catch (err) {
    $("start").disabled = false;
    $("stop").disabled = true;
    note(err.message);
  }
}

async function stopIndex() {
  try {
    await api("/api/stop", {});
  } catch (err) {
    note(err.message);
  }
}

function showProgress(p) {
  $("progress").hidden = !p.running && p.indexed === 0 && !p.error;
  const width = p.window_count > 0 ? (p.window * 100) / p.window_count : 0;
  $("progress-bar").style.width = width + "%";
  // The speed line is set first, so an error cannot leave stale text.
  $("speed").textContent = p.elapsed_ms > 0
    ? ((p.elapsed_ms / 1000).toFixed(1) + " s of work, " + p.backend)
    : "";

  if (p.error !== null) {
    $("progress-text").textContent = p.error;
    return;
  }
  if (p.running) {
    const rate = p.per_second > 0 ? " - " + p.per_second.toFixed(2) + " items a second" : "";
    $("progress-text").textContent =
      "File " + p.file_index + "/" + p.file_count +
      ", item " + p.window + "/" + p.window_count +
      " - " + p.file + rate;
  } else {
    const rate = p.per_second > 0 ? " at " + p.per_second.toFixed(2) + " items a second" : "";
    $("progress-text").textContent =
      "Done. " + p.indexed + " records indexed" + rate +
      ". " + p.backend + ", model load " + p.load_ms + " ms.";
  }
}

async function pollProgress() {
  let p;
  try {
    p = await api("/api/progress");
  } catch (err) {
    note(err.message);
    return;
  }
  showProgress(p);
  if (p.running) return;
  clearInterval(timer);
  timer = null;
  $("start").disabled = false;
  $("stop").disabled = true;
  await refreshVideos();
  await refreshState();
}

function startPolling() {
  if (timer !== null) return;
  timer = setInterval(pollProgress, 500);
  pollProgress();
}

async function runSearch(event) {
  event.preventDefault();
  const query = $("query").value.trim();
  if (query === "") return;
  note("");
  try {
    const answer = await api("/api/search", { query });
    renderGroups(answer.groups, answer.baseline);
  } catch (err) {
    note(err.message);
    renderGroups([], null);
  }
}

function renderGroups(groups, baseline) {
  const list = $("groups");
  list.textContent = "";
  const timeline = $("timeline");
  timeline.textContent = "";

  if (groups.length === 0) {
    note("No record scored above the floor.");
    return;
  }
  note("");
  for (const group of groups) {
    const item = document.createElement("li");

    const title = document.createElement("strong");
    title.textContent = group.video;
    item.appendChild(title);

    for (const hit of group.hits) {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "hit";

      const shot = document.createElement("img");
      shot.className = "shot";
      shot.loading = "lazy";
      shot.alt = "frame at " + hit.start.toFixed(2) + " s";
      shot.src = "/api/frame?key=" + encodeURIComponent(key) +
        "&path=" + encodeURIComponent(group.video) +
        "&t=" + hit.start.toFixed(2) + "&w=320";

      const text = document.createElement("span");
      text.textContent =
        hit.score.toFixed(3) + " (" + hit.margin.toFixed(3) + ")  " +
        hit.start.toFixed(2) + "-" + hit.end.toFixed(2) + " s";

      row.append(shot, text);
      row.addEventListener("click", () => play(group.video, hit.start));
      item.appendChild(row);
    }

    list.appendChild(item);
    timeline.appendChild(timelineBar(group, baseline));
  }
}

// The colour of one cell. Grey is weak, green is strong.
function strengthColor(t) {
  const weak = [48, 54, 61];
  const strong = [63, 185, 80];
  const mix = weak.map((w, i) => Math.round(w + (strong[i] - w) * t));
  return "rgb(" + mix.join(",") + ")";
}

// The highest score of the cells.
function topScore(cells) {
  let high = -Infinity;
  for (const cell of cells) {
    if (cell.score > high) high = cell.score;
  }
  return high;
}

// The most cells one gradient may hold. A long video keeps its strongest cell
// in each group of cells.
const CELL_CAP = 600;

function barCells(cells) {
  if (cells.length <= CELL_CAP) return cells;
  const step = cells.length / CELL_CAP;
  const out = [];
  for (let at = 0; at < cells.length; at += step) {
    const from = Math.floor(at);
    const to = Math.min(Math.floor(at + step), cells.length);
    let best = cells[from];
    for (let j = from; j < to; j++) {
      if (cells[j].score > best.score) best = cells[j];
    }
    out.push(best);
  }
  return out;
}

// One colour stop at the middle of each cell. The browser blends between the
// stops, so the bar runs from grey to green without hard steps.
//
// The scale is the margin over the store median. A cell at the median is grey,
// and the best cell of the video is green. A video with no real match stays
// grey the whole way, because no cell rises above the median.
function barGradient(cells, span, baseline) {
  const base = Number.isFinite(baseline) ? baseline : 0;
  const high = topScore(cells);
  const width = high - base;
  const middle = (cell) => (cell.start + cell.end) / 2;
  // A CSS gradient reads left to right, so the stops must climb. Cells of
  // different lengths can interleave, and the midpoint sets the order.
  const stops = cells
    .slice()
    .sort((a, b) => middle(a) - middle(b))
    .map((cell) => {
      const raw = width > 1e-6 ? (cell.score - base) / width : 0;
      const t = Math.max(0, Math.min(1, raw));
      const at = ((middle(cell) / span) * 100).toFixed(3);
      return strengthColor(t) + " " + at + "%";
    });
  return "linear-gradient(to right, " + stops.join(", ") + ")";
}

function timelineBar(group, baseline) {
  const track = document.createElement("div");
  track.className = "track";

  const name = document.createElement("span");
  name.className = "track-name";
  name.textContent = group.video;
  track.appendChild(name);

  const wrap = document.createElement("div");
  wrap.className = "bar-wrap";

  // The store holds no true duration. The largest record end is the axis.
  const cells = group.cells || [];
  const last = cells.length > 0 ? cells[cells.length - 1].end : 0;
  const span = group.duration > 0 ? group.duration : last > 0 ? last : 1;
  const shown = barCells(cells);

  const bar = document.createElement("button");
  bar.type = "button";
  bar.className = "match-bar";
  bar.style.background = barGradient(shown, span, baseline);
  bar.title = "Click to seek. Green is a strong match.";
  bar.addEventListener("click", (event) => {
    const box = bar.getBoundingClientRect();
    const at = ((event.clientX - box.left) / box.width) * span;
    play(group.video, Math.max(0, Math.min(at, span)));
  });
  wrap.appendChild(bar);

  const best = group.hits[0];
  if (best) {
    const peak = document.createElement("span");
    peak.className = "peak";
    peak.style.left = (best.start / span) * 100 + "%";
    peak.title = best.score.toFixed(3) + " (" + best.margin.toFixed(3) + ")  " +
      best.start.toFixed(2) + " s";
    wrap.appendChild(peak);
  }

  track.appendChild(wrap);
  return track;
}

function play(path, start) {
  const player = $("player");
  const source = "/api/video?key=" + encodeURIComponent(key) +
    "&path=" + encodeURIComponent(path);

  const seek = () => {
    player.currentTime = start;
    player.play();
  };

  if (player.dataset.path !== path) {
    player.dataset.path = path;
    player.addEventListener("loadedmetadata", seek, { once: true });
    player.src = source;
    return;
  }
  if (player.readyState >= 1) seek();
  else player.addEventListener("loadedmetadata", seek, { once: true });
}

async function quit() {
  try {
    await api("/api/quit", {});
  } catch (err) {
    note(err.message);
  }
}

function boot() {
  $("pick").addEventListener("click", pickFolder);
  $("start").addEventListener("click", startIndex);
  $("stop").addEventListener("click", stopIndex);
  $("search-form").addEventListener("submit", runSearch);
  $("quit").addEventListener("click", quit);
  refreshState().catch((err) => note(err.message));
  refreshVideos().catch((err) => note(err.message));
}

boot();
