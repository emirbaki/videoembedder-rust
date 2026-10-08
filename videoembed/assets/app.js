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
  $("status").textContent = state.records + " windows, " + state.videos + " videos";
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
    count.textContent = video.windows + " windows";

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
  note("");
  try {
    await api("/api/index", { path: folder });
    startPolling();
  } catch (err) {
    $("start").disabled = false;
    note(err.message);
  }
}

function showProgress(p) {
  $("progress").hidden = !p.running && p.indexed === 0 && !p.error;
  const width = p.window_count > 0 ? (p.window * 100) / p.window_count : 0;
  $("progress-bar").style.width = width + "%";

  if (p.error !== null) {
    $("progress-text").textContent = p.error;
    return;
  }
  if (p.running) {
    $("progress-text").textContent =
      "File " + p.file_index + "/" + p.file_count +
      ", window " + p.window + "/" + p.window_count +
      " - " + p.file;
    return;
  }
  $("progress-text").textContent = "Done. " + p.indexed + " windows indexed.";
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
    renderGroups(answer.groups);
  } catch (err) {
    note(err.message);
    renderGroups([]);
  }
}

function renderGroups(groups) {
  const list = $("groups");
  list.textContent = "";
  const timeline = $("timeline");
  timeline.textContent = "";

  if (groups.length === 0) {
    note("No window scored above 0.4.");
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
      row.textContent =
        hit.score.toFixed(3) + "  " + hit.start.toFixed(1) + "-" + hit.end.toFixed(1) + " s";
      row.addEventListener("click", () => play(group.video, hit.start));
      item.appendChild(row);
    }

    list.appendChild(item);
    timeline.appendChild(timelineBar(group));
  }
}

function timelineBar(group) {
  const bar = document.createElement("div");
  bar.className = "track";

  const name = document.createElement("span");
  name.className = "track-name";
  name.textContent = group.video;
  bar.appendChild(name);

  // The store holds no true duration. The largest window end is the axis.
  const last = group.hits.length > 0 ? group.hits[group.hits.length - 1].end : 0;
  const span = group.duration > 0 ? group.duration : last > 0 ? last : 1;

  for (const hit of group.hits) {
    const mark = document.createElement("button");
    mark.type = "button";
    mark.className = "mark";
    mark.style.left = (hit.start / span) * 100 + "%";
    mark.style.width = ((hit.end - hit.start) / span) * 100 + "%";
    mark.title = hit.score.toFixed(3) + "  " + hit.start.toFixed(1) + " s";
    mark.addEventListener("click", () => play(group.video, hit.start));
    bar.appendChild(mark);
  }
  return bar;
}

function play(path, start) {
  const player = $("player");
  if (player.dataset.path !== path) {
    player.dataset.path = path;
    player.src = "/api/video?key=" + encodeURIComponent(key) +
      "&path=" + encodeURIComponent(path);
  }
  const seek = () => {
    player.currentTime = start;
    player.play();
  };
  if (player.readyState >= 1) seek();
  else player.addEventListener("loadedmetadata", seek, { once: true });
}

function boot() {
  $("pick").addEventListener("click", pickFolder);
  $("start").addEventListener("click", startIndex);
  $("search-form").addEventListener("submit", runSearch);
  refreshState().catch((err) => note(err.message));
  refreshVideos().catch((err) => note(err.message));
}

boot();
