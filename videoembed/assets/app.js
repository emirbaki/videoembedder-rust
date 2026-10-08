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

function boot() {
  $("pick").addEventListener("click", pickFolder);
  $("start").addEventListener("click", startIndex);
  refreshState().catch((err) => note(err.message));
  refreshVideos().catch((err) => note(err.message));
}

boot();
