// SmartGuard JKN without a server. make_app_demo.py computed the answers for synthetic claims
// ahead of time; this file stands in for the API, and decisions stay in this browser.
const files = new Map();
const load = name => {
  if (!files.has(name)) files.set(name, fetch(`api/${name}.json`).then(res => {
    if (!res.ok) throw new Error(`${name}: ${res.status}`);
    return res.json();
  }));
  return files.get(name);
};

const LOG = "smartguard-reviews";
function readLog() {
  try { return JSON.parse(localStorage[LOG] || "[]"); } catch { return []; }
}
function writeLog(entries) {
  try { localStorage[LOG] = JSON.stringify(entries); } catch {}
}
const withDecisions = rows => {
  const latest = Object.fromEntries(readLog().map(entry => [entry.row, entry.decision]));
  return rows.map(row => ({ ...row, decision: latest[row.row] ?? null }));
};

function simulate(claim, changes) {
  const secs = [changes.sec1, changes.sec2, changes.sec3].filter(code => code && code !== "-").sort();
  const answer = (claim.simulate || {})[`${changes.sev}|${changes.los}|${secs}`];
  if (!answer) return { fields: claim.fields, severity: claim.severity, explanation: [
    "Versi online hanya memuat perubahan severity dan diagnosis sekunder yang sudah ada pada klaim di queue. Kombinasi lain dihitung langsung pada aplikasi lengkap.",
  ] };
  return {
    fields: claim.fields.map((field, i) => {
      const [value, p, anomaly] = answer.f[i];
      return { ...field, value, p, anomaly, name: value === field.value ? field.name : "" };
    }),
    severity: claim.severity.map((level, i) => ({ ...level, p: answer.s[i], claimed: level.level === changes.sev })),
    explanation: answer.e,
  };
}

async function staticApi(path, body) {
  const [, route, query] = path.match(/^\/api\/([^?]+)\??(.*)$/);
  const [, row, action] = route.match(/^claim\/(\d+)\/?(.*)$/) || [];
  if (route === "summary") {
    const entries = readLog();
    return { ...await load("summary"), reviewed: new Set(entries.map(e => e.row)).size,
             log_entries: entries.length, log_intact: true };
  }
  if (route === "queue") return withDecisions(await load(`queue-${new URLSearchParams(query).get("kind")}`));
  if (route.startsWith("facility/")) {
    const facility = await load(route);
    return { ...facility, claims: withDecisions(facility.claims) };
  }
  if (!row) return load(route);  // facilities, bridge
  if (action === "fhir" || action === "satusehat") return load(`${action}/${row}`);
  const claim = await load(`claim/${row}`);
  if (action === "simulate") return simulate(claim, body);
  if (action === "review") {
    const entry = { time: new Date().toISOString().slice(0, 19) + "+00:00", row: Number(row), ...body };
    writeLog([...readLog(), entry]);
    return entry;
  }
  return { ...claim, options: await load("options"), visits: withDecisions(claim.visits),
           reviews: readLog().filter(entry => entry.row === Number(row)) };
}
