/* Static build only (GitHub Pages): plays back recorded runs of the Distill engine in place of the live server.
   scripts/build_static.py records the runs and inlines this file into docs/index.html. */
const REPLAY = (() => {
  const SPEED = 8;  // invoices per second, as on the live dashboard
  const realFetch = window.fetch.bind(window);
  const R = { sets: {}, cur: "fatura", loaded: null };

  function fresh(d){
    return { d, pos: 0, running: false, ended: false, branch: null, bpos: 0, used: false,
             ticks: d.ticks.slice(), curve: d.curve.slice(), events: d.events.slice(), sc: d.scalars.slice(),
             docs: Object.assign({}, d.docs) };
  }
  function apply(st, f){
    st.ticks.push(f.t); if (st.ticks.length > 400) st.ticks.splice(0, st.ticks.length - 400);
    st.sc = f.s;
    if (f.c) st.curve.push(...f.c);
    if (f.e) st.events.push(...f.e);
    st.docs[f.t.doc_id] = f.d;
  }
  function next(st){
    const list = st.branch ? st.branch.frames : st.d.frames, i = st.branch ? st.bpos : st.pos;
    if (i >= list.length) return null;
    if (st.branch) st.bpos++; else st.pos++;
    return list[i];
  }
  setInterval(() => {
    const st = R.sets[R.cur];
    if (!st || !st.running) return;
    const f = next(st);
    if (!f){ st.running = false; st.ended = true; return; }
    apply(st, f);
  }, 1000 / SPEED);

  function snapshot(){
    const st = R.sets[R.cur]; if (!st) return null;
    const [docs, cost, small, large, acc, base_acc, lat, base_lat] = st.sc;
    return { dataset: R.cur, running: st.running, ended: st.ended, public: true, models: st.d.models,
             totals: { docs, cost_usd: cost }, counterfactual: { small, large }, accuracy: acc, baseline_accuracy: base_acc,
             latency_ms_p50: lat, baseline_latency_ms_p50: base_lat, curve: st.curve.slice(-400), ticks: st.ticks,
             events: st.events.slice(-60), vendors: st.d.vendors, drift: st.d.drift || null };
  }

  function changeTemplate(st, body){
    const dr = st.d.drift;
    if (!dr || body.vendor_id !== dr.vendor_id || body.kind !== dr.kind)
      return { ok: false, error: "This recorded run includes one template change: the silent price and quantity swap." };
    if (st.used) return { ok: false, error: "Already changed in this recorded run. Reload the page to watch it again." };
    // The change was recorded at several points of the run: use the next one (or the last), rebuilt from the start.
    const forks = st.d.forks, fork = forks.find(f => f.at >= st.pos) || forks[forks.length - 1];
    const running = st.running, rebuilt = fresh(st.d);
    for (let i = 0; i < fork.at; i++) apply(rebuilt, st.d.frames[i]);
    Object.assign(st, rebuilt, { pos: fork.at, running, used: true, branch: fork, bpos: 0 });
    st.events.push(...fork.events);
    return { ok: true, desc: fork.desc };
  }

  window.fetch = async (url, opts) => {
    const u = String(url);
    if (!u.startsWith("/api/")) return realFetch(url, opts);
    await R.loaded;
    const body = opts && opts.body ? JSON.parse(opts.body) : {};
    const reply = (o, status = 200) => new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json" } });
    const st = R.sets[R.cur];
    if (u.startsWith("/api/doc/")){
      const d = st.docs[decodeURIComponent(u.slice(9))];
      return d ? reply(d) : reply({ error: "not in the recorded run" }, 404);
    }
    if (u.startsWith("/api/vendor/")){
      const dr = st.d.drift;
      return reply({ drift_options: dr && dr.vendor_id === u.slice(12) && !st.used ? [dr.kind] : [] });
    }
    if (u === "/api/control"){
      if (body.action === "start" && !st.ended) st.running = true;
      else if (body.action === "pause") st.running = false;
      else if (body.action === "dataset" && R.sets[body.dataset]){ st.running = false; R.cur = body.dataset; }
      return reply({ ok: true });
    }
    if (u === "/api/drift") return reply(changeTemplate(st, body));
    return reply({ ok: false, error: "not available in the recorded run" }, 404);
  };

  R.loaded = Promise.all(["fatura", "synthetic"].map(async name => {
    const d = await (await realFetch(`data/${name}.json`)).json();
    R.sets[name] = fresh(d);
  }));
  R.snapshot = snapshot;
  return R;
})();

class ReplayStream {
  constructor(){
    this.onmessage = null;
    const note = () => {
      let el = document.getElementById("replay-note");
      if (!el){
        el = document.createElement("p"); el.id = "replay-note";
        el.style.cssText = "margin:8px 0 0;font-size:14px;color:var(--ink-2)";
        document.querySelector(".mast > div").appendChild(el);
      }
      return el;
    };
    note().textContent = "Loading the recorded runs…";
    REPLAY.loaded.then(() => {
      const send = () => {
        const s = REPLAY.snapshot();
        note().innerHTML = s.ended
          ? "<strong>End of the recorded run.</strong> Reload the page to watch it again."
          : "A recorded run of the real Distill engine, played back in your browser. AI calls are simulated at Claude's prices and timings.";
        if (this.onmessage) this.onmessage({ data: JSON.stringify(s) });
      };
      send(); setInterval(send, 500);
    });
  }
}
