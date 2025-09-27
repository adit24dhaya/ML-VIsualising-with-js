// PCA Visualizer (2D) — fixed HUD wiring, animation, color modes, clusters, point size.

// ---------- state ----------
let W = 960, H = 600;
let pts = [];           // screen-space points [{x,y}]
let N = 400;

let anis = 5;           // major/minor std ratio
let thetaDeg = 30;      // rotation of the Gaussian (deg)

let showProj = true;
let showRecon = false;
let showAxes = true;
let spin = false;
let spinSpeed = 30;     // deg/s
let ptSize = 3;
let colorMode = 'none'; // 'none' | 'pc1' | 'cluster'
let kClusters = 0;
let clusters = [];      // labels per point or []

let varMode = 'pca';    // 'pca' | 'view'

let cam = {tx:0, ty:0, zoom:1};

// stable latent samples; we rotate/scale these to get pts => smooth animation
let baseUV = []; // [{u,v}]

// ---------- util ----------
function el(id){ return document.getElementById(id); }

// Box–Muller Gaussian
function randn(){ let u=0,v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
function radians(d){ return d * Math.PI / 180; }

// vectors
function add2(a,b){ return {x:a.x+b.x, y:a.y+b.y}; }
function sub2(a,b){ return {x:a.x-b.x, y:a.y-b.y}; }
function mul2(v,s){ return {x:v.x*s, y:v.y*s}; }
function dot2(a,b){ return a.x*b.x + a.y*b.y; }

// math helpers
function mean2(A){ let sx=0, sy=0; for(const p of A){ sx+=p.x; sy+=p.y; } const n=Math.max(1,A.length); return {x:sx/n, y:sy/n}; }
function cov2(A, mu){
  let sxx=0, sxy=0, syy=0; const n=A.length;
  for(const p of A){ const dx=p.x-mu.x, dy=p.y-mu.y; sxx+=dx*dx; sxy+=dx*dy; syy+=dy*dy; }
  const c = 1/Math.max(1, n-1); return {sxx:sxx*c, sxy:sxy*c, syy:syy*c};
}
// eigen for symmetric 2x2
function eig2x2(S){
  const a=S.sxx, b=S.sxy, c=S.syy;
  const tr=a+c, det=a*c-b*b;
  const disc=Math.sqrt(Math.max(0,(tr*tr)/4 - det));
  const l1=tr/2 + disc, l2=tr/2 - disc;
  function vecFor(l){
    if (Math.abs(b) > 1e-12){ const vx=b, vy=l-a; const n=Math.hypot(vx,vy)||1; return {x:vx/n,y:vy/n}; }
    return (a>=c) ? {x:1,y:0} : {x:0,y:1};
  }
  return { vals:[l1,l2], vecs:[vecFor(l1), vecFor(l2)] };
}

// UI helpers
function nf(x, minInt=1, frac=1){ return Number(x).toFixed(frac); }

// ---------- k-means (simple) ----------
function kmeans(points, k, iters=10){
  if (k<=0 || points.length===0) return {labels:new Array(points.length).fill(0)};
  // k-means++ init
  const cent = [];
  cent.push({...points[Math.floor(Math.random()*points.length)]});
  while(cent.length<k){
    const d2 = points.map(p => Math.min(...cent.map(c => (p.x-c.x)**2 + (p.y-c.y)**2)));
    const sum = d2.reduce((a,b)=>a+b,0)||1;
    let r = Math.random()*sum, idx=0;
    for(; idx<d2.length-1 && (r -= d2[idx])>0; idx++);
    cent.push({...points[idx]});
  }
  let labels = new Array(points.length).fill(0);
  for(let t=0;t<iters;t++){
    // assign
    for (let i=0;i<points.length;i++){
      let best=0, bd=Infinity;
      for (let c=0;c<k;c++){
        const dx=points[i].x-cent[c].x, dy=points[i].y-cent[c].y;
        const d=dx*dx+dy*dy; if(d<bd){bd=d;best=c;}
      }
      labels[i]=best;
    }
    // move
    const acc = Array.from({length:k},()=>({x:0,y:0,n:0}));
    for(let i=0;i<points.length;i++){ const L=labels[i]; acc[L].x+=points[i].x; acc[L].y+=points[i].y; acc[L].n++; }
    for(let c=0;c<k;c++){ if(acc[c].n){ cent[c].x=acc[c].x/acc[c].n; cent[c].y=acc[c].y/acc[c].n; } }
  }
  return {labels};
}

// ---------- data ----------
function rebuildBaseUV(){
  baseUV = new Array(N);
  for (let i=0;i<N;i++) baseUV[i] = {u:randn(), v:randn()};
}

function regenerate(){
  if (baseUV.length !== N) rebuildBaseUV();

  pts = [];
  const sMajor = anis * 10, sMinor = 10;
  const cx = W*0.5, cy = H*0.5;
  const th = radians(thetaDeg);
  const ct = Math.cos(th), st = Math.sin(th);

  for (let i=0;i<N;i++){
    const u = baseUV[i].u * sMajor;
    const v = baseUV[i].v * sMinor;
    const x = u*ct - v*st + cx;
    const y = u*st + v*ct + cy;
    pts.push({x,y});
  
  // refresh clusters after rebuilding points
  if (kClusters > 0){ const res = kmeans(pts, kClusters); clusters = res.labels; } else { clusters = []; }
}

  // clusters
  if (kClusters > 0){
    const res = kmeans(pts, kClusters);
    clusters = res.labels;
  } else {
    clusters = [];
  }
}

// ---------- UI ----------
function wireUI(){
  const nPts = el('nPts'), nVal = el('nPtsVal');
  if (nPts){
    nVal && (nVal.textContent = nPts.value);
    nPts.oninput = ()=>{ N=+nPts.value; nVal && (nVal.textContent=N); rebuildBaseUV(); regenerate(); };
  }

  const anisS = el('anis'), anisVal = el('anisVal');
  if (anisS){
    anisVal && (anisVal.textContent = anisS.value+'×');
    anisS.oninput = ()=>{ anis=+anisS.value; anisVal && (anisVal.textContent=anis+'×'); regenerate(); };
  }

  const thetaS = el('theta'), thetaVal = el('thetaVal');
  if (thetaS){
    thetaVal && (thetaVal.textContent = thetaS.value+'°');
    thetaS.oninput = ()=>{ thetaDeg=+thetaS.value; thetaVal && (thetaVal.textContent=thetaDeg+'°'); regenerate(); };
  }

  const proj = el('showProj'); if (proj) proj.onchange = ()=> showProj = proj.checked;
  const recon = el('recon'); if (recon) recon.onchange = ()=> showRecon = recon.checked;
  const axes = el('showAxes'); if (axes) axes.onchange = ()=> showAxes = axes.checked;

  const kS = el('kClusters'), kVal = el('kVal');
  if (kS){
    kVal && (kVal.textContent = (+kS.value>0 ? kS.value : 'off'));
    kS.oninput = ()=>{
      kClusters = +kS.value;
      kVal && (kVal.textContent = kClusters>0 ? kClusters : 'off');
      // compute clusters immediately when enabled
      if (kClusters > 0){
        const res = kmeans(pts, kClusters);
        clusters = res.labels;
        const cm = el('colorMode');
        if (cm && cm.value !== 'cluster'){ cm.value = 'cluster'; colorMode = 'cluster'; }
      } else {
        clusters = [];
        const cm = el('colorMode');
        if (cm && cm.value === 'cluster'){ cm.value = 'none'; colorMode = 'none'; }
      }
    };
  }

  const ps = el('ptSize'), pv = el('ptVal');
  if (ps){
    pv && (pv.textContent = ps.value+'px');
    ps.oninput = ()=>{ ptSize=+ps.value; pv && (pv.textContent=ptSize+'px'); };
  }

  const cm = el('colorMode'); if (cm){ colorMode=cm.value; cm.onchange = ()=> colorMode=cm.value; }
  const vm = el('varMode');   if (vm){ varMode=vm.value; vm.onchange = ()=> varMode=vm.value; }

  const spinC = el('spin'), spd = el('spinSpeed'), sVal = el('speedVal');
  function updateSpinUI(){ if (spd) spd.disabled = !spinC.checked; if (sVal) sVal.textContent = (spinC.checked?spinSpeed:0)+'°/s'; }
  if (spinC){ spin = spinC.checked; updateSpinUI(); spinC.onchange = ()=>{ spin=spinC.checked; updateSpinUI(); }; }
  if (spd){ spd.value = String(spinSpeed); spd.oninput = e=>{ spinSpeed=+e.target.value; if(sVal) sVal.textContent=spinSpeed+'°/s'; }; }

  const expb = el('export'); if (expb) expb.onclick = exportCSV;
  const regenBtn = el('regen'); if (regenBtn) regenBtn.onclick = ()=>{ rebuildBaseUV(); regenerate(); };
  const saveBtn = el('save'); if (saveBtn) saveBtn.onclick = ()=> saveCanvas('pca', 'png');
}

// ---------- drawing helpers ----------
function drawGrid(){
  push();
  background(255);
  stroke(0,0,0,16); strokeWeight(1);
  const step = 40;
  for (let x = ((0 - cam.tx) % step + step) % step; x < W; x += step) line(x, 0, x, H);
  for (let y = ((0 - cam.ty) % step + step) % step; y < H; y += step) line(0, y, W, y);
  pop();
}
function drawPCArrow(center, v, L, col){
  push();
  stroke(col); strokeWeight(3);
  const x0=center.x - v.x*L, y0=center.y - v.y*L;
  const x1=center.x + v.x*L, y1=center.y + v.y*L;
  line(x0,y0,x1,y1);
  // arrowheads
  const ah=8, ang=Math.atan2(v.y, v.x);
  const a1=ang+Math.PI*0.9, a2=ang-Math.PI*0.9;
  line(x1,y1, x1-Math.cos(a1)*ah, y1-Math.sin(a1)*ah);
  line(x1,y1, x1-Math.cos(a2)*ah, y1-Math.sin(a2)*ah);
  pop();
}

// ---------- p5 ----------
function setup(){
  pixelDensity(1);
  createCanvas(W, H);
  wireUI();
  rebuildBaseUV();
  regenerate();
}

function draw(){
  if (spin){
    thetaDeg = (thetaDeg + spinSpeed/60) % 180;
    const thetaS = el('theta'), thetaVal = el('thetaVal');
    if (thetaS) thetaS.value = String(Math.round(thetaDeg));
    if (thetaVal) thetaVal.textContent = Math.round(thetaDeg) + '°';
    regenerate();
  }

  push();
  translate(cam.tx, cam.ty);
  scale(cam.zoom, cam.zoom);

  drawGrid();

  const mu = mean2(pts);
  const S = cov2(pts, mu);
  const {vals, vecs} = eig2x2(S);

  // ---------- draw points (uses ptSize + colorMode) ----------
  const v1 = vecs[0];
  // projection range for pc1 coloring
  let minp=Infinity, maxp=-Infinity;
  for (const p of pts){ const t = (p.x - mu.x)*v1.x + (p.y - mu.y)*v1.y; if(t<minp)minp=t; if(t>maxp)maxp=t; }
  const denom = Math.max(1e-9, maxp - minp);

  noStroke();
  for (let i=0;i<pts.length;i++){
    const p = pts[i];
    let f;
    if (colorMode === 'pc1'){
      const t = ((p.x - mu.x)*v1.x + (p.y - mu.y)*v1.y - minp) / denom;
      const c1 = color(30,120,255), c2 = color(255,90,30);
      f = lerpColor(c1, c2, Math.max(0, Math.min(1, t)));
    } else if (colorMode === 'cluster' && clusters.length === pts.length && kClusters>0){
      const pal = [ color(30,120,255), color(0,180,120), color(255,150,0), color(200,0,120), color(80,80,80), color(140,90,255), color(0,160,255), color(255,80,80) ];
      f = pal[ (clusters[i] % pal.length + pal.length) % pal.length ];
    } else {
      f = color(60,60,60, 120);
    }
    fill(f);
    ellipse(p.x, p.y, ptSize, ptSize);
  }

  // mean crosshair
  stroke(60); strokeWeight(2);
  line(mu.x-6, mu.y, mu.x+6, mu.y);
  line(mu.x, mu.y-6, mu.x, mu.y+6);

  // axes
  if (showAxes){
    const L1 = Math.max(24, Math.sqrt(Math.max(0, vals[0])) * 8);
    const L2 = Math.max(24, Math.sqrt(Math.max(0, vals[1])) * 8);
    drawPCArrow(mu, vecs[0], L1, color(30,120,255));
    drawPCArrow(mu, vecs[1], L2, color(0,180,90));
  }

  // projections onto PC1
  if (showProj){
    const v1 = vecs[0];
    stroke(255, 200, 0, 180); strokeWeight(1.5);
    for (const p of pts){
      const t = dot2(sub2(p, mu), v1);
      const proj = add2(mu, mul2(v1, t));
      line(p.x, p.y, proj.x, proj.y);
      noStroke(); fill(255,200,0,200); ellipse(proj.x, proj.y, Math.max(2, ptSize-1), Math.max(2, ptSize-1));
      stroke(255, 200, 0, 180);
    }
  }

  // reconstruction from PC1 only
  if (showRecon){
    const v1 = vecs[0];
    noStroke(); fill(0,0,0,60);
    for (const p of pts){
      const t = dot2(sub2(p, mu), v1);
      const recon = add2(mu, mul2(v1, t));
      ellipse(recon.x, recon.y, ptSize, ptSize);
    }
  }

  pop(); // end camera

  // HUD λ pill
  const dimsEl = document.getElementById('dims');
  if (dimsEl){
    const e1 = nf(vals[0],1,1), e2 = nf(vals[1],1,1);
    dimsEl.textContent = `λ₁=${e1}, λ₂=${e2}`;
  }

  // Update variance bars (throttled)
  if (frameCount % 10 === 0 && typeof updateExplained === 'function') {
    let pc1, pc2;
    if (varMode === 'view'){
      const a = S.sxx, b = S.sxy, c = S.syy;
      const rad = radians(thetaDeg);
      const cs = Math.cos(rad), sn = Math.sin(rad);
      const varView = a*cs*cs + 2*b*sn*cs + c*sn*sn;
      const total = Math.max(1e-12, a + c);
      pc1 = Math.min(1, Math.max(0, varView / total));
      pc2 = 1 - pc1;
    } else {
      const sum = Math.max(1e-12, vals[0] + vals[1]);
      pc1 = vals[0] / sum; pc2 = 1 - pc1;
    }
    updateExplained(pc1, pc2);
  }
}

// ---------- export ----------
function exportCSV(){
  const mu = mean2(pts);
  const S = cov2(pts, mu);
  const {vals, vecs} = eig2x2(S);
  let csv = "i,x,y,pc1,pc2,cluster\n";
  for (let i=0;i<pts.length;i++){
    const p = pts[i];
    const dx = p.x - mu.x, dy = p.y - mu.y;
    const pc1 = dx*vecs[0].x + dy*vecs[0].y;
    const pc2 = dx*vecs[1].x + dy*vecs[1].y;
    const cl = (clusters.length===pts.length && kClusters>0) ? clusters[i] : -1;
    csv += `${i},${p.x},${p.y},${pc1},${pc2},${cl}\n`;
  }
  csv += `# eigvals,${vals[0]},${vals[1]}\n`;
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'pca_points.csv'; a.click();
}

// ---------- keyboard ----------
function keyPressed(){
  if (key==='s' || key==='S'){ saveCanvas('pca','png'); }
  if (key==='r' || key==='R'){ rebuildBaseUV(); regenerate(); }
  if (key==='g' || key==='G'){ showProj = !showProj; }
}

// ---------- mouse nav (optional; leave minimal to keep code focused) ----------
// Add pan/zoom later if desired.

