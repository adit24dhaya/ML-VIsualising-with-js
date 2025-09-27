// PCA Visualizer (2D) — safer eigen visuals, clean center, save hotkey.

let W = 960, H = 600;
let pts = [], N = 400;
let anis = 5;         // variance ratio major/minor
let thetaDeg = 30;    // rotation degrees
let showProj = true, showRecon = false;
let showAxes = true;
let spin = false, spinSpeed = 30; // deg/sec
let ptSize = 3;
let colorMode = 'none';
let kClusters = 0;
let clusters = []; // cluster id per point
let cam = {tx:0, ty:0, zoom:1};
let selRect = null; let selected = new Set();

function clamp01(x){ return Number.isFinite(x) ? Math.max(0, Math.min(1, x)) : 0; }

function setup(){
  pixelDensity(1);
  createCanvas(W, H);
  wireUI();
  regenerate();
}

function draw(){
  if (spin){ thetaDeg = (thetaDeg + spinSpeed/60) % 360; }
  background(255);
  drawGrid();

  // mean & covariance
  const mu = mean2(pts);
  const S = cov2(pts, mu);

  // eigen for symmetric 2x2
  const {vals, vecs} = eig2x2(S);
    // range along PC1 for color scale
    let minp=1e9, maxp=-1e9; for (const p of pts){ const t = p.x*vecs[0].x + p.y*vecs[0].y; if(t<minp)minp=t; if(t>maxp)maxp=t; }

  
// draw points
  const v1 = vecs[0];
  // range along PC1 for color scale
  let minp=Infinity, maxp=-Infinity;
  for (const p of pts){
    const t = (p.x - mu.x)*v1.x + (p.y - mu.y)*v1.y;
    if (t<minp) minp=t; if (t>maxp) maxp=t;
  }
  const denom = Math.max(1e-9, maxp - minp);
  noStroke();
  for (let i=0;i<pts.length;i++){
    const p = pts[i];
    let f = null;
    if (colorMode === 'pc1'){
      const t = ((p.x - mu.x)*v1.x + (p.y - mu.y)*v1.y - minp) / denom;
      const c1 = color(30,120,255), c2 = color(255,90,30);
      f = lerpColor(c1, c2, Math.max(0, Math.min(1, t)));
    } else if (colorMode === 'cluster' && clusters.length === pts.length && kClusters > 0){
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

  // principal axes (lengths clamped to avoid sqrt of tiny negatives)
  const L1 = Math.max(24, Math.sqrt(Math.max(0, vals[0])) * 8);
  const L2 = Math.max(24, Math.sqrt(Math.max(0, vals[1])) * 8);
  drawPCArrow(mu, vecs[0], L1, color(30,120,255));   // PC1
  drawPCArrow(mu, vecs[1], L2, color(0,180,90));     // PC2

  // projections onto PC1
  if (showProj){
    const v1 = vecs[0];
    stroke(255, 200, 0, 180); strokeWeight(1.5);
    for (const p of pts){
      const t = dot2(sub2(p, mu), v1);
      const proj = add2(mu, mul2(v1, t));
      line(p.x, p.y, proj.x, proj.y);
      noStroke(); fill(255,200,0,200); ellipse(proj.x, proj.y, 4.5, 4.5);
    }
  }

  // reconstruction from PC1 only
  if (showRecon){
    const v1 = vecs[0];
    noStroke(); fill(0,0,0,60);
    for (const p of pts){
      const t = dot2(sub2(p, mu), v1);
      const recon = add2(mu, mul2(v1, t));
      ellipse(recon.x, recon.y, 5.5, 5.5);
    }
  }

  // HUD eigenvalues
  const dimsEl = document.getElementById('dims');
  if (dimsEl){
    const e1 = nf(vals[0],1,1), e2 = nf(vals[1],1,1);
    dimsEl.textContent = `λ₁=${e1}, λ₂=${e2}`;
  }

  // Update variance bars (throttled)
  if (frameCount % 10 === 0 && typeof updateExplained === 'function') {
    let pc1, pc2;
    if (typeof varMode !== 'undefined' && varMode === 'view'){
      const a = S.sxx, b = S.sxy, c = S.syy;
      const rad = radians(thetaDeg);
      const cs = Math.cos(rad), sn = Math.sin(rad);
      const varView = a*cs*cs + 2*b*sn*cs + c*sn*sn;
      const total = Math.max(1e-12, a + c);
      pc1 = Math.min(1, Math.max(0, varView / total));
      pc2 = 1 - pc1;
    } else {
      const sum = Math.max(1e-12, vals[0] + vals[1]);
      pc1 = vals[0] / sum;
      pc2 = 1 - pc1;
    }
    updateExplained(pc1, pc2);
  }
}

// ---------- data ----------
function regenerate(){
  pts = [];
  const sMajor = anis * 10, sMinor = 10; // visual scale
  const cx = W * 0.5, cy = H * 0.5;
  const th = radians(thetaDeg);
  const ct = Math.cos(th), st = Math.sin(th);

  for (let i=0;i<N;i++){
    const u = randn() * sMajor;
    const v = randn() * sMinor;
    const x = u*ct - v*st + cx;
    const y = u*st + v*ct + cy;
    pts.push({x,y
  // keep clusters fresh if enabled
  if (kClusters > 0){
    if (typeof frameCount === 'number'){
      if (frameCount % 10 === 0){ const res = kmeans(pts, kClusters); clusters = res.labels; }
    } else {
      const res = kmeans(pts, kClusters); clusters = res.labels;
    }
  } else {
    clusters = [];
  }
});
  }
}

// ---------- UI ----------

function wireUI(){
  const nPts = el('nPts'), nVal = el('nPtsVal');
  if (nPts){
    nVal && (nVal.textContent = nPts.value);
    nPts.oninput = ()=>{ N = +nPts.value; nVal && (nVal.textContent = N); rebuildBaseUV(); regenerate(); };
  }

  const anisS = el('anis'), anisVal = el('anisVal');
  if (anisS){
    anisVal && (anisVal.textContent = anisS.value + '×');
    anisS.oninput = ()=>{ anis = +anisS.value; anisVal && (anisVal.textContent = anis + '×'); regenerate(); };
  }

  const thetaS = el('theta'), thetaVal = el('thetaVal');
  if (thetaS){
    thetaVal && (thetaVal.textContent = thetaS.value + '°');
    thetaS.oninput = ()=>{ thetaDeg = +thetaS.value; thetaVal && (thetaVal.textContent = thetaDeg + '°'); regenerate(); };
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
      if (kClusters>0){ const res = kmeans(pts, kClusters); clusters = res.labels; } else { clusters = []; }
    };
  }

  const ps = el('ptSize'), pv = el('ptVal');
  if (ps){
    pv && (pv.textContent = ps.value + 'px');
    ps.oninput = ()=>{ ptSize = +ps.value; pv && (pv.textContent = ptSize + 'px'); };
  }

  const cm = el('colorMode'); if (cm){ colorMode = cm.value; cm.onchange = ()=> colorMode = cm.value; }

  const spinC = el('spin'), spd = el('spinSpeed'), sVal = el('speedVal');
  function updateSpinUI(){
    if (spd) spd.disabled = !spinC.checked;
    if (sVal) sVal.textContent = (spinC.checked ? spinSpeed : 0) + '°/s';
  }
  if (spinC){
    spin = spinC.checked; updateSpinUI();
    spinC.onchange = ()=>{ spin = spinC.checked; updateSpinUI(); };
  }
  if (spd){
    spd.value = String(spinSpeed);
    spd.oninput = (e)=>{ spinSpeed = +e.target.value; if (sVal) sVal.textContent = spinSpeed + '°/s'; };
  }

  const expb = el('export'); if (expb) expb.onclick = exportCSV;
  const regen = el('regen'); if (regen) regen.onclick = ()=>{ rebuildBaseUV(); regenerate(); };
  const saveBtn = el('save'); if (saveBtn) saveBtn.onclick = ()=> saveCanvas('pca', 'png');

  const varSel = el('varMode');
  if (varSel){ varMode = varSel.value; varSel.onchange = ()=>{ varMode = varSel.value; }; }
}
// ---------- math helpers ----------
function mean2(A){ let sx=0, sy=0; for (const p of A){ sx+=p.x; sy+=p.y; } const n=Math.max(1,A.length); return {x:sx/n,y:sy/n}; }
function cov2(A, mu){
  let sxx=0, sxy=0, syy=0; const n=A.length;
  for (const p of A){ const dx=p.x-mu.x, dy=p.y-mu.y; sxx+=dx*dx; sxy+=dx*dy; syy+=dy*dy; }
  const c = 1/Math.max(1, n-1); return {sxx:sxx*c, sxy:sxy*c, syy:syy*c};
}
// eigen for symmetric 2x2 [[a,b],[b,c]]
function eig2x2(S){
  const a=S.sxx, b=S.sxy, c=S.syy;
  const tr=a+c, det=a*c-b*b;
  const disc = Math.sqrt(Math.max(0, (tr*tr)/4 - det));
  const l1 = tr/2 + disc, l2 = tr/2 - disc;
  function vecFor(l){
    if (Math.abs(b) > 1e-12) { const vx=b, vy=l-a; const n=Math.hypot(vx,vy)||1; return {x:vx/n,y:vy/n}; }
    return (a>=c) ? {x:1,y:0} : {x:0,y:1};
  }
  return { vals:[l1,l2], vecs:[vecFor(l1), vecFor(l2)] };
}
function add2(a,b){ return {x:a.x+b.x, y:a.y+b.y}; }
function sub2(a,b){ return {x:a.x-b.x, y:a.y-b.y}; }
function mul2(v,s){ return {x:v.x*s, y:v.y*s}; }
function dot2(a,b){ return a.x*b.x + a.y*b.y; }

// ---------- drawing ----------
function drawPCArrow(center, v, len, col){
  push(); stroke(col); strokeWeight(3);
  const a = {x:center.x - v.x*len, y:center.y - v.y*len};
  const b = {x:center.x + v.x*len, y:center.y + v.y*len};
  line(a.x, a.y, b.x, b.y);
  arrowHead(b, v, col); arrowHead(a, {x:-v.x, y:-v.y}, col);
  pop();
}
function arrowHead(p, dir, col){
  const s=9, ang=Math.atan2(dir.y, dir.x);
  push(); translate(p.x,p.y); rotate(ang); noStroke(); fill(col);
  triangle(0,0, -s,  s*0.5, -s, -s*0.5); pop();
}

function drawGrid(step=40){
  stroke(235); strokeWeight(1);
  for (let x=0;x<=width;x+=step) line(x,0,x,height);
  for (let y=0;y<=height;y+=step) line(0,y,width,y);
}

// ---------- utils ----------
function el(id){ return document.getElementById(id); }
function randn(){ let u=0,v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
function keyPressed(){ if (key==='s' || key==='S') saveCanvas('pca', 'png'); }

// ---------- kmeans clustering ----------
function kmeans(points, k, iters=8){
  if (k<=0) return {centroids:[], labels:Array(points.length).fill(-1)};
  // init centroids by sampling
  const centroids = [];
  for (let i=0;i<k;i++){ const p = points[Math.floor(Math.random()*points.length)]; centroids.push({x:p.x,y:p.y}); }
  const labels = new Array(points.length).fill(0);
  for (let iter=0; iter<iters; iter++){
    // assign
    for (let i=0;i<points.length;i++){
      let bi=0, bd=1e9;
      for (let c=0;c<k;c++){
        const dx = points[i].x - centroids[c].x, dy = points[i].y - centroids[c].y;
        const d = dx*dx + dy*dy;
        if (d<bd){ bd=d; bi=c; }
      }
      labels[i]=bi;
    }
    // update
    const sum = Array.from({length:k}, ()=>({x:0,y:0,n:0}));
    for (let i=0;i<points.length;i++){ const c=labels[i]; sum[c].x+=points[i].x; sum[c].y+=points[i].y; sum[c].n++; }
    for (let c=0;c<k;c++){
      if (sum[c].n>0){ centroids[c].x=sum[c].x/sum[c].n; centroids[c].y=sum[c].y/sum[c].n; }
    }
  }
  return {centroids, labels};
}

// ---------- export CSV ----------
function exportCSV(){
  let csv = "x,y\\n";
  for (const p of pts) csv += `${p.x},${p.y}\\n`;
  const blob = new Blob([csv], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'points.csv';
  document.body.appendChild(a); a.click(); a.remove();
}

// ---------- colors ----------
const CLUSTER_COLS = [
  [239,83,80],[66,165,245],[102,187,106],[255,202,40],
  [171,71,188],[255,167,38],[38,166,154],[156,204,101]
];
function colForCluster(id){ const c = CLUSTER_COLS[id % CLUSTER_COLS.length]; return color(c[0],c[1],c[2]); }

function drawPointsEnhanced(pc){
  noStroke();
  for (let i=0;i<pts.length;i++){
    const p = pts[i];
    let col = color(60,60,60,200);
    if (colorMode==='pc1'){
      // color by projection onto PC1 mapped to 0..1
      const t = (p.x*pc.v1.x + p.y*pc.v1.y - pc.meanProjMin) / (pc.meanProjMax - pc.meanProjMin + 1e-6);
      col = color(50 + 205*t, 140, 255 - 205*t, 180);
    } else if (colorMode==='cluster' && clusters.length===pts.length && clusters[i]>=0){
      col = colForCluster(clusters[i]); col.setAlpha(200);
    }
    fill(col);
    circle(p.x, p.y, ptSize);
  }
}

function drawAxesEnhanced(mean, vecs, vals){
  if (!showAxes) return;
  push(); stroke(0,0,0,80); strokeWeight(2);
  const scale = 2 * Math.sqrt(Math.max(vals[0],0));
  line(mean.x - vecs[0].x*scale, mean.y - vecs[0].y*scale, mean.x + vecs[0].x*scale, mean.y + vecs[0].y*scale);
  stroke(0,0,0,50);
  const scale2 = 2 * Math.sqrt(Math.max(vals[1],0));
  line(mean.x - vecs[1].x*scale2, mean.y - vecs[1].y*scale2, mean.x + vecs[1].x*scale2, mean.y + vecs[1].y*scale2);
  pop();

  // overlay selection rectangle
  if (selRect){
    push(); noFill(); stroke(30,144,255,180); strokeWeight(2);
    rect(selRect.x, selRect.y, selRect.w, selRect.h);
    pop();
  }
  
}
function drawCovEllipse(S, vecs, vals){
  const mean = {x:S.mx, y:S.my};
  const kList = [1, 2]; // 1σ and 2σ
  for (let i=0;i<kList.length;i++){
    const k = kList[i];
    const a = k*Math.sqrt(Math.max(vals[0],0)) * 4;
    const b = k*Math.sqrt(Math.max(vals[1],0)) * 4;
    const angle = Math.atan2(vecs[0].y, vecs[0].x);
    push();
    translate(mean.x, mean.y);
    rotate(angle);
    noFill();
    stroke(50, 120, 255, i ? 80 : 130);
    strokeWeight(1.5);
    ellipse(0, 0, 2*a, 2*b);
    pop();
  }
}
