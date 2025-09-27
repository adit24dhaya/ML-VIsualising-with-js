let W = 900, H = 560;
let pts = [], N = 200;
let anis = 5;        // variance ratio major/minor
let thetaDeg = 30;   // rotation degrees
let showProj = true, showRecon = false;

// convenience
function toRad(deg){ return deg*Math.PI/180; }

function setup(){
  createCanvas(W, H);
  wireUI();
  regenerate();
}

function draw(){
  background(255);
  drawGrid();

  // compute mean
  const mu = mean2(pts);

  // covariance (2x2)
  const S = cov2(pts, mu);

  // eigen-decomposition for 2x2 analytically
  const {vals, vecs} = eig2x2(S);  // vals[0]>=vals[1], vecs columns

  // draw data
  noStroke();
  fill(60, 120);
  for (const p of pts){
    ellipse(p.x, p.y, 6, 6);
  }

  // draw mean
  stroke(60); strokeWeight(2);
  line(mu.x-6, mu.y, mu.x+6, mu.y);
  line(mu.x, mu.y-6, mu.x, mu.y+6);

  // principal axes arrows
  drawPCArrow(mu, vecs[0], Math.sqrt(vals[0])*8, color(30,120,255)); // PC1
  drawPCArrow(mu, vecs[1], Math.sqrt(vals[1])*8, color(0,180,90));   // PC2

  // projections onto PC1
  if (showProj){
    const v1 = vecs[0];
    stroke(255, 200, 0, 180); strokeWeight(1.5);
    for (const p of pts){
      const t = dot2(sub2(p, mu), v1); // scalar along PC1
      const proj = add2(mu, mul2(v1, t));
      line(p.x, p.y, proj.x, proj.y);
      noStroke(); fill(255,200,0,200);
      ellipse(proj.x, proj.y, 4.5, 4.5);
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

  // HUD info
  const dimsEl = document.getElementById('dims');
  if (dimsEl){
    const e1 = nf(vals[0],1,1), e2 = nf(vals[1],1,1);
    dimsEl.textContent = `λ₁=${e1}, λ₂=${e2}`;
  }
}

// -------------- data generation --------------
function regenerate(){
  pts = [];
  // axis-aligned Gaussian first: major variance = anis, minor = 1 (scaled)
  const sMajor = anis*10, sMinor = 10; // scale for visibility
  const cx = W * 0.5, cy = H * 0.5;    // ⟵ center exactly
  const th = toRad(thetaDeg);

  for (let i=0;i<N;i++){
    // sample (u,v) from N(0, sMajor^2) and N(0, sMinor^2)
    const u = randn()*sMajor, v = randn()*sMinor;
    // rotate by theta, center at (cx, cy)
    const x = u*Math.cos(th) - v*Math.sin(th) + cx;
    const y = u*Math.sin(th) + v*Math.cos(th) + cy;
    pts.push({x,y});
  }
}

function wireUI(){
  const nPts = sel('nPts'), nPtsVal = sel('nPtsVal');
  const anisS = sel('anis'), anisVal = sel('anisVal');
  const theta = sel('theta'), thetaVal = sel('thetaVal');
  const proj  = sel('showProj'), recon = sel('recon');
  const regen = sel('regen');

  N = +nPts.value; nPts.oninput = e=>{ N=+e.target.value; nPtsVal.textContent=N; regenerate(); };
  anis = +anisS.value; anisS.oninput = e=>{ anis=+e.target.value; anisVal.textContent=anis+'×'; regenerate(); };
  thetaDeg = +theta.value; theta.oninput = e=>{ thetaDeg=+e.target.value; thetaVal.textContent=thetaDeg+'°'; regenerate(); };

  showProj = proj.checked; proj.onchange = e=> showProj = e.target.checked;
  showRecon = recon.checked; recon.onchange = e=> showRecon = e.target.checked;

  regen.onclick = regenerate;

  // initial labels
  nPtsVal.textContent = N;
  anisVal.textContent = anis+'×';
  thetaVal.textContent = thetaDeg+'°';
}

// -------------- math helpers --------------
function mean2(A){
  let sx=0, sy=0; for (const p of A){ sx+=p.x; sy+=p.y; }
  const n = max(1,A.length); return {x:sx/n, y:sy/n};
}
function cov2(A, mu){
  let sxx=0, sxy=0, syy=0; const n=A.length;
  for (const p of A){
    const dx=p.x-mu.x, dy=p.y-mu.y;
    sxx += dx*dx; sxy += dx*dy; syy += dy*dy;
  }
  const c = 1/max(1, n-1);
  return { sxx:sxx*c, sxy:sxy*c, syy:syy*c };
}
function eig2x2(S){
  const a=S.sxx, b=S.sxy, c=S.syy;
  const tr = a+c, det = a*c - b*b;
  const disc = Math.sqrt(max(0, tr*tr/4 - det));
  const l1 = tr/2 + disc, l2 = tr/2 - disc;
  function vecFor(l){
    if (Math.abs(b) > 1e-12) {
      const v = {x:b, y:(l - a)};
      const n = Math.hypot(v.x, v.y) || 1; return {x:v.x/n, y:v.y/n};
    } else {
      return (a >= c) ? {x:1,y:0} : {x:0,y:1};
    }
  }
  let v1 = vecFor(l1), v2 = vecFor(l2);
  return { vals:[l1,l2], vecs:[v1,v2] };
}
function add2(a,b){ return {x:a.x+b.x, y:a.y+b.y}; }
function sub2(a,b){ return {x:a.x-b.x, y:a.y-b.y}; }
function mul2(v,s){ return {x:v.x*s, y:v.y*s}; }
function dot2(a,b){ return a.x*b.x + a.y*b.y; }
function drawPCArrow(center, v, len, col){
  push();
  stroke(col); strokeWeight(3);
  const a = {x:center.x - v.x*len, y:center.y - v.y*len};
  const b = {x:center.x + v.x*len, y:center.y + v.y*len};
  line(a.x, a.y, b.x, b.y);
  drawArrowHead(b, v, col);
  drawArrowHead(a, {x:-v.x, y:-v.y}, col);
  pop();
}
function drawArrowHead(p, dir, col){
  const s=9;
  const angle = Math.atan2(dir.y, dir.x);
  push(); translate(p.x, p.y); rotate(angle);
  fill(col); noStroke(); triangle(0,0, -s,  s*0.5, -s, -s*0.5); pop();
}
function sel(id){ return document.getElementById(id); }
function drawGrid(step=40){
  stroke(235); strokeWeight(1);
  for (let x=0;x<=width;x+=step) line(x,0,x,height);
  for (let y=0;y<=height;y+=step) line(0,y,width,y);
}
function randn(){
  let u=0, v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}
