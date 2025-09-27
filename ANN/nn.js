// ANN Forward-Pass Visualizer — complete self-contained sketch (p5.js)

/* ---------------- Geometry / Layout --------------- */
let W = 960, H = 600;
function computeSize() {
  const pad = 48;
  const stage = document.getElementById('stage');
  const maxW = Math.min(stage ? stage.clientWidth : window.innerWidth, 1280);
  W = Math.max(600, maxW - pad);
  H = Math.max(520, Math.round(W * 0.62));
}

/* ---------------- Network Params ------------------ */
let HIDDEN = 3;
let actName = 'relu';
let noisePct = 0;       // 0..100
let showHeat = false;
let showTrail = false;
let smoothK = 0.20; // EMA factor for display (0..0.95)
let holdNoise = false;
let partyMode = false;
let showStats = false;
let yHist = []; // entries: {t, y:[y1,y2]}

// weights
let W1, b1, W2, b2;

// activations for drawing
let x_in = [0.25, 0.25];
let h_act = [];
let y_out = [];
let y_disp = [0,0];

// positions for nodes
let pos = { inputs: [], hidden: [], outputs: [] };

function initWeights() {
  // Xavier-ish init
  const randn = () => (Math.random() * 2 - 1);
  W1 = Array.from({length: HIDDEN}, _ => [randn()*0.8, randn()*0.8]);
  b1 = Array.from({length: HIDDEN}, _ => randn()*0.1);
  W2 = Array.from({length: 2}, _ => Array.from({length: HIDDEN}, __ => randn()*0.8));
  b2 = [randn()*0.1, randn()*0.1];
}

function setHidden(k){ HIDDEN = Math.max(2, Math.min(64, k|0)); initWeights(); layoutNodes(); }
function setActivation(name){ actName = name; }
function setNoise(v){ noisePct = Math.max(0, Math.min(100, v)); }
function setHeatmap(on){ showHeat = !!on; }
function reinitWeights(){ initWeights(); }

/* ---------------- Activation funcs ---------------- */
function activate(v, name){
  if(name==='tanh') return Math.tanh(v);
  if(name==='sigmoid') return 1/(1+Math.exp(-v));
  // relu
  return Math.max(0, v);
}

/* ---------------- Layout nodes -------------------- */
function layoutNodes(){
  pos.inputs = [];
  pos.hidden = [];
  pos.outputs = [];

  const leftX = W*0.18, rightX = W*0.82, centerX = W*0.5;
  const topY = H*0.18, botY = H*0.82;

  // inputs: two nodes left vertical
  pos.inputs.push({x:leftX, y: topY, label:'x₁ (mouse x)'});
  pos.inputs.push({x:leftX, y: botY, label:'x₂ (mouse y)'});

  // hidden: centered column
  const hTop = H*0.28, hBot = H*0.72;
  for(let i=0;i<HIDDEN;i++){
    const yy = HIDDEN===1 ? (H*0.5) : (hTop + (hBot-hTop)*i/(HIDDEN-1));
    pos.hidden.push({x:centerX, y: yy});
  }

  // outputs: two nodes on right vertical
  pos.outputs.push({x:rightX, y: topY, label:'y₁'});
  pos.outputs.push({x:rightX, y: botY, label:'y₂'});
}

/* ---------------- Forward pass -------------------- */
function forward(xy){
  // noise on inputs
  const std = noisePct/100 * 0.25;
  // update persistent noise only when mouse moves (to avoid twitch)
  const moved = Math.hypot(mouseX - prevMouse.x, mouseY - prevMouse.y) > 1;
  if (!holdNoise) {
    noiseOffset = [randomGaussian(0, std||0), randomGaussian(0, std||0)];
  } else if (moved) {
    noiseOffset = [randomGaussian(0, std||0), randomGaussian(0, std||0)];
    prevMouse.x = mouseX; prevMouse.y = mouseY;
  }
  const n0 = (std ? xy[0] + noiseOffset[0] : xy[0]);
  const n1 = (std ? xy[1] + noiseOffset[1] : xy[1]);
  const x = [constrain(n0, 0, 1), constrain(n1, 0, 1)];

  // hidden pre-acts
  const z1 = new Array(HIDDEN).fill(0);
  for(let i=0;i<HIDDEN;i++){
    z1[i] = W1[i][0]*x[0] + W1[i][1]*x[1] + b1[i];
  }
  h_act = z1.map(v => activate(v, actName));

  // outputs
  const z2 = [0, 0];
  for(let o=0;o<2;o++){
    let sum = b2[o];
    for(let i=0;i<HIDDEN;i++) sum += W2[o][i] * h_act[i];
    z2[o] = sum;
  }
  y_out = [activate(z2[0], actName), activate(z2[1], actName)];
  return {x, h:h_act, y:y_out, z1, z2};
}

/* ---------------- Helpers for drawing ------------- */
function drawGrid(){
  push();
  stroke(230); strokeWeight(1);
  const cell = 24;
  for(let x=0;x<W;x+=cell){ line(x, 0, x, H); }
  for(let y=0;y<H;y+=cell){ line(0, y, W, y); }
  pop();
}

function node(x,y, r, fillv, label){
  noStroke();
  fill(fillv);
  circle(x,y,r);
  if(label){
    fill(60); noStroke(); textSize(12); textAlign(CENTER, BOTTOM);
    text(label, x, y-14);
  }
}

function edge(x1,y1, x2,y2, w){
  const s = Math.min(7, 1 + Math.abs(w)*6);
  const alpha = Math.min(255, 80 + Math.abs(w)*140);
  stroke(w>=0 ? color(0,120,255, alpha) : color(0,0,0, alpha)); // positive = blue, negative = dark
  strokeWeight(s);
  line(x1,y1,x2,y2);
}

function drawHeat(){
  // coarse heatmap (grid 30x18)
  const cols = 30, rows = 18;
  const cw = W/cols, ch = H/rows;
  noStroke();
  for(let i=0;i<cols;i++){
    for(let j=0;j<rows;j++){
      const xx = (i+0.5)/cols;
      const yy = (j+0.5)/rows;
      const {y} = forward([xx,yy]);
      const v = (y[0] - y[1]) * 0.5 + 0.5; // map difference to 0..1
      const a = 80;
      fill(255*v, 120*(1-v), 255*(1-v), a);
      rect(i*cw, j*ch, cw+1, ch+1);
    }
  }
}

// trail + particles
let mouseTrace = []; const TRACE_MAX = 140;
let particles = [];
let prevMouse = {x:-999, y:-999};
let noiseOffset = [0,0];
function spawnParticle(x,y, strength){
  particles.push({
    x, y,
    vx: (Math.random()*2-1)*2.2,
    vy: (Math.random()*2-1)*2.2 - 1.5,
    life: 30 + Math.floor(30*strength),
    age: 0
  });
}
function updateParticles(){
  for (let i=particles.length-1;i>=0;i--){
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.04; // gravity
    p.age++;
    if (p.age > p.life) particles.splice(i,1);
  }
}
function drawParticles(){
  noStroke();
  for (const p of particles){
    const a = map(p.life - p.age, 0, p.life, 0, 180);
    fill(255, 200, 80, a);
    circle(p.x, p.y, 3);
  }
}

/* ---------------- p5 lifecycle ------------------- */
function setup(){
  computeSize();
  const c = createCanvas(W,H);
  (document.getElementById('stage') || document.body).appendChild(c.elt);
  textFont('ui-sans-serif, system-ui, -apple-system, Arial');
  initWeights();
  layoutNodes();
  bindUI();
  // setup complete
}

function windowResized(){
  computeSize();
  resizeCanvas(W,H);
  layoutNodes();
}

function draw(){
  background(255);
  if(showHeat) drawHeat();
  drawGrid();

  // read mouse as x,y in [0,1] when inside canvas
  let mx = constrain(map(mouseX, 0, width, 0, 1), 0, 1);
  let my = constrain(map(mouseY, 0, height, 0, 1), 0, 1);
  const inside = (mouseX>=0 && mouseX<=width && mouseY>=0 && mouseY<=height);
  if(!inside) { mx=x_in[0]; my=x_in[1]; }
  x_in = [mx, my];

  // trail
  if (inside && showTrail){
    mouseTrace.push({x:mouseX, y:mouseY});
    if (mouseTrace.length>TRACE_MAX) mouseTrace.shift();
  }
  if (showTrail && mouseTrace.length){
    noFill(); stroke(120,140,255,120); strokeWeight(2);
    beginShape();
    for (const p of mouseTrace) vertex(p.x, p.y);
    endShape();
  }

  const pass = forward(x_in);

  // party particles from strong hidden activations
  if (partyMode){
    for (let i=0;i<HIDDEN;i++){
      const v = constrain(h_act[i], 0, 1);
      if (v > 0.9 && frameCount % 2 === 0){
        const h = pos.hidden[i];
        spawnParticle(h.x, h.y, v);
      }
    }
    updateParticles();
    drawParticles();
  }

  // edges: inputs -> hidden
  for(let i=0;i<HIDDEN;i++){
    const h = pos.hidden[i];
    edge(pos.inputs[0].x, pos.inputs[0].y, h.x, h.y, W1[i][0]*x_in[0]);
    edge(pos.inputs[1].x, pos.inputs[1].y, h.x, h.y, W1[i][1]*x_in[1]);
  }
  // edges: hidden -> outputs
  for(let o=0;o<2;o++){
    const out = pos.outputs[o];
    for(let i=0;i<HIDDEN;i++){
      const h = pos.hidden[i];
      edge(h.x, h.y, out.x, out.y, W2[o][i]*h_act[i]);
    }
  }

  // nodes
  node(pos.inputs[0].x, pos.inputs[0].y, 20+12*x_in[0], color(120,170,255,180), pos.inputs[0].label);
  node(pos.inputs[1].x, pos.inputs[1].y, 20+12*x_in[1], color(120,170,255,180), pos.inputs[1].label);

  for(let i=0;i<HIDDEN;i++){
    const v = constrain(h_act[i], 0, 1);
    node(pos.hidden[i].x, pos.hidden[i].y, 14+10*v, color(90, 200, 120, 180));
  }

  // output nodes with numeric labels
  node(pos.outputs[0].x, pos.outputs[0].y, 16+12*y_out[0], color(60,60,60,200), 'y₁');
  node(pos.outputs[1].x, pos.outputs[1].y, 16+12*y_out[1], color(60,60,60,200), 'y₂');

  fill(30); noStroke(); textAlign(CENTER, TOP); textSize(12);
  y_disp[0] = (1 - smoothK) * y_disp[0] + smoothK * y_out[0];
  y_disp[1] = (1 - smoothK) * y_disp[1] + smoothK * y_out[1];
  text(y_disp[0].toFixed(3), pos.outputs[0].x, pos.outputs[0].y + 14);
  text(y_disp[1].toFixed(3), pos.outputs[1].x, pos.outputs[1].y + 14);

  // --- rolling stats over ~2s ---
  const now = millis();
  yHist.push({t: now, y: [y_out[0], y_out[1]]});
  const cutoff = now - 2000; // 2 seconds
  while (yHist.length && yHist[0].t < cutoff) yHist.shift();

  if (showStats && yHist.length){
    const n = yHist.length;
    // compute mean/std for each output
    const mean = [0,0], std = [0,0];
    for (const e of yHist){ mean[0]+=e.y[0]; mean[1]+=e.y[1]; }
    mean[0]/=n; mean[1]/=n;
    for (const e of yHist){ std[0]+= (e.y[0]-mean[0])**2; std[1]+= (e.y[1]-mean[1])**2; }
    std[0] = Math.sqrt(std[0]/Math.max(1,n-1));
    std[1] = Math.sqrt(std[1]/Math.max(1,n-1));

    // draw small badges near outputs
    const pad = 6, r = 6;
    const badge = (x,y, m,s) => {
      const txt = `μ=${m.toFixed(3)}  σ=${s.toFixed(3)}`;
      const w = textWidth(txt) + pad*2, h = 20;
      noStroke(); fill(255,255,255,230); rect(x - w/2, y + 32, w, h, r);
      fill(40); textAlign(CENTER, CENTER); textSize(12); text(txt, x, y + 32 + h/2);
    };
    badge(pos.outputs[0].x, pos.outputs[0].y, mean[0], std[0]);
    badge(pos.outputs[1].x, pos.outputs[1].y, mean[1], std[1]);
  }

  // crosshair + tooltip for inputs
  if (inside){
    stroke(0,0,0,40); strokeWeight(1); line(mouseX, 0, mouseX, height); line(0, mouseY, width, mouseY);
    const tip = `x₁=${x_in[0].toFixed(3)}  x₂=${x_in[1].toFixed(3)}`;
    const tw = textWidth(tip) + 12, th = 20;
    const tx = Math.min(width-tw-8, mouseX+12), ty = Math.min(height-th-8, mouseY+12);
    noStroke(); fill(255,255,255,220); rect(tx, ty, tw, th, 8);
    fill(40); textAlign(LEFT, CENTER); textSize(12); text(tip, tx+6, ty+th/2);
  }
}

/* ---------------- UI wiring ---------------------- */
function bindUI(){
  const hsize = document.getElementById('hsize');
  const hval  = document.getElementById('hval');
  const act   = document.getElementById('act');
  const noise = document.getElementById('noise');
  const nval  = document.getElementById('nval');
  const heat  = document.getElementById('heat');
  const smooth= document.getElementById('smooth');
  const sval  = document.getElementById('sval');
  const hold  = document.getElementById('hold');
  const trail = document.getElementById('trail');
  const party = document.getElementById('party');
  const stats = document.getElementById('stats');

  const re = document.getElementById('reinit');
  const sv = document.getElementById('save');

  if (hsize){ hsize.oninput = () => { hval.textContent = hsize.value; setHidden(parseInt(hsize.value,10)); }; }
  if (act){ act.onchange  = () => setActivation(act.value); }
  if (noise){ noise.oninput = () => { nval.textContent = noise.value; setNoise(parseInt(noise.value,10)); }; }
  if (heat){ heat.onchange = () => setHeatmap(heat.checked); }
  if (smooth){ smooth.oninput = () => { smoothK = Math.min(0.95, Math.max(0, parseInt(smooth.value,10)/100)); if(sval) sval.textContent = smoothK.toFixed(2); }; }
  if (hold){ hold.onchange = () => (holdNoise = hold.checked); }
  if (trail){ trail.onchange = () => (showTrail = trail.checked); }
  if (party){ party.onchange = () => (partyMode = party.checked); }
  if (stats){ stats.onchange = () => (showStats = stats.checked); }

  if (re) re.onclick = reinitWeights;
  if (sv) sv.onclick = () => saveCanvas('nn-forward-pass', 'png');
}

function keyPressed(){ if (key==='s' || key==='S') saveCanvas('nn-forward-pass', 'png'); }
