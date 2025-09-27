/* KNN visualizer with Chart Axes mode:
   - Points live in data space [0..100] for both features
   - Chart mode draws axes, ticks, labels; maps data <-> pixels
   - Works with Dots or Emoji themes; neighbors highlighted
*/

let canvasW = 900, canvasH = 560;

// UI-controlled (synced on setup)
let NUM_POINTS = 20;
let K = 3;
let NUM_CLASSES = 2;
let THEME = "dots";        // "dots" | "emoji"
let CHART_MODE = false;    // toggle via checkbox

// Colors & Emojis
const CLASS_COLORS = [
  [  0, 200,  90],
  [230,  65,  65],
  [ 55, 120, 255],
  [255, 190,  30],
  [160,  80, 220],
];
const CLASS_EMOJIS = ["🧙‍♂️","⚔️","🛡️","🏹","🗡️"];
const EMOJI_FONT = 'Apple Color Emoji, "Segoe UI Emoji", "Noto Color Emoji", "Twemoji Mozilla", system-ui, sans-serif';

// Grid (used only in non-chart mode, to keep chart clean)
let SHOW_GRID = true;
let GRID_STEP = 40;

// Chart paddings (in px)
const PAD = { left: 70, right: 30, top: 30, bottom: 60 };

let points = [];
let has2DShadow = false;

function showError(msg) {
  const el = document.getElementById('err');
  if (!el) return;
  el.textContent = msg;
  el.style.display = 'block';
}

// ---------- Data <-> Pixel mapping (chart area) ----------
function plotLeft()   { return PAD.left; }
function plotRight()  { return width - PAD.right; }
function plotTop()    { return PAD.top; }
function plotBottom() { return height - PAD.bottom; }

function dataToPxX(x) { return map(x, 0, 100, plotLeft(),  plotRight()); }
function dataToPxY(y) { return map(y, 0, 100, plotBottom(), plotTop()); } // invert Y
function pxToDataX(px){ return map(px, plotLeft(), plotRight(), 0, 100); }
function pxToDataY(py){ return map(py, plotBottom(), plotTop(), 0, 100); }

function mouseInPlot() {
  return mouseX >= plotLeft() && mouseX <= plotRight() &&
         mouseY >= plotTop()  && mouseY <= plotBottom();
}

// ---------- Draw axes, ticks, labels ----------
function drawAxes() {
  // axes
  stroke(80); strokeWeight(2);
  line(plotLeft(), plotTop(),    plotLeft(),  plotBottom()); // Y axis
  line(plotLeft(), plotBottom(), plotRight(), plotBottom()); // X axis

  // ticks every 20
  textFont('system-ui, -apple-system, Arial'); textSize(12); fill(60); noStroke();
  for (let v = 0; v <= 100; v += 20) {
    // X ticks
    const x = dataToPxX(v);
    stroke(200); strokeWeight(1);
    line(x, plotBottom(), x, plotBottom() - 6);
    noStroke(); textAlign(CENTER, TOP);
    text(v.toString(), x, plotBottom() + 8);

    // Y ticks
    const y = dataToPxY(v);
    stroke(200); strokeWeight(1);
    line(plotLeft(), y, plotLeft() + 6, y);
    noStroke(); textAlign(RIGHT, CENTER);
    text(v.toString(), plotLeft() - 8, y);
  }

  // axis labels
  noStroke(); fill(40); textSize(14);
  textAlign(CENTER, TOP);
  text('health', (plotLeft() + plotRight()) / 2, height - PAD.bottom + 32);

  push();
  translate(PAD.left - 42, (plotTop() + plotBottom()) / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, TOP);
  text('strength', 0, 0);
  pop();
}

// ---------- Point ----------
class Point {
  constructor() {
    // store in data space [0..100]
    this.dx = random(0, 100);
    this.dy = random(0, 100);
    this.classId = floor(random(NUM_CLASSES));
    this.isNeighbor = false;
  }

  // get drawing coordinates depending on mode
  get px() { return CHART_MODE ? dataToPxX(this.dx) : map(this.dx, 0, 100, 0, width); }
  get py() { return CHART_MODE ? dataToPxY(this.dy) : map(this.dy, 0, 100, 0, height); }

  display() {
    push();

    if (this.isNeighbor) {
      if (has2DShadow) { drawingContext.shadowBlur = 16; drawingContext.shadowColor = 'rgba(255,215,0,0.9)'; }
      stroke(255, 215, 0); strokeWeight(4);
    } else {
      if (has2DShadow) drawingContext.shadowBlur = 0;
      stroke(0); strokeWeight(2);
    }

    const x = this.px, y = this.py;

    if (THEME === "emoji") {
      // colored dot behind emoji (always visible)
      const col = CLASS_COLORS[this.classId % CLASS_COLORS.length] || [0,0,0];
      fill(col[0], col[1], col[2], 160);
      ellipse(x, y, 14, 14);

      // emoji on top
      noStroke();
      textAlign(CENTER, CENTER);
      textFont(EMOJI_FONT); textSize(20);
      const emo = CLASS_EMOJIS[this.classId % CLASS_EMOJIS.length];
      text(emo, x, y + 1);
    } else {
      const c = CLASS_COLORS[this.classId % CLASS_COLORS.length] || [0,0,0];
      fill(c[0], c[1], c[2]);
      ellipse(x, y, 16, 16);
    }

    pop();
    this.isNeighbor = false;
  }
}

// ---------- p5 lifecycle ----------
function setup() {
  try {
    createCanvas(canvasW, canvasH);
    has2DShadow = !!(window.drawingContext && 'shadowBlur' in drawingContext);
    wireUI();
    regeneratePoints();
  } catch (e) { console.error(e); showError('Setup error: ' + e.message); }
}

function draw() {
  try {
    background(255);

    if (CHART_MODE) {
      // clean chart, no background grid
      drawAxes();
    } else {
      // non-chart mode uses a subtle grid
      drawGrid();
    }

    for (const p of points) p.display();

    // classify and draw cursor if inside canvas/plot
    const canClassify = CHART_MODE ? mouseInPlot() : (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height);
    if (canClassify && points.length > 0) {
      // query in DATA space
      const qx = CHART_MODE ? pxToDataX(mouseX) : map(mouseX, 0, width, 0, 100);
      const qy = CHART_MODE ? pxToDataY(mouseY) : map(mouseY, 0, height, 0, 100);

      const { prediction, neighborIndices } = classify(qx, qy, K);

      // highlight neighbors
      for (const idx of neighborIndices) if (points[idx]) points[idx].isNeighbor = true;

      // draw cursor marker at pixel position
      const cx = CHART_MODE ? mouseX : map(qx, 0, 100, 0, width);
      const cy = CHART_MODE ? mouseY : map(qy, 0, 100, 0, height);
      const t = frameCount * 0.2;

      if (THEME === "emoji") {
        textAlign(CENTER, CENTER); textFont(EMOJI_FONT); textSize(18);
        const emo = CLASS_EMOJIS[prediction % CLASS_EMOJIS.length];
        noStroke(); text(emo, cx, cy);
        noFill(); stroke(0,0,0,40);
        const r = 16 + 2 * Math.sin(t); ellipse(cx, cy, r + 8, r + 8);
      } else {
        const c = CLASS_COLORS[prediction % CLASS_COLORS.length];
        noStroke(); fill(c[0], c[1], c[2]);
        const r = 10 + 2 * Math.sin(t); ellipse(cx, cy, r, r);
        noFill(); stroke(0,0,0,40); ellipse(cx, cy, r + 10, r + 10);
      }
    }
  } catch (e) { console.error(e); showError('Draw error: ' + e.message); }
}

// ---------- KNN in data space ----------
function classify(qx, qy, k) {
  const ds = []; // [distance^2, index, class]
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const dx = p.dx - qx, dy = p.dy - qy;
    const d2 = dx*dx + dy*dy;
    ds.push([d2, i, p.classId]);
  }
  ds.sort((a, b) => a[0] - b[0]);

  const neighbors = ds.slice(0, Math.min(k, ds.length));
  const votes = Array(Math.max(1, NUM_CLASSES)).fill(0);
  const neighborIndices = [];
  for (const [, idx, cls] of neighbors) {
    votes[cls] = (votes[cls] || 0) + 1;
    neighborIndices.push(idx);
  }

  let bestClass = 0, bestVote = -1;
  for (let c = 0; c < votes.length; c++) if (votes[c] > bestVote) { bestVote = votes[c]; bestClass = c; }
  return { prediction: bestClass, neighborIndices };
}

// ---------- Background grid (non-chart mode) ----------
function drawGrid(step = GRID_STEP) {
  if (!SHOW_GRID || CHART_MODE) return;
  stroke(230); strokeWeight(1);
  for (let x = 0; x <= width; x += step) line(x, 0, x, height);
  for (let y = 0; y <= height; y += step) line(0, y, width, y);
}

// ---------- UI wiring ----------
function regeneratePoints() {
  points = [];
  for (let i = 0; i < NUM_POINTS; i++) points.push(new Point());
}
function shuffleClasses() {
  for (const p of points) p.classId = floor(random(NUM_CLASSES));
}
function wireUI() {
  const pointsSlider  = document.getElementById("points");
  const kSlider       = document.getElementById("k");
  const classesSlider = document.getElementById("classes");
  const themeSel      = document.getElementById("theme");
  const chartChk      = document.getElementById("chart");

  const pointsVal  = document.getElementById("pointsVal");
  const kVal       = document.getElementById("kVal");
  const classesVal = document.getElementById("classesVal");

  const regenBtn   = document.getElementById("regen");
  const shuffleBtn = document.getElementById("shuffle");

  NUM_POINTS   = +pointsSlider.value;
  K            = +kSlider.value;
  NUM_CLASSES  = +classesSlider.value;
  THEME        = themeSel.value;
  CHART_MODE   = !!chartChk?.checked;

  kSlider.max = Math.max(1, NUM_POINTS);

  pointsVal.textContent  = NUM_POINTS;
  kVal.textContent       = K;
  classesVal.textContent = NUM_CLASSES;

  pointsSlider.oninput = (e) => {
    NUM_POINTS = +e.target.value;
    pointsVal.textContent = NUM_POINTS;
    kSlider.max = Math.max(1, NUM_POINTS);
    if (K > NUM_POINTS) { K = NUM_POINTS; kSlider.value = K; kVal.textContent = K; }
    regeneratePoints();
  };
  kSlider.oninput = (e) => {
    K = max(1, min(+e.target.value, points.length || 1));
    kVal.textContent = K;
  };
  classesSlider.oninput = (e) => {
    NUM_CLASSES = +e.target.value || 2;
    classesVal.textContent = NUM_CLASSES;
    shuffleClasses();
  };
  themeSel.onchange = (e) => { THEME = e.target.value; };
  if (chartChk) chartChk.onchange = (e) => { CHART_MODE = e.target.checked; };

  regenBtn.onclick = regeneratePoints;
  shuffleBtn.onclick = shuffleClasses;
}

// Keyboard shortcuts for K
function keyPressed() {
  const kSlider = document.getElementById("k");
  if (key === '[') {
    K = Math.max(1, K - 1);
    kSlider.value = K; document.getElementById("kVal").textContent = K;
  }
  if (key === ']') {
    K = Math.min(points.length, K + 1);
    kSlider.value = K; document.getElementById("kVal").textContent = K;
  }
}
