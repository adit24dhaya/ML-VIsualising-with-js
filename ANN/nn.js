// Neural network forward-pass visualizer
// Architecture: 2 -> H -> 2. We draw nodes in columns and edges with thickness/alpha ~ |activation|.
// Input comes from mouse position normalized to [0,1]^2.

let W = 900, H = 560;
let HSIZE = 3;
let ACT = 'relu';
let NOISE = 0;

let W1, b1, W2, b2;   // parameters
let layout;           // node positions for drawing

function setup(){
  createCanvas(W, H);
  wireUI();
  initParams();
  layout = computeLayout(HSIZE);
}

function draw(){
  background(255);
  drawGrid();

  // input from mouse in [0,1]
  const mx = constrain(map(mouseX, 0, width, 0, 1), 0, 1);
  const my = constrain(map(mouseY, 0, height, 0, 1), 0, 1);
  const x = [mx, my];

  // forward pass
  const z1 = addVec(matVec(W1, x), b1);          // pre-activation
  const a1 = activate(z1, ACT);                  // hidden activation
  const z2 = addVec(matVec(W2, a1), b2);         // pre-activation at outputs
  const y  = activate(z2, 'sigmoid');            // output (0..1) for visualization

  // optional noise (for motion)
  const n = NOISE/100;
  for (let i=0;i<a1.length;i++) a1[i] += (Math.random()*2-1)*n*0.05;

  // draw network
  drawNetwork(layout, x, z1, a1, z2, y);
}

function wireUI(){
  const hsize = sel('hsize'), hval = sel('hval');
  const act   = sel('act');
  const reinit= sel('reinit');
  const noise = sel('noise'), nval = sel('nval');

  HSIZE = +hsize.value; hval.textContent = HSIZE;
  ACT = act.value;
  NOISE = +noise.value; nval.textContent = NOISE;

  hsize.oninput = e => {
    HSIZE = +e.target.value; hval.textContent = HSIZE;
    layout = computeLayout(HSIZE);
    initParams();
  };
  act.onchange = e => { ACT = e.target.value; };
  reinit.onclick = () => initParams();
  noise.oninput = e => { NOISE = +e.target.value; nval.textContent = NOISE; };
}

// --------------- params & math ---------------
function initParams(){
  // W1: H x 2, b1: H; W2: 2 x H, b2: 2
  W1 = randMat(HSIZE, 2, 0.8);
  b1 = randVec(HSIZE, 0.2);
  W2 = randMat(2, HSIZE, 0.8);
  b2 = randVec(2, 0.2);
}

function activate(v, name){
  const out = new Array(v.length);
  for (let i=0;i<v.length;i++){
    const z = v[i];
    if (name==='relu') out[i] = Math.max(0, z);
    else if (name==='tanh') out[i] = Math.tanh(z);
    else if (name==='sigmoid') out[i] = 1/(1+Math.exp(-z));
    else out[i] = z;
  }
  return out;
}

function matVec(M, v){
  const r = M.length, c = M[0].length, out = new Array(r).fill(0);
  for (let i=0;i<r;i++){
    let s=0; for (let j=0;j<c;j++) s += M[i][j]*v[j];
    out[i]=s;
  }
  return out;
}
function addVec(a,b){ const o=new Array(a.length); for(let i=0;i<a.length;i++) o[i]=a[i]+b[i]; return o; }

function randMat(r,c,scale){
  const M = new Array(r);
  for (let i=0;i<r;i++){ M[i]=new Array(c); for (let j=0;j<c;j++) M[i][j]=(Math.random()*2-1)*scale; }
  return M;
}
function randVec(n,scale){ const v=new Array(n); for(let i=0;i<n;i++) v[i]=(Math.random()*2-1)*scale; return v; }

// --------------- layout & drawing ---------------
function computeLayout(H){
  const left = 170, right = W-170;
  const cols = [
    {x:left,        ys: spaced(2, 120, H*0+1) }, // input column baseline
    {x:W/2,         ys: spaced(H, 80, 1)     },  // hidden
    {x:right,       ys: spaced(2, 120, H*0+1)}   // output
  ];
  // For inputs/outputs we want exactly 2 nodes vertically centered
  cols[0].ys = fixedSlots(2, H/2);
  cols[2].ys = fixedSlots(2, H/2);
  return cols;
}

function spaced(n, gap, scale){
  const start = H/2 - (n-1)*gap/2;
  const ys=[]; for(let i=0;i<n;i++) ys.push(start+i*gap*scale); return ys;
}
function fixedSlots(n, cy){
  const gap = 120;
  const start = cy - (n-1)*gap/2;
  const ys=[]; for(let i=0;i<n;i++) ys.push(start+i*gap); return ys;
}

function drawNetwork(cols, x, z1, a1, z2, y){
  // node positions
  const inX = cols[0].x, hidX = cols[1].x, outX = cols[2].x;
  const inY = cols[0].ys, hidY = spaced(a1.length, 80, 1), outY = cols[2].ys;

  // edges input->hidden
  for (let i=0;i<HSIZE;i++){
    for (let j=0;j<2;j++){
      const val = Math.abs(W1[i][j]*x[j]); // contribution magnitude
      const a = constrain(val, 0, 1);
      stroke(30, 120, 255, map(a,0,1,30,220));
      strokeWeight(map(a,0,1,1,6));
      line(inX, inY[j], hidX, hidY[i]);
    }
  }
  // edges hidden->output
  for (let i=0;i<2;i++){
    for (let j=0;j<HSIZE;j++){
      const val = Math.abs(W2[i][j]*a1[j]);
      const a = constrain(val, 0, 1);
      stroke(0, 180, 90, map(a,0,1,30,220));
      strokeWeight(map(a,0,1,1,6));
      line(hidX, hidY[j], outX, outY[i]);
    }
  }

  // input nodes (encode x as fill)
  for (let j=0;j<2;j++){
    const v = x[j];
    drawNode(inX, inY[j], map(v,0,1,40,220), color(30,120,255));
    drawNodeLabel(inX, inY[j], j===0?'x₁ (mouse x)':'x₂ (mouse y)');
  }

  // hidden nodes (pre-activation color, activation size)
  for (let i=0;i<HSIZE;i++){
    const z = z1[i], a = a1[i];
    const sz = map(Math.abs(a), 0, 1.5, 18, 34);
    const col = z>=0 ? color(0,150,60) : color(220,60,60);
    drawNode(hidX, hidY[i], 180, col, sz);
  }

  // output nodes (sigmoid 0..1)
  for (let i=0;i<2;i++){
    const v = y[i];
    drawNode(outX, outY[i], map(v,0,1,40,220), color(0,0,0));
    drawNodeLabel(outX, outY[i], i===0?'y₁':'y₂');
  }
}

function drawNode(x,y,alpha,col, size=26){
  noStroke();
  fill(red(col), green(col), blue(col), alpha);
  ellipse(x, y, size, size);
  stroke(0,30); noFill(); ellipse(x, y, size+8, size+8);
}
function drawNodeLabel(x,y,textStr){
  noStroke(); fill(50); textAlign(CENTER, TOP); textSize(12);
  text(textStr, x, y+18);
}

// UI + grid
function sel(id){ return document.getElementById(id); }
function drawGrid(step=40){
  stroke(235); strokeWeight(1);
  for (let x=0;x<=width;x+=step) line(x,0,x,height);
  for (let y=0;y<=height;y+=step) line(0,y,width,y);
}
