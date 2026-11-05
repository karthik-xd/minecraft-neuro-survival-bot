'use strict';
// brain.js - a growing, self-evolving neural network for Minecraft.
// Pure JavaScript, zero external ML dependencies.
//
// Capabilities:
// 1. Behavioral Cloning with Neurogenesis: When observing the human player,
//    evaluates every hidden neuron against human decisions -> keeps effective neurons,
//    prunes detrimental ones, and sprouts fresh plastic neurons ("neuron forming").
// 2. Reinforcement Learning (REINFORCE): When surviving/dying in-game,
//    scores neurons on survival rewards, transfers successful tactics, and evolves.
// 3. Explainable AI: Explains *why* the bot chose an action, highlighting active neurons.

const fs = require('fs');
const path = require('path');

// ---------------- What the brain sees and does ----------------
const TASKS = ['chase_player', 'kill_skeleton', 'kill_zombie', 'kill_mob', 'go_to', 'survive'];
const IN = {
  DIST: 0,
  SIN: 1,
  COS: 2,
  DY: 3,
  HP: 4,
  FOOD: 5,
  WALL: 6,
  WALL_HEAD: 7,
  DROP: 8,
  LIQUID: 9,
  GROUND: 10,
  IN_RANGE: 11,
  TASK0: 12, // 12..17 are TASKS (6 tasks)
  HELD_SWORD: 18,
  HELD_PICKAXE: 19,
  HELD_AXE: 20,
  HELD_SHOVEL: 21,
  IS_DAY: 22,
  INV_WOOD: 23,
  INV_PLANKS: 24,
  INV_STICKS: 25,
  INV_FOOD: 26,
  INV_SWORD: 27,
  INV_PICKAXE: 28,
  TARGET_IS_TREE: 29,
  TARGET_IS_ORE: 30,
  MOB_NEARBY: 31,
  CROUCHING: 32
};
const N_IN = 33; // 18 base/task + 15 survival/inventory features
const ACTIONS = ['idle', 'forward', 'left', 'right', 'jump', 'attack', 'mine', 'use_item', 'sneak', 'sprint'];
const ACT = {
  IDLE: 0,
  FWD: 1,
  LEFT: 2,
  RIGHT: 3,
  JUMP: 4,
  ATTACK: 5,
  MINE: 6,
  USE_ITEM: 7,
  SNEAK: 8,
  SPRINT: 9
};
const N_OUT = ACTIONS.length;

// ---------------- Growth settings ----------------
const MAX_HIDDEN = 128;        // Expanded memory cap for complex survival skills
const GROW = 4;                // New plastic neurons added each generation
const MIN_KEEP = 8;            // Never prune below foundational instincts
const MIN_STEPS_TO_JUDGE = 25; // Minimum sample size for ablation scoring

// ---------------- Math helpers ----------------
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function sigmoid(v) {
  if (v > 15) return 1;
  if (v < -15) return 0;
  return 1 / (1 + Math.exp(-v));
}

function softmax(lg) {
  const m = Math.max(...lg);
  const e = lg.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

function logSumExp(lg) {
  const m = Math.max(...lg);
  return m + Math.log(lg.reduce((a, v) => a + Math.exp(v - m), 0));
}

function sample(p) {
  let r = Math.random(), c = 0;
  for (let k = 0; k < p.length; k++) {
    c += p[k];
    if (r <= c) return k;
  }
  return p.length - 1;
}

// ---------------- The Gated Recurrent Unit (GRU) ----------------
// Solves vanishing gradients & catastrophic temporal forgetting via:
// - Update Gate (z_t): Controls how much previous memory h_{t-1} to preserve vs update
// - Reset Gate (r_t): Controls how past state enters the candidate memory
// - Candidate Memory (c_t): Nonlinear feature combination
// Memory equation: h_t = (1 - z_t) * h_{t-1} + z_t * c_t
class Net {
  constructor(H) {
    this.H = H;

    // 1. Update Gate weights & biases: z = sigmoid(Wz * x + Uz * h_{t-1} + bz)
    this.Wz = Array.from({ length: H }, () => new Array(N_IN).fill(0));
    this.Uz = Array.from({ length: H }, () => new Array(H).fill(0));
    this.bz = new Array(H).fill(-1.2); // Initialized negative so gate preserves past memory by default!

    // 2. Reset Gate weights & biases: r = sigmoid(Wr * x + Ur * h_{t-1} + br)
    this.Wr = Array.from({ length: H }, () => new Array(N_IN).fill(0));
    this.Ur = Array.from({ length: H }, () => new Array(H).fill(0));
    this.br = new Array(H).fill(0);

    // 3. Candidate Hidden weights & biases: c = tanh(Wh * x + Uh * (r * h_{t-1}) + bh)
    this.Wh = Array.from({ length: H }, () => new Array(N_IN).fill(0));
    this.Uh = Array.from({ length: H }, () => new Array(H).fill(0));
    this.bh = new Array(H).fill(0);

    // 4. Output classification policy: logits = W2 * h + b2
    this.W2 = Array.from({ length: N_OUT }, () => new Array(H).fill(0));
    this.b2 = new Array(N_OUT).fill(0);

    // Temporal hidden state (persistent memory vector)
    this.h_state = new Array(H).fill(0);
  }

  resetState() {
    this.h_state = new Array(this.H).fill(0);
  }

  // Forward step across one tick with gated memory retention
  step(x, h_prev = this.h_state) {
    const prev = h_prev || new Array(this.H).fill(0);
    const H = this.H;
    const z = new Array(H), r = new Array(H), c = new Array(H), h = new Array(H);

    // Compute Update and Reset gates
    for (let j = 0; j < H; j++) {
      let sz = this.bz[j];
      let sr = this.br[j];
      for (let i = 0; i < N_IN; i++) {
        const xi = x[i] || 0;
        sz += this.Wz[j][i] * xi;
        sr += this.Wr[j][i] * xi;
      }
      for (let k = 0; k < H; k++) {
        const pk = prev[k] || 0;
        sz += this.Uz[j][k] * pk;
        sr += this.Ur[j][k] * pk;
      }
      z[j] = sigmoid(sz);
      r[j] = sigmoid(sr);
    }

    // Compute Candidate state and final Gated Hidden state
    for (let j = 0; j < H; j++) {
      let sc = this.bh[j];
      for (let i = 0; i < N_IN; i++) {
        sc += this.Wh[j][i] * (x[i] || 0);
      }
      for (let k = 0; k < H; k++) {
        sc += this.Uh[j][k] * (r[k] * (prev[k] || 0));
      }
      c[j] = Math.tanh(sc);

      // Gated interpolation: linearly carries forward past memory without vanishing gradients!
      h[j] = (1 - z[j]) * (prev[j] || 0) + z[j] * c[j];
    }

    return { z, r, c, h };
  }

  hidden(x, h_prev = this.h_state) {
    return this.step(x, h_prev).h;
  }

  logits(h) {
    return this.W2.map((row, k) => {
      let s = this.b2[k];
      for (let j = 0; j < this.H; j++) s += row[j] * h[j];
      return s;
    });
  }

  probs(x, updateState = true) {
    const fwd = this.step(x, this.h_state);
    if (updateState) {
      this.h_state = fwd.h; // Advance internal temporal state
    }
    return { h: fwd.h, p: softmax(this.logits(fwd.h)), z: fwd.z, r: fwd.r, c: fwd.c };
  }

  clone() {
    return Net.fromJSON(JSON.parse(JSON.stringify(this)));
  }

  static fromJSON(d) {
    if (!d || !d.H) return null;
    const n = new Net(d.H);
    if (d.Wz && d.Uz) {
      n.Wz = d.Wz; n.Uz = d.Uz; n.bz = d.bz;
      n.Wr = d.Wr; n.Ur = d.Ur; n.br = d.br;
      n.Wh = d.Wh; n.Uh = d.Uh; n.bh = d.bh;
      n.W2 = d.W2; n.b2 = d.b2;
    } else {
      return null;
    }
    n.resetState();
    return n;
  }
}

// ---------------- Foundational Minecraft instincts (Newbie Core) ----------------
function newbieNet() {
  const H = 14;
  const n = new Net(H);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < N_IN; i++) {
      n.Wz[j][i] = randn() * 0.04;
      n.Wr[j][i] = randn() * 0.04;
      n.Wh[j][i] = randn() * 0.06;
    }
    for (let k = 0; k < H; k++) {
      n.Uz[j][k] = randn() * 0.04;
      n.Ur[j][k] = randn() * 0.04;
      n.Uh[j][k] = randn() * 0.04;
    }
    for (let k = 0; k < N_OUT; k++) {
      n.W2[k][j] = randn() * 0.06;
    }
    n.Uh[j][j] = 0.50; // Self-recurrent persistence
    n.bz[j] = -1.2;   // Preserves memory by default (anti-forgetting)
  }

  // Neuron 0: "Target ahead -> move forward"
  n.Wh[0][IN.COS] = 2.5;
  n.bh[0] = -0.3;
  n.W2[ACT.FWD][0] = 2.0;

  // Neuron 1: "Cliff or water/lava hazard -> do not advance, veer away"
  n.Wh[1][IN.DROP] = 3.0;
  n.Wh[1][IN.LIQUID] = 3.0;
  n.W2[ACT.FWD][1] = -4.0;
  n.W2[ACT.JUMP][1] = -3.0;
  n.W2[ACT.LEFT][1] = 1.5;

  // Neuron 2: "Target on left/right -> steer toward target"
  n.Wh[2][IN.SIN] = 4.0;
  n.W2[ACT.LEFT][2] = 2.5;
  n.W2[ACT.RIGHT][2] = -2.5;

  // Neuron 3: "Target behind -> turn around"
  n.Wh[3][IN.COS] = -2.0;
  n.W2[ACT.LEFT][3] = 1.5;

  // Neuron 4: "Block ahead with clearance above -> jump hurdle"
  n.Wh[4][IN.WALL] = 3.0;
  n.Wh[4][IN.WALL_HEAD] = -3.0;
  n.bh[4] = -1.5;
  n.W2[ACT.JUMP][4] = 3.5;

  // Neuron 5: "Target in melee reach and facing -> attack with weapon"
  n.Wh[5][IN.IN_RANGE] = 2.5;
  n.Wh[5][IN.COS] = 1.5;
  n.Wh[5][IN.HELD_SWORD] = 1.5;
  n.bh[5] = -2.0;
  n.W2[ACT.ATTACK][5] = 4.0;
  n.W2[ACT.FWD][5] = -1.0;

  // Neuron 6: "Target is Tree/Wood ahead -> mine wood (especially with axe)"
  n.Wh[6][IN.TARGET_IS_TREE] = 3.5;
  n.Wh[6][IN.HELD_AXE] = 1.5;
  n.bh[6] = -1.5;
  n.W2[ACT.MINE][6] = 4.0;
  n.W2[ACT.FWD][6] = -1.0;

  // Neuron 7: "Target is Ore/Stone ahead -> mine with pickaxe"
  n.Wh[7][IN.TARGET_IS_ORE] = 3.5;
  n.Wh[7][IN.HELD_PICKAXE] = 1.5;
  n.bh[7] = -1.5;
  n.W2[ACT.MINE][7] = 4.0;

  // Neuron 8: "Food is low (<12) & has food -> eat / use_item"
  n.Wh[8][IN.FOOD] = -3.0;
  n.Wh[8][IN.INV_FOOD] = 2.0;
  n.bh[8] = 0.5;
  n.W2[ACT.USE_ITEM][8] = 3.5;

  // Neuron 9: "Target far away and safe terrain ahead -> sprint"
  n.Wh[9][IN.DIST] = 2.0;
  n.Wh[9][IN.COS] = 2.0;
  n.Wh[9][IN.DROP] = -3.0;
  n.Wh[9][IN.LIQUID] = -3.0;
  n.bh[9] = -1.0;
  n.W2[ACT.SPRINT][9] = 3.0;

  // Default biases
  n.b2[ACT.FWD] = 0.2;
  n.b2[ACT.ATTACK] = -0.8;
  n.b2[ACT.MINE] = -1.0;
  n.b2[ACT.USE_ITEM] = -1.5;
  return n;
}

// ---------------- Turning facts into input vector ----------------
function makeObs(f) {
  const x = new Array(N_IN).fill(0);
  x[IN.DIST] = Math.min((f.dist || 0) / 20, 1);
  x[IN.SIN] = Math.sin(f.rel || 0);
  x[IN.COS] = Math.cos(f.rel || 0);
  x[IN.DY] = Math.max(-1, Math.min(1, (f.dy || 0) / 5));
  x[IN.HP] = (f.health ?? 20) / 20;
  x[IN.FOOD] = (f.food ?? 20) / 20;
  x[IN.WALL] = f.wall ? 1 : 0;
  x[IN.WALL_HEAD] = f.wallHead ? 1 : 0;
  x[IN.DROP] = f.drop ? 1 : 0;
  x[IN.LIQUID] = f.liquid ? 1 : 0;
  x[IN.GROUND] = f.onGround ? 1 : 0;
  x[IN.IN_RANGE] = f.inRange ? 1 : 0;

  const ti = TASKS.indexOf(f.task);
  if (ti >= 0) x[IN.TASK0 + ti] = 1;

  // Level 1 Survival, Tool & Inventory perception
  x[IN.HELD_SWORD] = f.heldSword ? 1 : 0;
  x[IN.HELD_PICKAXE] = f.heldPickaxe ? 1 : 0;
  x[IN.HELD_AXE] = f.heldAxe ? 1 : 0;
  x[IN.HELD_SHOVEL] = f.heldShovel ? 1 : 0;
  x[IN.IS_DAY] = f.isDay ? 1 : 0;
  x[IN.INV_WOOD] = Math.min(1, (f.invWood || 0) / 64);
  x[IN.INV_PLANKS] = Math.min(1, (f.invPlanks || 0) / 64);
  x[IN.INV_STICKS] = Math.min(1, (f.invSticks || 0) / 16);
  x[IN.INV_FOOD] = Math.min(1, (f.invFood || 0) / 32);
  x[IN.INV_SWORD] = f.invSword ? 1 : 0;
  x[IN.INV_PICKAXE] = f.invPickaxe ? 1 : 0;
  x[IN.TARGET_IS_TREE] = f.targetIsTree ? 1 : 0;
  x[IN.TARGET_IS_ORE] = f.targetIsOre ? 1 : 0;
  x[IN.MOB_NEARBY] = Math.min(1, (f.mobNearby || 0) / 5);
  x[IN.CROUCHING] = f.crouching ? 1 : 0;

  return x;
}

// ---------------- Reward function ----------------
function stepReward(pf, f, a, opts = {}) {
  let r = -0.01; // Small time penalty
  if (pf && pf.hasTarget && f.hasTarget && !opts.targetChanged) {
    r += (pf.dist - f.dist); // Reward approaching target
  }
  if (a === ACT.ATTACK && f.inRange && Math.abs(f.rel) < 0.6) {
    r += 0.4; // Landed melee hit
    if (f.heldSword) r += 0.2; // Extra reward for proper weapon usage!
  }
  if (a === ACT.MINE && (f.targetIsTree || f.targetIsOre)) {
    r += 0.4; // Rewarding mining trees or ores
    if (f.targetIsTree && f.heldAxe) r += 0.2;
    if (f.targetIsOre && f.heldPickaxe) r += 0.2;
  }
  if (a === ACT.USE_ITEM && f.food < 14 && f.invFood > 0) {
    r += 0.5; // Rewarding eating when hungry
  }
  if (pf && f.health < pf.health) {
    r -= 0.5 * (pf.health - f.health); // Penalty for taking damage
  }
  if (opts.killed) {
    r += 10.0; // Elimination bonus
  }
  if (opts.mined) {
    r += 3.0; // Mining block completion bonus
  }
  if (f.task === 'go_to' && f.hasTarget && f.dist < 1.5) {
    r += 0.5; // Reached goal waypoint
  }
  return r;
}

// ---------------- Policy gradient training (REINFORCE & Behavioral Cloning) ----------------
function advantages(rewards, gamma = 0.97) {
  const n = rewards.length, G = new Array(n);
  let run = 0;
  for (let t = n - 1; t >= 0; t--) {
    run = rewards[t] + gamma * run;
    G[t] = run;
  }
  const mean = G.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(G.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1;
  return G.map(g => (g - mean) / std);
}

function train(net, steps, adv, lr = 0.12, beta = 0.005) {
  const H = net.H, n = steps.length;
  if (!n) return;

  const gWz = net.Wz.map(r => r.map(() => 0)), gUz = net.Uz.map(r => r.map(() => 0)), gbz = new Array(H).fill(0);
  const gWr = net.Wr.map(r => r.map(() => 0)), gUr = net.Ur.map(r => r.map(() => 0)), gbr = new Array(H).fill(0);
  const gWh = net.Wh.map(r => r.map(() => 0)), gUh = net.Uh.map(r => r.map(() => 0)), gbh = new Array(H).fill(0);
  const gW2 = net.W2.map(r => r.map(() => 0)), gb2 = new Array(N_OUT).fill(0);

  net.resetState();
  const hiddenHistory = [new Array(H).fill(0)];
  const forwardHistory = [];
  const probsHistory = [];

  // 1. Forward Pass Through Time (GRU Gated Unrolling)
  for (let t = 0; t < n; t++) {
    const { x } = steps[t];
    const prev = hiddenHistory[t];
    const fwd = net.step(x, prev);
    hiddenHistory.push(fwd.h);
    forwardHistory.push(fwd);
    probsHistory.push(softmax(net.logits(fwd.h)));
  }

  let dh_next = new Array(H).fill(0);

  // 2. Backward Pass Through Time (BPTT with GRU Gradient Highways)
  for (let t = n - 1; t >= 0; t--) {
    const { x, a } = steps[t];
    const { z, r, c, h } = forwardHistory[t];
    const h_prev = hiddenHistory[t];
    const p = probsHistory[t];
    const ent = -p.reduce((s, v) => s + v * Math.log(v + 1e-12), 0);
    const dl = p.map((pk, k) => adv[t] * ((k === a ? 1 : 0) - pk) + beta * (-pk * (Math.log(pk + 1e-12) + ent)));

    for (let k = 0; k < N_OUT; k++) {
      gb2[k] += dl[k];
      for (let j = 0; j < H; j++) gW2[k][j] += dl[k] * h[j];
    }

    // Gradient arriving at h_t from output policy logits + step t+1 recurrence
    const dh = new Array(H);
    for (let j = 0; j < H; j++) {
      let d = dh_next[j];
      for (let k = 0; k < N_OUT; k++) d += dl[k] * net.W2[k][j];
      dh[j] = d;
    }

    // Gate derivatives:
    // h = (1 - z) * h_prev + z * c
    const dz_raw = new Array(H), dc_raw = new Array(H);
    for (let j = 0; j < H; j++) {
      const dc = dh[j] * z[j];
      dc_raw[j] = dc * (1 - c[j] * c[j]); // tanh derivative

      const dz = dh[j] * (c[j] - (h_prev[j] || 0));
      dz_raw[j] = dz * z[j] * (1 - z[j]); // sigmoid derivative

      gbh[j] += dc_raw[j];
      gbz[j] += dz_raw[j];

      for (let i = 0; i < N_IN; i++) {
        const xi = x[i] || 0;
        gWh[j][i] += dc_raw[j] * xi;
        gWz[j][i] += dz_raw[j] * xi;
      }
      for (let k = 0; k < H; k++) {
        gUz[j][k] += dz_raw[j] * (h_prev[k] || 0);
      }
    }

    // Reset gate derivatives:
    // candidate sc_j = ... + sum_k Uh[j][k] * (r_k * h_prev_k)
    const d_rh = new Array(H).fill(0);
    for (let k = 0; k < H; k++) {
      for (let j = 0; j < H; j++) {
        d_rh[k] += dc_raw[j] * net.Uh[j][k];
        gUh[j][k] += dc_raw[j] * (r[k] * (h_prev[k] || 0));
      }
    }

    const dr_raw = new Array(H);
    for (let k = 0; k < H; k++) {
      const dr = d_rh[k] * (h_prev[k] || 0);
      dr_raw[k] = dr * r[k] * (1 - r[k]); // sigmoid derivative
      gbr[k] += dr_raw[k];

      for (let i = 0; i < N_IN; i++) {
        gWr[k][i] += dr_raw[k] * (x[i] || 0);
      }
      for (let m = 0; m < H; m++) {
        gUr[k][m] += dr_raw[k] * (h_prev[m] || 0);
      }
    }

    // Gradient flowing to h_prev for step t-1:
    // (1 - z_k) creates an additive linear shortcut that ELIMINATES vanishing gradients!
    dh_next = new Array(H);
    for (let k = 0; k < H; k++) {
      let dprev = dh[k] * (1 - z[k]) + d_rh[k] * r[k];
      for (let j = 0; j < H; j++) {
        dprev += dz_raw[j] * net.Uz[j][k];
        dprev += dr_raw[j] * net.Ur[j][k];
      }
      dh_next[k] = Math.max(-4, Math.min(4, dprev)); // Gradient clipping
    }
  }

  // 3. Clipped parameter updates
  const up = g => lr * Math.max(-1, Math.min(1, g / n));
  for (let k = 0; k < N_OUT; k++) {
    net.b2[k] += up(gb2[k]);
    for (let j = 0; j < H; j++) net.W2[k][j] += up(gW2[k][j]);
  }
  for (let j = 0; j < H; j++) {
    net.bz[j] += up(gbz[j]);
    net.br[j] += up(gbr[j]);
    net.bh[j] += up(gbh[j]);
    for (let i = 0; i < N_IN; i++) {
      net.Wz[j][i] += up(gWz[j][i]);
      net.Wr[j][i] += up(gWr[j][i]);
      net.Wh[j][i] += up(gWh[j][i]);
    }
    for (let k = 0; k < H; k++) {
      net.Uz[j][k] += up(gUz[j][k]);
      net.Ur[j][k] += up(gUr[j][k]);
      net.Uh[j][k] += up(gUh[j][k]);
    }
  }
}

// ---------------- Neuron ablation scoring ----------------
function scoreNeurons(net, steps, adv) {
  const score = new Array(net.H).fill(0);
  net.resetState();
  const hiddenHistory = [new Array(net.H).fill(0)];

  for (let t = 0; t < steps.length; t++) {
    const { x, a } = steps[t];
    const h = net.hidden(x, hiddenHistory[t]);
    hiddenHistory.push(h);
    const lg = net.logits(h);
    const lpFull = lg[a] - logSumExp(lg);

    for (let j = 0; j < net.H; j++) {
      const lg2 = lg.map((v, k) => v - net.W2[k][j] * h[j]); // Neuron j silenced
      score[j] += adv[t] * (lpFull - (lg2[a] - logSumExp(lg2)));
    }
  }
  return score.map(s => s / (steps.length || 1));
}

// ---------------- Neurogenesis & Evolution: Generation A -> Generation B ----------------
function nextGeneration(Anet, steps, adv, lr = 0.12) {
  const scores = scoreNeurons(Anet, steps, adv);
  const judged = steps.length >= MIN_STEPS_TO_JUDGE;

  // Identify neurons to prune:
  // - Never prune the foundational newbie core (first MIN_KEEP = 8 neurons)
  // - Only prune non-core neurons that actively contradict demonstrations (score < -0.015)
  let candidateCuts = [];
  if (judged) {
    for (let j = MIN_KEEP; j < Anet.H; j++) {
      if (scores[j] < -0.015) {
        candidateCuts.push({ index: j, score: scores[j] });
      }
    }
    candidateCuts.sort((a, b) => a.score - b.score); // Worst first
  }

  // Pruning policy:
  // If below MAX_HIDDEN (128), prune AT MOST 1 neuron so net growth is always >= +3!
  const room = MAX_HIDDEN - Anet.H;
  let cutCount = 0;
  if (room >= GROW) {
    cutCount = candidateCuts.length > 0 ? 1 : 0;
  } else {
    const needed = GROW - room;
    cutCount = Math.min(candidateCuts.length, Math.min(needed, Anet.H - MIN_KEEP));
  }

  const cutIndices = new Set(candidateCuts.slice(0, cutCount).map(c => c.index));
  const kept = [];
  for (let j = 0; j < Anet.H; j++) {
    if (!cutIndices.has(j)) kept.push(j);
  }

  // Always add GROW (4) fresh neurons, capped at MAX_HIDDEN
  const newN = Math.min(GROW, MAX_HIDDEN - kept.length);
  const B = new Net(kept.length + newN);

  // Preserve learned GRU weights of kept neurons
  kept.forEach((j, idx) => {
    B.Wz[idx] = Anet.Wz[j].slice();
    B.Wr[idx] = Anet.Wr[j].slice();
    B.Wh[idx] = Anet.Wh[j].slice();
    B.bz[idx] = Anet.bz[j];
    B.br[idx] = Anet.br[j];
    B.bh[idx] = Anet.bh[j];
    for (let k = 0; k < N_OUT; k++) B.W2[k][idx] = Anet.W2[k][j];
  });

  // Preserve learned recurrent memory connections between kept neurons
  kept.forEach((jSrc, idxSrc) => {
    kept.forEach((jDst, idxDst) => {
      B.Uz[idxSrc][idxDst] = Anet.Uz[jSrc][jDst];
      B.Ur[idxSrc][idxDst] = Anet.Ur[jSrc][jDst];
      B.Uh[idxSrc][idxDst] = Anet.Uh[jSrc][jDst];
    });
  });

  // Initialize fresh plastic GRU neurons with memory persistence
  for (let idx = kept.length; idx < kept.length + newN; idx++) {
    for (let i = 0; i < N_IN; i++) {
      B.Wz[idx][i] = randn() * 0.04;
      B.Wr[idx][i] = randn() * 0.04;
      B.Wh[idx][i] = randn() * 0.18;
    }
    for (let k = 0; k < N_OUT; k++) B.W2[k][idx] = randn() * 0.05;
    for (let k = 0; k < B.H; k++) {
      B.Uz[idx][k] = randn() * 0.04;
      B.Ur[idx][k] = randn() * 0.04;
      B.Uh[idx][k] = randn() * 0.04;
      B.Uz[k][idx] = randn() * 0.04;
      B.Ur[k][idx] = randn() * 0.04;
      B.Uh[k][idx] = randn() * 0.04;
    }
    B.Uh[idx][idx] = 0.50; // Self-memory persistence diagonal
    B.bz[idx] = -1.2;     // Anti-forgetting update gate bias
  }
  B.b2 = Anet.b2.slice();

  // Train new network on demonstrations
  for (let pass = 0; pass < 3; pass++) train(B, steps, adv, lr);

  return {
    B,
    scores,
    report: {
      hiddenBefore: Anet.H,
      kept: kept.length,
      cut: cutCount,
      added: newN,
      hiddenAfter: B.H,
      judged
    }
  };
}

// ---------------- Brain Controller ----------------
class Brain {
  constructor(dir) {
    this.dir = dir || path.join(__dirname, 'data_bot');
    fs.mkdirSync(this.dir, { recursive: true });
    this.file = path.join(this.dir, 'brain.json');
    this.historyFile = path.join(this.dir, 'history.json');

    if (fs.existsSync(this.file)) {
      try {
        const d = JSON.parse(fs.readFileSync(this.file, 'utf8'));
        // Verify architecture dimensions and GRU gate presence
        const isDimMatch = d.net &&
          Array.isArray(d.net.Wh) && d.net.Wh[0] && d.net.Wh[0].length === N_IN &&
          Array.isArray(d.net.W2) && d.net.W2.length === N_OUT &&
          Array.isArray(d.net.Wz) && Array.isArray(d.net.Uz);

        if (isDimMatch) {
          this.net = Net.fromJSON(d.net);
          this.generation = d.generation || 0;
        } else {
          console.log(`[BRAIN NOTICE] Upgrading architecture from Vanilla RNN to GRU (Gated Recurrent Unit with Memory Retention Gates).`);
          const backupName = path.join(this.dir, `brain_legacy_gen${d.generation || 0}.json`);
          fs.writeFileSync(backupName, JSON.stringify(d, null, 2));
          this.net = newbieNet();
          this.generation = 0;
          this.save();
        }
      } catch (e) {
        console.error('Error loading brain.json, starting newbie GRU:', e.message);
        this.net = newbieNet();
        this.generation = 0;
        this.save();
      }
    } else {
      this.net = newbieNet();
      this.generation = 0;
      this.save();
    }
  }

  act(x) {
    const { p } = this.net.probs(x);
    return { a: sample(p), p };
  }

  // Explainable AI: returns top action and which neurons drove it
  explain(x) {
    const { h, p } = this.net.probs(x);
    let best = 0;
    for (let k = 1; k < N_OUT; k++) {
      if (p[k] > p[best]) best = k;
    }

    // Rank hidden neurons by their positive contribution to the chosen action
    const neuronContributions = h.map((actVal, j) => ({
      index: j,
      activation: +actVal.toFixed(2),
      contribution: +(this.net.W2[best][j] * actVal).toFixed(2)
    })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    const activeNeurons = h.filter(v => Math.abs(v) > 0.25).length;

    return {
      action: ACTIONS[best],
      actionIndex: best,
      confidence: +(p[best] * 100).toFixed(1),
      probabilities: ACTIONS.map((name, i) => ({ name, prob: +(p[i] * 100).toFixed(1) })),
      topNeurons: neuronContributions.slice(0, 4),
      activeNeurons,
      totalNeurons: this.net.H
    };
  }

  save() {
    fs.writeFileSync(this.file, JSON.stringify({ generation: this.generation, net: this.net }, null, 2));
  }

  // Measure cloning accuracy on a sample set
  testAccuracy(samples) {
    if (!samples || !samples.length) return 0;
    this.net.resetState();
    let ok = 0;
    for (const d of samples) {
      const { p } = this.net.probs(d.x, true);
      let best = 0;
      for (let k = 1; k < N_OUT; k++) {
        if (p[k] > p[best]) best = k;
      }
      if (best === d.a) ok++;
    }
    return ok / samples.length;
  }

  // ---------------- Continual Learning from Demonstrations ----------------
  // Called whenever the user stops playing / records:
  // Evaluates current neurons, prunes bad ones, forms GROW new neurons,
  // trains on demonstrations, increments generation, and saves seamlessly.
  evolveFromDemos(samples, opts = {}) {
    if (!samples || samples.length < 15) {
      return { error: `Need at least 15 demonstration samples (got ${samples ? samples.length : 0})` };
    }

    const epochs = opts.epochs || 30;
    const lr = opts.lr || 0.12;

    // Filter excessive idle steps (keep only 25% of idle) so bot stays agile
    let balanced = samples.filter(d => d.a !== ACT.IDLE || Math.random() < 0.25);
    if (balanced.length < 10) balanced = samples;

    const accBefore = this.testAccuracy(balanced);

    // Archive Model A
    const archivePath = path.join(this.dir, `gen_${this.generation}_A.json`);
    fs.writeFileSync(archivePath, JSON.stringify(this.net));

    // Unit advantages = imitation objective (copy human demonstration)
    const adv = balanced.map(() => 1.0);

    // Neurogenesis: score A's neurons against human actions -> prune & sprout new neurons into B
    const { B, scores, report } = nextGeneration(this.net, balanced, adv, lr);

    // Behavioral cloning training loop on Model B
    for (let e = 1; e <= epochs; e++) {
      // Shuffle samples
      for (let i = balanced.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [balanced[i], balanced[j]] = [balanced[j], balanced[i]];
      }
      for (let i = 0; i < balanced.length; i += 64) {
        const batch = balanced.slice(i, i + 64);
        train(B, batch, batch.map(() => 1.0), lr, 0.005);
      }
    }

    this.net = B;
    const accAfter = this.testAccuracy(balanced);
    this.generation++;
    this.save();

    const record = {
      type: 'demo_evolution',
      generation: this.generation,
      timestamp: new Date().toISOString(),
      samples: balanced.length,
      accuracyBefore: +(accBefore * 100).toFixed(1),
      accuracyAfter: +(accAfter * 100).toFixed(1),
      ...report
    };

    this.appendHistory(record);
    return record;
  }

  // ---------------- Reinforcement Learning Life Cycle ----------------
  // Called once per survival life when the bot dies or completes a session
  endLife(steps, reason = 'died') {
    if (steps.length < 2) return null;
    const total = steps.reduce((s, st) => s + (st.r || 0), 0);
    const adv = advantages(steps.map(s => s.r || 0));

    // Archive Model A
    fs.writeFileSync(path.join(this.dir, `gen_${this.generation}_A.json`), JSON.stringify(this.net));

    // Evolve Model B from life experience
    const { B, report } = nextGeneration(this.net, steps, adv);
    this.net = B;
    this.generation++;
    this.save();

    const record = {
      type: 'survival_life',
      generation: this.generation,
      reason,
      steps: steps.length,
      totalReward: +total.toFixed(2),
      timestamp: new Date().toISOString(),
      ...report
    };

    this.appendHistory(record);
    return record;
  }

  appendHistory(record) {
    let history = [];
    if (fs.existsSync(this.historyFile)) {
      try {
        history = JSON.parse(fs.readFileSync(this.historyFile, 'utf8'));
      } catch {
        history = [];
      }
    }
    history.push(record);
    fs.writeFileSync(this.historyFile, JSON.stringify(history, null, 2));
  }

  getArchitecture() {
    return {
      type: 'GRU (Gated Recurrent Unit)',
      hasGatedMemory: true,
      hasUpdateGate: true,
      hasResetGate: true,
      preventsForgetting: true,
      generation: this.generation,
      hiddenCount: this.net.H,
      maxHidden: MAX_HIDDEN,
      growRate: GROW,
      inputFeatures: N_IN,
      outputActions: N_OUT,
      actions: ACTIONS,
      tasks: TASKS
    };
  }
}

module.exports = {
  Brain,
  Net,
  newbieNet,
  makeObs,
  stepReward,
  IN,
  ACT,
  ACTIONS,
  TASKS,
  N_IN,
  MAX_HIDDEN,
  GROW
};
