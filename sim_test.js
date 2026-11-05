'use strict';
// sim_test.js - train the brain in a fast fake arena (no Minecraft needed).
// Run:  node sim_test.js 300      (300 lives)
// It uses the same Brain, same inputs, same rewards as bot.js, so what it learns carries over.

const path = require('path');
const { Brain, makeObs, stepReward, ACT } = require('./brain');

const LIVES = +(process.argv[2] || 200);
const TURN = 0.3, SPEED = 0.25, MAX_STEPS = 400;
const brain = new Brain(path.join(__dirname, 'data_sim'));

function facts(me, tgt) {
  const dx = tgt.x - me.x, dz = tgt.z - me.z;
  const dist = Math.hypot(dx, dz);
  let rel = Math.atan2(-dx, -dz) - me.yaw;
  rel = Math.atan2(Math.sin(rel), Math.cos(rel));
  return { hasTarget: true, dist, rel, dy: 0, health: me.hp, food: 20, wall: false, wallHead: false,
           drop: false, liquid: false, onGround: true, inRange: dist < 3, task: 'kill_skeleton' };
}

function oneLife() {
  const me = { x: 0, z: 0, yaw: Math.random() * 6.28, hp: 20 };
  const ang = Math.random() * 6.28, d = 6 + Math.random() * 10;
  const tgt = { x: Math.cos(ang) * d, z: Math.sin(ang) * d, hp: 3 };
  const steps = []; let prev = null, killed = false;
  for (let t = 0; t < MAX_STEPS && !killed; t++) {
    const f = facts(me, tgt);
    if (prev) prev.step.r = stepReward(prev.f, f, prev.step.a, {});
    const x = makeObs(f);
    const { a } = brain.act(x);
    const step = { x, a, r: 0 };
    steps.push(step); prev = { f, step };
    if (a === ACT.LEFT) me.yaw += TURN;
    if (a === ACT.RIGHT) me.yaw -= TURN;
    if (a === ACT.FWD || a === ACT.JUMP) { me.x += -Math.sin(me.yaw) * SPEED; me.z += -Math.cos(me.yaw) * SPEED; }
    if (a === ACT.ATTACK && f.inRange && Math.abs(f.rel) < 0.6 && --tgt.hp <= 0) {
      killed = true; step.r += stepReward(f, f, a, { killed: true });
    }
  }
  return { steps, killed, total: steps.reduce((s, st) => s + st.r, 0) };
}

let window = [];
for (let life = 1; life <= LIVES; life++) {
  const r = oneLife();
  const rep = brain.endLife(r.steps);
  window.push({ killed: r.killed, total: r.total, steps: r.steps.length });
  if (life % 20 === 0) {
    const k = window.filter(w => w.killed).length;
    const avg = window.reduce((s, w) => s + w.total, 0) / window.length;
    const len = window.reduce((s, w) => s + w.steps, 0) / window.length;
    console.log(`lives ${life - 19}-${life}: kills ${k}/20  avg reward ${avg.toFixed(1)}  avg steps ${len.toFixed(0)}  hidden ${rep.hiddenAfter} (kept ${rep.kept}, cut ${rep.cut}, added ${rep.added})`);
    window = [];
  }
}
