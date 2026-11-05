'use strict';
// generate_demos.js - Synthesizes high-quality expert Minecraft demonstration data.
// Allows you to train the neural brain instantly without having to play manually!

const fs = require('fs');
const path = require('path');
const { makeObs, ACT, TASKS } = require('./brain');

function generateExpertDemos(count = 1000, demoDir = path.join(__dirname, 'demos')) {
  fs.mkdirSync(demoDir, { recursive: true });
  const fileName = path.join(demoDir, `synthetic_expert_${Date.now()}.jsonl`);
  const stream = fs.createWriteStream(fileName, { flags: 'a' });

  const samples = [];
  const actionCounts = {
    idle: 0, forward: 0, left: 0, right: 0, jump: 0,
    attack: 0, mine: 0, use_item: 0, sneak: 0, sprint: 0
  };

  for (let i = 0; i < count; i++) {
    // Generate realistic combat, mining, crafting, survival scenarios
    const scenarioType = Math.random();
    let f, a;

    if (scenarioType < 0.22) {
      // Scenario A: Melee Combat with Sword (Target close and facing)
      const dist = 1.0 + Math.random() * 2.2;
      const rel = (Math.random() - 0.5) * 0.4; // Facing target
      f = {
        dist, rel, dy: 0,
        health: 12 + Math.floor(Math.random() * 9),
        food: 16 + Math.floor(Math.random() * 5),
        wall: false, wallHead: false, drop: false, liquid: false,
        onGround: true, inRange: true, task: 'kill_skeleton',
        hasTarget: true,
        heldSword: true, heldPickaxe: false, heldAxe: false,
        invSword: true, mobNearby: 1, isDay: false
      };
      a = Math.random() < 0.2 ? ACT.JUMP : ACT.ATTACK;

    } else if (scenarioType < 0.40) {
      // Scenario B: Tree Mining / Wood Gathering
      f = {
        dist: 1.5, rel: 0.1, dy: 0,
        health: 20, food: 18,
        wall: true, wallHead: true, drop: false, liquid: false,
        onGround: true, inRange: false, task: 'survive',
        hasTarget: false, targetIsTree: true,
        aheadBlock: 'oak_log', headBlock: 'oak_log',
        heldAxe: true, invWood: Math.floor(Math.random() * 6),
        isDay: true
      };
      a = ACT.MINE;

    } else if (scenarioType < 0.52) {
      // Scenario C: Stone / Ore Mining with Pickaxe
      f = {
        dist: 1.8, rel: 0.05, dy: -1,
        health: 20, food: 17,
        wall: true, wallHead: false, drop: false, liquid: false,
        onGround: true, inRange: false, task: 'survive',
        hasTarget: false, targetIsOre: true,
        aheadBlock: 'iron_ore',
        heldPickaxe: true, invPickaxe: true,
        isDay: true
      };
      a = ACT.MINE;

    } else if (scenarioType < 0.62) {
      // Scenario D: Hunger Low -> Eat Food
      f = {
        dist: 10.0, rel: 0.0, dy: 0,
        health: 18, food: 9,
        wall: false, wallHead: false, drop: false, liquid: false,
        onGround: true, inRange: false, task: 'survive',
        hasTarget: false,
        invFood: 5, isDay: true
      };
      a = ACT.USE_ITEM;

    } else if (scenarioType < 0.75) {
      // Scenario E: Sprinting toward target / destination
      const dist = 6.0 + Math.random() * 12.0;
      const rel = (Math.random() - 0.5) * 0.2;
      f = {
        dist, rel, dy: 0,
        health: 20, food: 20,
        wall: false, wallHead: false, drop: false, liquid: false,
        onGround: true, inRange: false, task: 'chase_player',
        hasTarget: true, isDay: true
      };
      a = ACT.SPRINT;

    } else if (scenarioType < 0.85) {
      // Scenario F: Hurdle Jumping over 1-block wall
      f = {
        dist: 5.0, rel: 0.1, dy: 0,
        health: 20, food: 20,
        wall: true, wallHead: false, drop: false, liquid: false,
        onGround: true, inRange: false, task: 'survive',
        hasTarget: false, isDay: true
      };
      a = ACT.JUMP;

    } else if (scenarioType < 0.92) {
      // Scenario G: Hazard Avoidance (Cliff or lava ahead -> turn away)
      const isLava = Math.random() < 0.5;
      f = {
        dist: 8.0, rel: 0.0, dy: 0,
        health: 20, food: 20,
        wall: false, wallHead: false,
        drop: !isLava, liquid: isLava,
        onGround: true, inRange: false, task: 'survive',
        hasTarget: false
      };
      a = Math.random() < 0.5 ? ACT.LEFT : ACT.RIGHT;

    } else {
      // Scenario H: Stealth / Sneaking near ledge
      f = {
        dist: 3.0, rel: 0.0, dy: 0,
        health: 20, food: 20,
        wall: false, drop: true, liquid: false,
        onGround: true, inRange: false, task: 'survive',
        crouching: true, isDay: false
      };
      a = ACT.SNEAK;
    }

    const obs = makeObs(f);
    const sample = { x: obs, a, task: f.task };
    samples.push(sample);
    stream.write(JSON.stringify(sample) + '\n');

    const actNames = ['idle', 'forward', 'left', 'right', 'jump', 'attack', 'mine', 'use_item', 'sneak', 'sprint'];
    actionCounts[actNames[a]]++;
  }

  stream.end();
  return { fileName, count: samples.length, samples, actionCounts };
}

// Support CLI run
if (require.main === module) {
  const count = parseInt(process.argv[2] || '1000', 10);
  console.log(`Generating ${count} synthetic expert Minecraft demonstration samples...`);
  const result = generateExpertDemos(count);
  console.log(`Saved ${result.count} expert samples to: ${result.fileName}`);
  console.log('Action breakdown:', result.actionCounts);
}

module.exports = { generateExpertDemos };
