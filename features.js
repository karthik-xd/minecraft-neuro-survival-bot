'use strict';
// features.js - turns the Minecraft world into structured facts and diagnostics.
// Used by recorder.js (watching the player), bot.js (survival bot), and the web terminal.

const HOSTILE_MOBS = new Set([
  'skeleton', 'stray', 'wither_skeleton',
  'zombie', 'husk', 'drowned', 'zombie_villager',
  'spider', 'cave_spider', 'creeper', 'witch', 'enderman',
  'slime', 'magma_cube', 'phantom', 'pillager', 'ravager'
]);

// Find the target for a given task, relative to `me` (entity)
function findTarget(bot, task, goal, me) {
  if (task === 'go_to') {
    return goal ? { position: goal, isGoal: true, name: 'destination' } : null;
  }

  const self = me || bot.entity;
  if (!self || !self.position) return null;

  let wanted;
  if (task === 'kill_skeleton') {
    wanted = e => e.name === 'skeleton' || e.name === 'stray' || e.name === 'wither_skeleton' || e.mobType === 'Skeleton';
  } else if (task === 'kill_zombie') {
    wanted = e => e.name === 'zombie' || e.name === 'husk' || e.name === 'drowned' || e.name === 'zombie_villager';
  } else if (task === 'kill_mob') {
    wanted = e => HOSTILE_MOBS.has(e.name?.toLowerCase());
  } else if (task === 'chase_player') {
    wanted = e => e.type === 'player' && e !== self && e !== bot.entity && bot.players[e.username]?.gamemode !== 3;
  } else if (task === 'survive') {
    // In survival mode, target nearest hostile mob within 16 blocks
    wanted = e => HOSTILE_MOBS.has(e.name?.toLowerCase()) || (e.type === 'mob' && e.kind === 'Hostile');
  } else {
    // Default: hostile mobs or players
    wanted = e => HOSTILE_MOBS.has(e.name?.toLowerCase());
  }

  let best = null, bestD = Infinity;
  for (const id in bot.entities) {
    const e = bot.entities[id];
    if (!e || !e.position || e === self || e === bot.entity) continue;
    if (!wanted(e)) continue;

    const d = e.position.distanceTo(self.position);
    if (d < bestD) {
      bestD = d;
      best = e;
    }
  }

  return best;
}

// Extract environmental facts from Minecraft blocks, spatial vectors, inventory and tools
function facts(bot, state, target, task, extra = {}) {
  const pos = state.pos || state.position;
  const yaw = state.yaw || 0;

  const solid = b => !!b && b.boundingBox === 'block';
  const liquid = b => !!b && (b.name === 'lava' || b.name === 'water' || b.name === 'flowing_water' || b.name === 'flowing_lava');

  const fx = -Math.sin(yaw), fz = -Math.cos(yaw);
  const at = (dx, dy, dz) => {
    try {
      return bot.blockAt(pos.offset(dx, dy, dz));
    } catch {
      return null;
    }
  };

  const ahead = at(fx, 0, fz);
  const head = at(fx, 1, fz);
  const below = at(fx, -1, fz);
  const currentFeet = at(0, 0, 0);

  const aheadName = ahead ? ahead.name : 'air';
  const headName = head ? head.name : 'air';
  const belowName = below ? below.name : 'air';

  // Held item detection (from state, player entity, or extra)
  const held = extra.heldItem || state.heldItem || (state.equipment && state.equipment[0]) || (bot.heldItem || null);
  const heldName = (held && held.name ? held.name : '').toLowerCase();
  const heldSword = heldName.includes('sword');
  const heldPickaxe = heldName.includes('pickaxe');
  const heldAxe = heldName.includes('axe') && !heldName.includes('pickaxe');
  const heldShovel = heldName.includes('shovel');

  // Time of Day (ticks 0-24000; day is 0-12999)
  const timeOfDay = bot.time ? bot.time.timeOfDay : (extra.timeOfDay ?? 6000);
  const isDay = timeOfDay < 13000;

  // Inventory inspection (from bot or extra)
  let invWood = extra.invWood ?? 0;
  let invPlanks = extra.invPlanks ?? 0;
  let invSticks = extra.invSticks ?? 0;
  let invFood = extra.invFood ?? 0;
  let invSword = extra.invSword ?? heldSword;
  let invPickaxe = extra.invPickaxe ?? heldPickaxe;

  if (bot && bot.inventory && typeof bot.inventory.items === 'function') {
    const items = bot.inventory.items();
    for (const it of items) {
      const nm = (it.name || '').toLowerCase();
      if (/log|wood|stem/.test(nm)) invWood += it.count;
      else if (/planks?/.test(nm)) invPlanks += it.count;
      else if (/stick/.test(nm)) invSticks += it.count;
      else if (/bread|apple|beef|porkchop|chicken|mutton|cookie|carrot|potato|melon|stew/.test(nm)) invFood += it.count;
      if (nm.includes('sword')) invSword = true;
      if (nm.includes('pickaxe')) invPickaxe = true;
    }
  }

  // Block type inspection (Tree/Ore identification)
  const targetIsTree = /log|wood|stem/i.test(aheadName) || /log|wood|stem/i.test(headName);
  const targetIsOre = /ore|stone|cobble|granite|diorite|andesite|deepslate/i.test(aheadName) || /ore|stone/i.test(belowName);

  // Nearby hostile mobs scan within 12 blocks
  let mobNearby = 0;
  if (bot && bot.entities && pos) {
    for (const id in bot.entities) {
      const ent = bot.entities[id];
      if (!ent || !ent.position || ent === bot.entity || ent === state) continue;
      if (HOSTILE_MOBS.has(ent.name?.toLowerCase())) {
        if (ent.position.distanceTo(pos) < 12) {
          mobNearby++;
        }
      }
    }
  }

  const crouching = !!(state.crouching || extra.crouching || (state.metadata && (state.metadata[0] & 0x02)));

  const base = {
    health: state.health ?? 20,
    food: state.food ?? 20,
    wall: solid(ahead),
    wallHead: solid(head),
    drop: !solid(below) && !solid(at(fx, -2, fz)),
    liquid: liquid(ahead) || liquid(below) || liquid(currentFeet),
    onGround: state.onGround ?? true,
    task,
    aheadBlock: aheadName,
    headBlock: headName,
    belowBlock: belowName,
    heldSword,
    heldPickaxe,
    heldAxe,
    heldShovel,
    heldItemName: heldName || 'none',
    isDay,
    invWood,
    invPlanks,
    invSticks,
    invFood,
    invSword,
    invPickaxe,
    targetIsTree,
    targetIsOre,
    mobNearby,
    crouching
  };

  if (!target || !target.position) {
    return {
      ...base,
      hasTarget: false,
      targetName: 'none',
      dist: 20,
      rel: Math.PI / 2,
      dy: 0,
      inRange: false
    };
  }

  const tp = target.position;
  const dx = tp.x - pos.x, dz = tp.z - pos.z, dy = tp.y - pos.y;
  const dist = Math.hypot(dx, dy, dz);
  let rel = Math.atan2(-dx, -dz) - yaw;
  rel = Math.atan2(Math.sin(rel), Math.cos(rel));

  return {
    ...base,
    hasTarget: true,
    targetName: target.name || target.username || 'target',
    targetHp: target.metadata ? target.metadata[9] : (target.health || 20),
    dist,
    rel,
    dy,
    inRange: !target.isGoal && dist < 3.2
  };
}

// Diagnostic helper: explains why the bot stopped or cannot advance
function diagnoseObstacle(bot, state, f) {
  if (f.health <= 0) return '💀 Defeated in combat (Health reached 0)';
  if (f.liquid) return `🌊 Blocked by liquid hazard (${f.aheadBlock} ahead)`;
  if (f.drop) return '⚠️ Stopped before a hazardous cliff / drop';
  if (f.wall && f.wallHead) return `🧱 Path blocked by a 2-block high wall (${f.aheadBlock})`;
  if (f.wall && !f.wallHead) return `🚧 1-block hurdle ahead (${f.aheadBlock}) - Jump to clear`;
  if (!f.hasTarget) return '🔍 No target found within range (Searching...)';
  if (f.inRange) return `⚔️ Engaged in combat with ${f.targetName} (${f.dist.toFixed(1)}m away)`;
  return '⏸️ Standing still / Idle';
}

module.exports = {
  findTarget,
  facts,
  diagnoseObstacle,
  HOSTILE_MOBS
};
