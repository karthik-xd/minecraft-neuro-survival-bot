'use strict';
// record.js / DemonstrationRecorder - watches YOU play, derives actions and environmental facts,
// and saves human demonstration trajectories for behavioral cloning and neurogenesis.

const mineflayer = require('mineflayer');
const fs = require('fs');
const path = require('path');
const { makeObs, ACT, ACTIONS, TASKS } = require('./brain');
const { findTarget, facts } = require('./features');

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

function deriveAction(prev, cur, swung, f) {
  const dyaw = wrap(cur.yaw - prev.yaw);
  const dx = cur.pos.x - prev.pos.x, dz = cur.pos.z - prev.pos.z, dy = cur.pos.y - prev.pos.y;
  const fwd = dx * -Math.sin(prev.yaw) + dz * -Math.cos(prev.yaw);

  // 1. Attack detection (swung at hostile/entity within range)
  if (swung && f.inRange) return ACT.ATTACK;

  // 2. Mining detection (swung at blocks / tree / ore without entity in reach)
  const isMiningTarget = f.targetIsTree || f.targetIsOre || f.wall || (f.aheadBlock && f.aheadBlock !== 'air');
  if (swung && isMiningTarget) return ACT.MINE;

  // 3. Sneaking / crouching
  if (cur.crouching) return ACT.SNEAK;

  // 4. Sprinting (fast forward movement)
  if (fwd > 0.22) return ACT.SPRINT;

  // 5. Hurdle Jump
  if (prev.onGround && dy > 0.2) return ACT.JUMP;

  // 6. Steering
  const turning = Math.abs(dyaw) > 0.12, moving = fwd > 0.10;
  if (turning && (!moving || Math.abs(f.rel) > 0.5)) return dyaw > 0 ? ACT.LEFT : ACT.RIGHT;

  // 7. Standard forward
  if (moving) return ACT.FWD;

  return ACT.IDLE;
}

class DemonstrationRecorder {
  constructor(opts = {}) {
    this.host = opts.host || process.env.MC_HOST || 'localhost';
    this.port = +(opts.port || process.env.MC_PORT || 25565);
    this.playerName = opts.playerName || process.env.PLAYER || 'ObserverTarget';
    this.task = opts.task || 'kill_skeleton';
    this.goal = opts.goal || null;
    this.demoDir = opts.demoDir || path.join(__dirname, 'demos');
    this.username = opts.username || 'Observer';

    fs.mkdirSync(this.demoDir, { recursive: true });
    this.bot = null;
    this.isRecording = false;
    this.samples = [];
    this.fileStream = null;
    this.currentFileName = null;
    this.intervalId = null;
    this.swung = false;
    this.prev = null;
    this.actionCounts = {
      idle: 0, forward: 0, left: 0, right: 0, jump: 0,
      attack: 0, mine: 0, use_item: 0, sneak: 0, sprint: 0
    };

    this.onSample = opts.onSample || (() => {});
    this.onLog = opts.onLog || (() => {});
  }

  log(msg) {
    const time = new Date().toLocaleTimeString();
    const entry = `[RECORDER ${time}] ${msg}`;
    console.log(entry);
    this.onLog(entry);
  }

  start() {
    if (this.isRecording) return;
    this.isRecording = true;
    this.samples = [];
    this.actionCounts = {
      idle: 0, forward: 0, left: 0, right: 0, jump: 0,
      attack: 0, mine: 0, use_item: 0, sneak: 0, sprint: 0
    };

    this.currentFileName = path.join(this.demoDir, `demo_${Date.now()}.jsonl`);
    this.fileStream = fs.createWriteStream(this.currentFileName, { flags: 'a' });

    this.log(`Connecting Observer bot to watch player "${this.playerName}" on ${this.host}:${this.port}...`);

    this.bot = mineflayer.createBot({
      host: this.host,
      port: this.port,
      username: this.username
    });

    this.bot.on('entitySwingArm', e => {
      const matchName = e.username?.toLowerCase() === this.playerName.toLowerCase();
      const onlyPlayer = Object.values(this.bot.players).filter(p => p.username !== this.bot.username).length === 1;
      if (matchName || (onlyPlayer && e.username !== this.bot.username)) {
        this.swung = true;
      }
    });

    this.bot.once('spawn', () => {
      try {
        this.bot.chat('/gamemode spectator');
      } catch {}
      setTimeout(() => {
        try {
          this.bot.chat(`/tp @s ${this.playerName}`);
        } catch {}
      }, 1000);
      this.log(`Observer spawned. Tracking player: ${this.playerName}. Task: ${this.task}`);
      this.startLoop();
    });

    this.bot.on('error', err => {
      this.log(`⚠️ Connection error: ${err.message}`);
      if (err.code === 'ECONNREFUSED') {
        this.log(`❌ Could not connect to Minecraft on ${this.host}:${this.port}. Please verify the port in Open to LAN.`);
      }
      this.stop();
    });

    this.bot.on('kicked', r => {
      let reasonStr = typeof r === 'string' ? r : JSON.stringify(r);
      this.log(`⚠️ Bot kicked from server: ${reasonStr}`);
      if (reasonStr.includes('Failed to verify username') || reasonStr.includes('Invalid credentials')) {
        this.log(`🔒 Mojang Auth Notice: Singleplayer LAN requires online accounts. For offline bots, use a local server with online-mode=false or an offline LAN mod.`);
      }
      this.stop();
    });
  }

  getPlayerEntity() {
    if (!this.bot || !this.bot.players) return null;
    const lower = this.playerName.toLowerCase();
    for (const name of Object.keys(this.bot.players)) {
      if (name.toLowerCase() === lower) {
        return this.bot.players[name]?.entity || null;
      }
    }
    // If exact name didn't match, find the first human player in world
    const others = Object.values(this.bot.players).filter(p => p.username !== this.bot.username);
    if (others.length === 1 && others[0].entity) {
      return others[0].entity;
    }
    return null;
  }

  startLoop() {
    if (this.intervalId) clearInterval(this.intervalId);
    let waitTicks = 0;

    this.intervalId = setInterval(() => {
      if (!this.bot || !this.isRecording) return;

      const playerEntity = this.getPlayerEntity();
      if (!playerEntity) {
        waitTicks++;
        if (waitTicks % 30 === 0) {
          try {
            this.bot.chat(`/tp @s ${this.playerName}`);
          } catch {}
          this.log(`Waiting for "${this.playerName}" to enter loaded chunks... Attempting teleport.`);
        }
        return;
      }

      const heldItem = playerEntity.heldItem || (playerEntity.equipment && playerEntity.equipment[0]) || null;
      const isCrouching = !!(playerEntity.metadata && (playerEntity.metadata[0] & 0x02));

      const cur = {
        pos: playerEntity.position.clone(),
        yaw: playerEntity.yaw,
        onGround: playerEntity.onGround,
        heldItem,
        crouching: isCrouching
      };

      if (this.prev) {
        const target = findTarget(this.bot, this.task, this.goal, playerEntity);
        const f = facts(this.bot, this.prev, target, this.task, {
          heldItem: this.prev.heldItem || heldItem,
          crouching: this.prev.crouching
        });
        const a = deriveAction(this.prev, cur, this.swung, f);
        const actionName = ACTIONS[a];
        this.actionCounts[actionName] = (this.actionCounts[actionName] || 0) + 1;

        const record = {
          x: makeObs(f),
          a,
          task: this.task,
          actionName,
          targetName: f.targetName,
          timestamp: Date.now()
        };

        this.samples.push(record);
        if (this.fileStream) {
          this.fileStream.write(JSON.stringify({ x: record.x, a: record.a, task: record.task }) + '\n');
        }

        this.onSample({
          count: this.samples.length,
          lastAction: actionName,
          actionCounts: this.actionCounts,
          target: f.targetName,
          dist: +f.dist.toFixed(1)
        });

        if (this.samples.length % 50 === 0) {
          this.log(`Recorded ${this.samples.length} demonstration samples from ${this.playerName}`);
        }
      }

      this.prev = cur;
      this.swung = false;
    }, 100);
  }

  stop() {
    this.isRecording = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    if (this.fileStream) {
      this.fileStream.end();
      this.fileStream = null;
    }
    if (this.bot) {
      try {
        this.bot.quit();
      } catch {}
      this.bot = null;
    }
    this.log(`Recording ended. Total samples saved: ${this.samples.length}. File: ${this.currentFileName}`);

    return {
      samples: this.samples,
      count: this.samples.length,
      file: this.currentFileName,
      actionCounts: this.actionCounts
    };
  }
}

// Support CLI run
if (require.main === module) {
  const playerName = process.argv[2] || process.env.PLAYER;
  if (!playerName) {
    console.log('Usage: node record.js YourMinecraftName [task]');
    process.exit(1);
  }
  const task = process.argv[3] || 'kill_skeleton';
  const recorder = new DemonstrationRecorder({ playerName, task });
  recorder.start();

  process.on('SIGINT', () => {
    console.log('\nStopping recorder...');
    const res = recorder.stop();
    console.log(`Saved ${res.count} samples.`);
    process.exit(0);
  });
}

module.exports = { DemonstrationRecorder };
