'use strict';
// bot.js - Minecraft Survival Bot powered by the Neuro-Evolutionary Brain.
// Can be run from CLI or imported and controlled by server.js.

const mineflayer = require('mineflayer');
const path = require('path');
const { pathfinder } = require('mineflayer-pathfinder');
const { Brain, makeObs, stepReward, ACT, ACTIONS } = require('./brain');
const { findTarget, facts, HOSTILE_MOBS } = require('./features');
const { SurvivalManager } = require('./survival');
const { SurvivalPlanner } = require('./planner');
const { MinecraftKnowledgeEngine } = require('./knowledge');
const { ObjectiveEngine, ROLES } = require('./objectives');
const { ChestMemoryEngine } = require('./chest_memory');

class SurvivalBot {
  constructor(opts = {}) {
    this.host = opts.host || process.env.MC_HOST || 'localhost';
    this.port = +(opts.port || process.env.MC_PORT || 25565);
    this.username = opts.username || process.env.MC_USER || 'Zombo';
    this.version = opts.version || process.env.MC_VERSION || false;
    this.task = opts.task || 'survive'; // 'survive', 'kill_skeleton', 'kill_zombie', 'kill_mob', 'chase_player', 'go_to'
    this.goal = opts.goal || null;
    this.dataDir = opts.dataDir || path.join(__dirname, 'data_bot');
    this.viewerPort = +(opts.viewerPort || 3007);
    this.enableViewer = opts.enableViewer !== false;

    this.brain = new Brain(this.dataDir);
    this.chestMemory = new ChestMemoryEngine(this.dataDir);
    this.bot = null;
    this.survival = null;
    this.planner = null;
    this.knowledge = new MinecraftKnowledgeEngine(this.version || '1.20.1');
    this.objectives = null;
    this.viewer = null;
    this.intervalId = null;
    this.isDigging = false;

    this.currentAction = 'idle';
    this.actionProb = 0;
    this.lifeSteps = [];
    this.prevStep = null;
    this.recentLogs = [];
    this.recentDecisions = [];
    this.isRunning = false;

    this.onTelemetry = opts.onTelemetry || (() => {});
    this.onLog = opts.onLog || (() => {});
  }

  log(msg) {
    const time = new Date().toLocaleTimeString();
    const entry = `[${time}] ${msg}`;
    console.log(entry);
    this.recentLogs.push(entry);
    if (this.recentLogs.length > 50) this.recentLogs.shift();
    this.onLog(entry);
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.log(`Connecting to Minecraft server at ${this.host}:${this.port} as "${this.username}"...`);

    this.bot = mineflayer.createBot({
      host: this.host,
      port: this.port,
      username: this.username,
      ...(this.version ? { version: this.version } : {})
    });

    this.bot.loadPlugin(pathfinder);
    this.survival = new SurvivalManager(this.bot, msg => this.log(msg));

    this.bot.once('spawn', () => {
      this.brain.net.resetState();
      this.knowledge.setVersion(this.bot.version);
      this.planner = new SurvivalPlanner(this.bot, msg => this.log(msg));
      this.objectives = new ObjectiveEngine(this.bot, this.knowledge, msg => this.log(msg), this.chestMemory);
      this.chestMemory.attachToBot(this.bot, msg => this.log(msg));
      this.log(`Bot spawned successfully in world! Game Mode: ${this.bot.game.gameMode} | Version: ${this.bot.version}`);

      // Start 3D Prismarine viewer if enabled
      if (this.enableViewer) {
        try {
          if (this.bot.viewer && typeof this.bot.viewer.close === 'function') {
            try { this.bot.viewer.close(); } catch {}
          }
          const { mineflayer: prismarineViewer } = require('prismarine-viewer');
          prismarineViewer(this.bot, { port: this.viewerPort, firstPerson: true });
          this.log(`3D First-Person Game Viewer live at http://localhost:${this.viewerPort}`);
        } catch (err) {
          this.log(`Prismarine viewer notice: ${err.message}`);
        }
      }

      this.survival.autoEquipGear();
      this.startLoop();
    });

    this.bot.on('death', () => {
      this.log('Bot was defeated in survival mode!');
      if (this.survival) {
        this.survival.status = 'DEAD';
        this.survival.stopReason = '💀 Defeated in combat. Running death autopsy and neurogenesis...';
      }

      // Evolve Brain A into Brain B from this life's experience
      const rep = this.brain.endLife(this.lifeSteps, 'killed_in_combat');
      if (rep) {
        this.log(`[EVOLUTION] Generation ${this.brain.generation} formed! Kept: ${rep.kept}, Cut: ${rep.cut}, Added: ${rep.added} new neurons. Total neurons: ${rep.hiddenAfter}`);
      }

      this.lifeSteps = [];
      this.prevStep = null;

      setTimeout(() => {
        if (this.bot && this.isRunning) {
          try {
            this.bot.respawn();
            this.brain.net.resetState();
            this.log('Bot respawned with newly evolved brain and reset RNN memory!');
          } catch {}
        }
      }, 1500);
    });

    this.bot.on('chat', (username, message) => {
      if (username === this.bot.username) return;
      if (this.objectives) {
        this.objectives.handleChatCommand(username, message);
      }
      if (message.startsWith('!task ')) {
        const newTask = message.split(' ')[1];
        this.task = newTask;
        this.log(`Task switched to: ${newTask}`);
      } else if (message === '!eat') {
        this.survival.checkAndEat();
      } else if (message === '!equip') {
        this.survival.autoEquipGear();
      }
    });

    this.bot.on('kicked', reason => {
      const reasonStr = typeof reason === 'string' ? reason : JSON.stringify(reason);
      this.log(`⚠️ Bot kicked from server: ${reasonStr}`);
      if (reasonStr.includes('Failed to verify username') || reasonStr.includes('Invalid credentials')) {
        this.log(`🔒 Mojang Auth Notice: Server rejected offline bot. Make sure online-mode=false on your server!`);
      }
      this.stop();
    });

    this.bot.on('error', err => {
      this.log(`⚠️ Bot network error: ${err.message}`);
      if (err.code === 'ECONNREFUSED') {
        this.log(`❌ Could not connect to Minecraft on ${this.host}:${this.port}. Check if world is open.`);
      }
      this.stop();
    });
  }

  startLoop() {
    if (this.intervalId) clearInterval(this.intervalId);

    this.intervalId = setInterval(async () => {
      if (!this.bot || !this.bot.entity) return;

      // 1. Survival reflexes
      await this.survival.checkWaterClutch();
      const isEating = await this.survival.checkAndEat();
      if (isEating || this.survival.isUnstucking) {
        this.emitTelemetry();
        return;
      }

      // 2. Sense surroundings
      const target = findTarget(this.bot, this.task, this.goal);
      const f = facts(this.bot, this.bot.entity, target, this.task);

      // 3. Calculate reward for previous step if in combat/learning
      if (this.prevStep) {
        const killed = !target && this.prevStep.f.hasTarget && this.prevStep.f.dist < 3.5;
        this.prevStep.step.r = stepReward(this.prevStep.f, f, this.prevStep.step.a, { killed });
      }

      // 4. Stuck & Stop Diagnostics
      this.survival.checkStuck(this.currentAction, f);
      if (this.survival.isStuck) {
        this.survival.tryUnstuck(f);
        this.emitTelemetry(f);
        return;
      }

      // 5. Brain decision & explainability
      const x = makeObs(f);
      const explanation = this.brain.explain(x);
      const { a } = this.brain.act(x);

      this.currentAction = ACTIONS[a];
      this.actionProb = explanation.confidence;

      const step = { x, a, r: 0 };
      this.lifeSteps.push(step);
      this.prevStep = { f, step };

      // Keep recent decision log
      this.recentDecisions.unshift({
        time: new Date().toLocaleTimeString(),
        action: this.currentAction,
        confidence: this.actionProb,
        target: f.targetName,
        dist: +f.dist.toFixed(1),
        topNeurons: explanation.topNeurons
      });
      if (this.recentDecisions.length > 20) this.recentDecisions.pop();

      // 6. Execute action in Minecraft
      this.applyAction(a, target, f);

      // 7. Emit live telemetry
      this.emitTelemetry(f, explanation);
    }, 100);
  }

  async equipTool(pattern) {
    if (!this.bot || !this.bot.inventory) return;
    const re = typeof pattern === 'string' ? new RegExp(pattern, 'i') : pattern;
    const item = this.bot.inventory.items().find(i => re.test(i.name));
    if (item && (!this.bot.heldItem || !re.test(this.bot.heldItem.name))) {
      try {
        await this.bot.equip(item, 'hand');
      } catch {}
    }
  }

  async applyAction(a, target, f) {
    if (!this.bot || !this.bot.entity) return;

    // Turn toward target if target exists
    if (target && target.position) {
      const tp = target.position;
      const dx = tp.x - this.bot.entity.position.x;
      const dz = tp.z - this.bot.entity.position.z;
      const targetYaw = Math.atan2(-dx, -dz);
      this.bot.look(targetYaw, 0, true);
    }

    if (a === ACT.IDLE) {
      this.bot.clearControlStates();
    } else if (a === ACT.FWD) {
      this.bot.setControlState('forward', true);
      this.bot.setControlState('left', false);
      this.bot.setControlState('right', false);
      this.bot.setControlState('jump', false);
      this.bot.setControlState('sneak', false);
    } else if (a === ACT.LEFT) {
      this.bot.entity.yaw += 0.25;
      this.bot.setControlState('forward', true);
    } else if (a === ACT.RIGHT) {
      this.bot.entity.yaw -= 0.25;
      this.bot.setControlState('forward', true);
    } else if (a === ACT.JUMP) {
      this.bot.setControlState('jump', true);
      this.bot.setControlState('forward', true);
      setTimeout(() => {
        if (this.bot) this.bot.setControlState('jump', false);
      }, 250);
    } else if (a === ACT.ATTACK) {
      // Auto-equip sword for combat!
      await this.equipTool(/sword/i);
      if (target && f.inRange) {
        this.bot.attack(target);
      } else {
        this.bot.swingArm();
      }
    } else if (a === ACT.MINE) {
      // Auto-equip axe for trees, pickaxe for stone/ore
      if (f.targetIsTree) {
        await this.equipTool(/axe/i);
      } else {
        await this.equipTool(/pickaxe/i);
      }

      if (!this.isDigging) {
        const fx = -Math.sin(this.bot.entity.yaw), fz = -Math.cos(this.bot.entity.yaw);
        const ahead = this.bot.blockAt(this.bot.entity.position.offset(fx, 0, fz));
        let blockTarget = (ahead && ahead.boundingBox === 'block') ? ahead : null;

        if (!blockTarget && f.targetIsTree && this.planner) {
          blockTarget = this.planner.findNearestBlock(/log|wood|stem/, 4);
        } else if (!blockTarget && f.targetIsOre && this.planner) {
          blockTarget = this.planner.findNearestBlock(/stone|cobble|ore/, 4);
        }

        if (blockTarget && this.bot.canDigBlock(blockTarget)) {
          this.isDigging = true;
          this.bot.dig(blockTarget)
            .catch(() => {})
            .finally(() => { this.isDigging = false; });
        } else {
          this.bot.swingArm();
        }
      }
    } else if (a === ACT.USE_ITEM) {
      if (f.invFood > 0) {
        await this.equipTool(/bread|apple|beef|porkchop|chicken|mutton|cookie|carrot|potato|melon/i);
      }
      try {
        this.bot.activateItem();
      } catch {}
    } else if (a === ACT.SNEAK) {
      this.bot.setControlState('sneak', true);
      this.bot.setControlState('forward', true);
    } else if (a === ACT.SPRINT) {
      this.bot.setControlState('sprint', true);
      this.bot.setControlState('forward', true);
    }
  }

  emitTelemetry(f = {}, explanation = {}) {
    if (!this.bot || !this.bot.entity) return;

    const pos = this.bot.entity.position || { x: 0, y: 0, z: 0 };
    const held = this.bot.heldItem ? this.bot.heldItem.name : 'empty';

    // Extract complete 46-slot inventory array
    const inventorySlots = [];
    if (this.bot && this.bot.inventory && Array.isArray(this.bot.inventory.slots)) {
      for (let i = 0; i < 46; i++) {
        const item = this.bot.inventory.slots[i];
        if (item) {
          inventorySlots.push({
            slot: i,
            name: item.name,
            displayName: item.displayName || item.name,
            count: item.count,
            maxStackSize: item.maxStackSize || 64
          });
        } else {
          inventorySlots.push(null);
        }
      }
    }

    const onlinePlayers = (this.bot && this.bot.players) ? Object.keys(this.bot.players) : [];
    const techTreeReport = (this.knowledge && this.bot && this.bot.inventory) ?
      this.knowledge.evaluateTechTree(this.bot.inventory.items()) : null;
    const objectiveReport = this.objectives ? this.objectives.getStatus() : null;

    const telemetry = {
      online: true,
      username: this.bot.username,
      health: this.bot.health ?? 20,
      food: this.bot.food ?? 20,
      oxygen: this.bot.oxygenLevel ?? 20,
      armor: this.bot.experience?.points || 0,
      xpLevel: this.bot.experience?.level || 0,
      pos: {
        x: +pos.x.toFixed(2),
        y: +pos.y.toFixed(2),
        z: +pos.z.toFixed(2)
      },
      yaw: +(this.bot.entity.yaw || 0).toFixed(2),
      pitch: +(this.bot.entity.pitch || 0).toFixed(2),
      heldItem: held,
      inventorySlots,
      onlinePlayers,
      techTree: techTreeReport,
      objective: objectiveReport,
      task: this.task,
      target: {
        name: f.targetName || 'none',
        dist: +(f.dist || 0).toFixed(1),
        inRange: !!f.inRange
      },
      currentAction: this.currentAction,
      actionProb: this.actionProb,
      stuckStatus: this.survival ? this.survival.status : 'ACTIVE',
      stopReason: this.survival ? this.survival.stopReason : null,
      brainGen: this.brain.generation,
      hiddenNeurons: this.brain.net.H,
      activeNeurons: explanation.activeNeurons || 0,
      rnnMemoryActive: (this.brain.net.h_state || []).filter(v => Math.abs(v) > 0.2).length,
      isRNN: true,
      topNeurons: explanation.topNeurons || [],
      recentLogs: this.recentLogs.slice(-10),
      recentDecisions: this.recentDecisions.slice(0, 5),
      viewerPort: this.viewerPort,
      plannerStatus: this.planner ? this.planner.getStatus(f) : null
    };

    this.onTelemetry(telemetry);
  }

  stop() {
    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    if (this.bot) {
      try {
        if (this.bot.viewer && typeof this.bot.viewer.close === 'function') {
          this.bot.viewer.close();
        }
      } catch {}
      try {
        this.bot.quit();
      } catch {}
      this.bot = null;
    }
    this.log('Survival Bot disconnected and stopped.');
  }
}

// Support running standalone from CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  const getArg = (flag, def) => {
    const idx = args.indexOf(flag);
    return idx !== -1 && args[idx + 1] ? args[idx + 1] : def;
  };

  const host = getArg('--host', 'localhost');
  const port = +getArg('--port', 25565);
  const username = getArg('--username', 'SurvivalBot');
  const task = getArg('--task', 'survive');

  const bot = new SurvivalBot({ host, port, username, task });
  bot.start();

  process.on('SIGINT', () => {
    console.log('\nStopping bot...');
    bot.stop();
    process.exit(0);
  });
}

module.exports = { SurvivalBot };
