'use strict';
// objectives.js - Autonomous Objective & Role Execution Engine for Minecraft Bot.
// Handles role-based behaviors, Commander/Master player order-taking,
// and objective dispatching (mine, combat, player assassination, protect, explore, food, farming).

const { goals, Movements } = require('mineflayer-pathfinder');
const { GoalNear, GoalBlock, GoalFollow, GoalXZ } = goals;
const { PromptCommander } = require('./prompt_commander');

const ROLES = {
  BODYGUARD: 'bodyguard',
  MINER: 'miner',
  FARMER: 'farmer',
  LUMBERJACK: 'lumberjack',
  FOOD_GATHERER: 'food_gatherer',
  EXPLORER: 'explorer',
  HUNTER: 'hunter',
  AUTONOMOUS: 'autonomous'
};

class ObjectiveEngine {
  constructor(bot, knowledgeEngine, onLog = console.log, chestMemory = null) {
    this.bot = bot;
    this.knowledge = knowledgeEngine;
    this.log = msg => onLog(`[OBJECTIVE] ${msg}`);
    this.chestMemory = chestMemory;

    this.currentRole = ROLES.AUTONOMOUS;
    this.masterPlayer = null; // Assigned Commander from Web UI

    this.activeObjective = null;
    this.objectiveInterval = null;
    this.isExecuting = false;
    this.promptCommander = new PromptCommander(bot.username || 'Zombo', chestMemory);

    // Default movements config
    this.movements = null;
    this.initMovements();
  }

  initMovements() {
    if (!this.bot) return;
    try {
      const mcData = this.knowledge.mcData;
      this.movements = new Movements(this.bot, mcData);
      this.movements.canDig = true;
      this.movements.allow1by1towers = true;
      this.movements.scafoldingBlocks = [mcData.blocksByName.dirt?.id, mcData.blocksByName.cobblestone?.id].filter(Boolean);
      if (this.bot.pathfinder) {
        this.bot.pathfinder.setMovements(this.movements);
      }
    } catch (e) {
      this.log(`Movements init notice: ${e.message}`);
    }
  }

  setRole(roleName, master = null) {
    const validRoles = Object.values(ROLES);
    const role = roleName ? roleName.toLowerCase() : ROLES.AUTONOMOUS;
    if (validRoles.includes(role)) {
      this.currentRole = role;
      if (master) this.masterPlayer = master;
      this.log(`Role updated to: "${this.currentRole.toUpperCase()}" | Commander: "${this.masterPlayer || 'None'}"`);
      this.startRoleBehavior();
    }
  }

  setMaster(masterName) {
    this.masterPlayer = masterName ? masterName.trim() : null;
    this.log(`Commander assigned: "${this.masterPlayer || 'All Players'}"`);
  }

  // Equips weapon, tool, or shield from inventory
  async equipBest(pattern, destination = 'hand') {
    if (!this.bot || !this.bot.inventory) return;
    const re = pattern instanceof RegExp ? pattern : new RegExp(pattern, 'i');
    const items = this.bot.inventory.items().filter(i => re.test(i.name));
    if (!items.length) return;

    const tierOrder = ['netherite', 'diamond', 'iron', 'golden', 'stone', 'wooden'];
    items.sort((a, b) => {
      const aT = tierOrder.findIndex(t => a.name.includes(t));
      const bT = tierOrder.findIndex(t => b.name.includes(t));
      return (aT === -1 ? 99 : aT) - (bT === -1 ? 99 : bT);
    });

    try {
      await this.bot.equip(items[0], destination);
    } catch {}
  }

  stopCurrentObjective() {
    if (this.objectiveInterval) {
      clearInterval(this.objectiveInterval);
      this.objectiveInterval = null;
    }
    this.isExecuting = false;
    this.activeObjective = null;
    if (this.bot && this.bot.pathfinder) {
      try { this.bot.pathfinder.stop(); } catch {}
    }
    this.log('Active objective stopped. Bot standing by.');
  }

  // 1. Dispatch Mining Objective
  async startMining(blockPattern = 'iron_ore', targetCount = 10) {
    this.stopCurrentObjective();
    this.activeObjective = {
      type: 'mine',
      target: blockPattern,
      targetCount: +targetCount || 10,
      collected: 0,
      status: 'SEARCHING'
    };
    this.log(`Dispatched Mining Objective: Gather ${targetCount}x "${blockPattern}"`);

    this.objectiveInterval = setInterval(async () => {
      if (this.isExecuting || !this.activeObjective) return;
      this.isExecuting = true;

      try {
        const re = new RegExp(this.activeObjective.target, 'i');
        const block = this.bot.findBlock({
          matching: b => b && re.test(b.name),
          maxDistance: 32
        });

        if (!block) {
          this.activeObjective.status = 'SCOUTING';
          this.log(`No "${this.activeObjective.target}" within 32 blocks. Exploring area...`);
          // Wander to discover new chunks
          const p = this.bot.entity.position;
          const rx = p.x + (Math.random() - 0.5) * 40;
          const rz = p.z + (Math.random() - 0.5) * 40;
          await this.bot.pathfinder.goto(new GoalNear(rx, p.y, rz, 2));
          this.isExecuting = false;
          return;
        }

        this.activeObjective.status = `MINING (${block.name} at ${block.position.x}, ${block.position.y}, ${block.position.z})`;
        // Equip best pickaxe/tool
        await this.equipBest(/pickaxe/i);

        // Pathfind to block
        await this.bot.pathfinder.goto(new GoalBlock(block.position.x, block.position.y, block.position.z));
        if (this.bot.canDigBlock(block)) {
          await this.bot.dig(block);
          this.activeObjective.collected++;
          this.log(`Mined ${block.name}! (${this.activeObjective.collected}/${this.activeObjective.targetCount})`);

          // Collect drop
          await new Promise(r => setTimeout(r, 400));
          const dropped = this.bot.nearestEntity(e => e.name === 'item' && e.position.distanceTo(block.position) < 5);
          if (dropped) {
            await this.bot.pathfinder.goto(new GoalNear(dropped.position.x, dropped.position.y, dropped.position.z, 0.5));
          }

          if (this.activeObjective.collected >= this.activeObjective.targetCount) {
            this.log(`✅ Objective Completed: Mined ${this.activeObjective.collected}x "${this.activeObjective.target}"!`);
            this.stopCurrentObjective();
          }
        }
      } catch (err) {
        this.log(`Mining notice: ${err.message}`);
      } finally {
        this.isExecuting = false;
      }
    }, 1500);
  }

  // 2. Dispatch Combat / Player Kill Objective
  async startCombat(targetName, isPlayer = false) {
    this.stopCurrentObjective();
    this.activeObjective = {
      type: isPlayer ? 'kill_player' : 'kill_mob',
      target: targetName,
      status: 'HUNTING'
    };
    this.log(`Dispatched Combat Objective: Hunt & Eliminate ${isPlayer ? 'Player' : 'Mob'} "${targetName}"`);

    this.objectiveInterval = setInterval(async () => {
      if (this.isExecuting || !this.activeObjective) return;
      this.isExecuting = true;

      try {
        let targetEntity = null;
        if (isPlayer) {
          const player = this.bot.players[targetName];
          targetEntity = player ? player.entity : null;
        } else {
          const re = new RegExp(targetName, 'i');
          targetEntity = this.bot.nearestEntity(e => (e.type === 'mob' || e.type === 'hostile') && re.test(e.name || ''));
        }

        if (!targetEntity) {
          this.activeObjective.status = 'TRACKING (Target not in line of sight)';
          this.isExecuting = false;
          return;
        }

        this.activeObjective.status = `ENGAGING ${targetName} (${targetEntity.position.distanceTo(this.bot.entity.position).toFixed(1)}m)`;
        await this.equipBest(/sword|axe/i);
        await this.equipBest(/shield/i, 'off-hand');

        const dist = targetEntity.position.distanceTo(this.bot.entity.position);
        if (dist > 3.0) {
          this.bot.pathfinder.setGoal(new GoalFollow(targetEntity, 2.5), true);
        } else {
          // Look & Attack with critical strike jump
          await this.bot.lookAt(targetEntity.position.offset(0, targetEntity.height * 0.8, 0));
          if (this.bot.entity.onGround) this.bot.setControlState('jump', true);
          this.bot.attack(targetEntity);
          this.bot.setControlState('jump', false);
        }
      } catch (err) {
        this.log(`Combat notice: ${err.message}`);
      } finally {
        this.isExecuting = false;
      }
    }, 300);
  }

  // 3. Dispatch Bodyguard / Protect Me & Follow
  async startBodyguard(masterName) {
    this.stopCurrentObjective();
    const commander = masterName || this.masterPlayer;
    if (!commander) {
      this.log('Cannot start Bodyguard mode: No Commander player specified!');
      return;
    }
    this.masterPlayer = commander;
    this.currentRole = ROLES.BODYGUARD;

    this.activeObjective = {
      type: 'protect_and_follow',
      target: commander,
      status: 'GUARDING'
    };
    this.log(`🛡️ Bodyguard Mode Activated! Protecting Commander "${commander}"`);

    this.objectiveInterval = setInterval(async () => {
      if (this.isExecuting || !this.activeObjective) return;
      this.isExecuting = true;

      try {
        const player = this.bot.players[commander];
        const playerEntity = player ? player.entity : null;

        if (!playerEntity) {
          this.activeObjective.status = `Waiting for Commander "${commander}" to enter render distance...`;
          this.isExecuting = false;
          return;
        }

        // 1. Scan for any hostile mob targeting or close to master
        const threat = this.bot.nearestEntity(e => {
          if (!e || e.type !== 'mob') return false;
          const dToMaster = e.position.distanceTo(playerEntity.position);
          return dToMaster < 8.0 && /zombie|skeleton|creeper|spider|pillager|enderman|witch/i.test(e.name || '');
        });

        if (threat) {
          this.activeObjective.status = `INTERCEPTING THREAT: ${threat.name} threatening ${commander}!`;
          await this.equipBest(/sword|axe/i);
          const dist = threat.position.distanceTo(this.bot.entity.position);
          if (dist > 3.0) {
            this.bot.pathfinder.setGoal(new GoalFollow(threat, 2), true);
          } else {
            await this.bot.lookAt(threat.position.offset(0, threat.height * 0.7, 0));
            this.bot.attack(threat);
          }
        } else {
          // Follow master closely (stay within 3 blocks)
          this.activeObjective.status = `Escorting ${commander}`;
          const distToMaster = playerEntity.position.distanceTo(this.bot.entity.position);
          if (distToMaster > 4.0) {
            this.bot.pathfinder.setGoal(new GoalFollow(playerEntity, 3), true);
          }
        }
      } catch (err) {
        this.log(`Bodyguard notice: ${err.message}`);
      } finally {
        this.isExecuting = false;
      }
    }, 350);
  }

  // 4. Dispatch Coordinate Navigation
  async startGoTo(x, y, z) {
    this.stopCurrentObjective();
    this.activeObjective = {
      type: 'go_to',
      target: { x: +x, y: +y, z: +z },
      status: 'NAVIGATING'
    };
    this.log(`Navigating to coordinates: X=${x}, Y=${y}, Z=${z}...`);

    try {
      await this.bot.pathfinder.goto(new GoalNear(+x, +y, +z, 1));
      this.log(`🏁 Arrived at coordinates X=${x}, Y=${y}, Z=${z}!`);
      this.stopCurrentObjective();
    } catch (err) {
      this.log(`Navigation notice: ${err.message}`);
      this.stopCurrentObjective();
    }
  }

  // 5. Dispatch Find a Village
  async startFindVillage() {
    this.stopCurrentObjective();
    this.activeObjective = {
      type: 'find_village',
      status: 'SCOUTING'
    };
    this.log('🧭 Scouting world to locate nearest Village...');

    this.objectiveInterval = setInterval(async () => {
      if (this.isExecuting || !this.activeObjective) return;
      this.isExecuting = true;

      try {
        // Village signature blocks
        const villageBlock = this.bot.findBlock({
          matching: b => b && /bell|composter|lectern|smoker|blast_furnace|brewing_stand|stonecutter/i.test(b.name),
          maxDistance: 48
        });

        // Or village entities (villager or iron golem)
        const villager = this.bot.nearestEntity(e => e && /villager|iron_golem/i.test(e.name || ''));

        if (villageBlock || villager) {
          const loc = villageBlock ? villageBlock.position : villager.position;
          this.log(`🎉 VILLAGE DISCOVERED at X:${Math.round(loc.x)}, Y:${Math.round(loc.y)}, Z:${Math.round(loc.z)}!`);
          await this.bot.pathfinder.goto(new GoalNear(loc.x, loc.y, loc.z, 3));
          this.stopCurrentObjective();
          return;
        }

        // Spiral search exploration
        const p = this.bot.entity.position;
        const angle = Math.random() * Math.PI * 2;
        const destX = p.x + Math.cos(angle) * 35;
        const destZ = p.z + Math.sin(angle) * 35;
        this.activeObjective.status = `Scouting perimeter (Current X:${Math.round(p.x)}, Z:${Math.round(p.z)})...`;
        await this.bot.pathfinder.goto(new GoalNear(destX, p.y, destZ, 2));
      } catch (err) {
        this.log(`Village search notice: ${err.message}`);
      } finally {
        this.isExecuting = false;
      }
    }, 2000);
  }

  // 6. Dispatch Collect Food & Bring to Player
  async startCollectFood() {
    this.stopCurrentObjective();
    this.activeObjective = {
      type: 'collect_food',
      status: 'HUNTING_FOOD'
    };
    this.log('🍗 Searching for food (Hunting animals & harvesting crops) to deliver to Commander...');

    this.objectiveInterval = setInterval(async () => {
      if (this.isExecuting || !this.activeObjective) return;
      this.isExecuting = true;

      try {
        // Count food in inventory
        const foodCount = this.bot.inventory.items()
          .filter(i => /beef|porkchop|chicken|mutton|bread|apple|carrot|potato/i.test(i.name))
          .reduce((s, i) => s + i.count, 0);

        if (foodCount >= 6) {
          // Food acquired! Deliver to master player
          if (this.masterPlayer && this.bot.players[this.masterPlayer]?.entity) {
            const masterEnt = this.bot.players[this.masterPlayer].entity;
            this.activeObjective.status = `Delivering ${foodCount} food items to ${this.masterPlayer}...`;
            await this.bot.pathfinder.goto(new GoalNear(masterEnt.position.x, masterEnt.position.y, masterEnt.position.z, 2));
            await this.bot.lookAt(masterEnt.position.offset(0, 1.2, 0));

            // Toss food items
            for (const it of this.bot.inventory.items()) {
              if (/beef|porkchop|chicken|mutton|bread|apple|carrot|potato/i.test(it.name)) {
                await this.bot.toss(it.type, null, it.count);
              }
            }
            this.log(`🎁 Delivered rations to Commander ${this.masterPlayer}!`);
            this.stopCurrentObjective();
            return;
          }
        }

        // Hunt nearby food animal
        const animal = this.bot.nearestEntity(e => e && /cow|pig|sheep|chicken/i.test(e.name || ''));
        if (animal) {
          this.activeObjective.status = `Hunting ${animal.name} for food...`;
          await this.equipBest(/sword|axe/i);
          if (animal.position.distanceTo(this.bot.entity.position) > 2.5) {
            await this.bot.pathfinder.goto(new GoalFollow(animal, 2));
          } else {
            await this.bot.lookAt(animal.position);
            this.bot.attack(animal);
          }
          return;
        }

        // Or harvest mature wheat (age 7)
        const ripeCrop = this.bot.findBlock({
          matching: b => b && /wheat|carrots|potatoes/i.test(b.name) && b.metadata === 7,
          maxDistance: 24
        });
        if (ripeCrop) {
          this.activeObjective.status = `Harvesting ripe ${ripeCrop.name}...`;
          await this.bot.pathfinder.goto(new GoalBlock(ripeCrop.position.x, ripeCrop.position.y, ripeCrop.position.z));
          await this.bot.dig(ripeCrop);
          return;
        }

        // Roam to find animals
        const p = this.bot.entity.position;
        await this.bot.pathfinder.goto(new GoalNear(p.x + (Math.random() - 0.5) * 30, p.y, p.z + (Math.random() - 0.5) * 30, 2));
      } catch (err) {
        this.log(`Collect food notice: ${err.message}`);
      } finally {
        this.isExecuting = false;
      }
    }, 1500);
  }

  // 7. Dispatch Build / Tend Farm
  async startFarming() {
    this.stopCurrentObjective();
    this.activeObjective = {
      type: 'farm',
      status: 'TENDING_FARM'
    };
    this.log('🌾 Farming Mode: Tilling soil, planting seeds, and harvesting ripe crops...');

    this.objectiveInterval = setInterval(async () => {
      if (this.isExecuting || !this.activeObjective) return;
      this.isExecuting = true;

      try {
        // 1. Harvest ripe crops first
        const ripe = this.bot.findBlock({
          matching: b => b && /wheat|carrots|potatoes|beetroots/i.test(b.name) && b.metadata === 7,
          maxDistance: 16
        });
        if (ripe) {
          this.activeObjective.status = `Harvesting ripe ${ripe.name}`;
          await this.bot.pathfinder.goto(new GoalBlock(ripe.position.x, ripe.position.y, ripe.position.z));
          await this.bot.dig(ripe);
          return;
        }

        // 2. Plant seeds on empty farmland
        const seeds = this.bot.inventory.items().find(i => /seeds|carrot|potato/i.test(i.name));
        const emptyFarmland = this.bot.findBlock({
          matching: b => b && b.name === 'farmland',
          maxDistance: 16
        });

        if (seeds && emptyFarmland) {
          // Check block above is air
          const above = this.bot.blockAt(emptyFarmland.position.offset(0, 1, 0));
          if (above && above.name === 'air') {
            this.activeObjective.status = `Planting ${seeds.name}`;
            await this.bot.equip(seeds, 'hand');
            await this.bot.pathfinder.goto(new GoalNear(emptyFarmland.position.x, emptyFarmland.position.y, emptyFarmland.position.z, 2));
            await this.bot.activateBlock(emptyFarmland);
            return;
          }
        }

        // 3. Till dirt near water if hoe available
        const hoe = this.bot.inventory.items().find(i => /hoe/i.test(i.name));
        const water = this.bot.findBlock({ matching: b => b && b.name === 'water', maxDistance: 12 });
        if (hoe && water) {
          const dirt = this.bot.findBlock({
            matching: b => b && (b.name === 'dirt' || b.name === 'grass_block') && b.position.distanceTo(water.position) <= 4,
            maxDistance: 12
          });
          if (dirt) {
            this.activeObjective.status = 'Tilling dirt into farmland';
            await this.bot.equip(hoe, 'hand');
            await this.bot.pathfinder.goto(new GoalNear(dirt.position.x, dirt.position.y, dirt.position.z, 2));
            await this.bot.activateBlock(dirt);
            return;
          }
        }

        this.activeObjective.status = 'Farms in order. Patrolling fields...';
      } catch (err) {
        this.log(`Farming notice: ${err.message}`);
      } finally {
        this.isExecuting = false;
      }
    }, 2000);
  }

  // Handle in-game chat commands from Commander/Master player (Natural English Prompts & Legacy ! Commands)
  handleChatCommand(sender, message) {
    if (this.masterPlayer && sender !== this.masterPlayer) {
      return;
    }

    const onlinePlayerNames = this.bot && this.bot.players ? Object.keys(this.bot.players) : [];
    const botName = (this.bot && this.bot.username) || 'Zombo';
    const parsed = this.promptCommander.parsePrompt(message, botName, sender, onlinePlayerNames);

    if (parsed.success) {
      this.promptCommander.executeCommand(parsed, this, (replyMsg) => {
        if (this.bot && typeof this.bot.chat === 'function') {
          this.bot.chat(replyMsg);
        }
      });
      return;
    }

    // Fallback to legacy ! commands
    const msg = message.trim();
    if (msg === '!follow' || msg === '!protect') {
      this.startBodyguard(sender);
    } else if (msg.startsWith('!mine ')) {
      const parts = msg.split(' ');
      const block = parts[1] || 'iron_ore';
      const count = parseInt(parts[2] || '10', 10);
      this.startMining(block, count);
    } else if (msg.startsWith('!kill ')) {
      const target = msg.replace('!kill ', '').trim();
      const isPlayer = !!this.bot.players[target];
      this.startCombat(target, isPlayer);
    } else if (msg === '!food') {
      this.startCollectFood();
    } else if (msg === '!farm') {
      this.startFarming();
    } else if (msg === '!village') {
      this.startFindVillage();
    } else if (msg.startsWith('!goto ')) {
      const parts = msg.split(' ');
      if (parts.length >= 4) {
        this.startGoTo(+parts[1], +parts[2], +parts[3]);
      }
    } else if (msg === '!stop') {
      this.stopCurrentObjective();
    }
  }

  startRoleBehavior() {
    switch (this.currentRole) {
      case ROLES.BODYGUARD:
        if (this.masterPlayer) this.startBodyguard(this.masterPlayer);
        break;
      case ROLES.MINER:
        this.startMining('iron_ore', 16);
        break;
      case ROLES.FARMER:
        this.startFarming();
        break;
      case ROLES.LUMBERJACK:
        this.startMining('log', 16);
        break;
      case ROLES.FOOD_GATHERER:
        this.startCollectFood();
        break;
      case ROLES.EXPLORER:
        this.startFindVillage();
        break;
      default:
        this.stopCurrentObjective();
        break;
    }
  }

  getStatus() {
    return {
      currentRole: this.currentRole,
      masterPlayer: this.masterPlayer,
      activeObjective: this.activeObjective
    };
  }
}

module.exports = { ObjectiveEngine, ROLES };
