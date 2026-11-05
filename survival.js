'use strict';
// survival.js - Autonomous survival reflexes and stuck diagnostics for Minecraft bots.
// Handles auto-eating, gear management, stuck detection, and survival instincts.

const FOOD_PRIORITY = [
  'golden_apple', 'enchanted_golden_apple',
  'cooked_beef', 'cooked_porkchop', 'cooked_mutton',
  'cooked_salmon', 'cooked_chicken', 'baked_potato',
  'bread', 'cooked_cod', 'carrot', 'apple',
  'melon_slice', 'sweet_berries'
];

const WEAPON_PRIORITY = [
  'netherite_sword', 'diamond_sword', 'iron_sword', 'stone_sword', 'golden_sword', 'wooden_sword',
  'netherite_axe', 'diamond_axe', 'iron_axe', 'stone_axe', 'wooden_axe'
];

const ARMOR_SLOTS = {
  head: ['netherite_helmet', 'diamond_helmet', 'iron_helmet', 'chainmail_helmet', 'golden_helmet', 'leather_helmet'],
  torso: ['netherite_chestplate', 'diamond_chestplate', 'iron_chestplate', 'chainmail_chestplate', 'golden_chestplate', 'leather_chestplate'],
  legs: ['netherite_leggings', 'diamond_leggings', 'iron_leggings', 'chainmail_leggings', 'golden_leggings', 'leather_leggings'],
  feet: ['netherite_boots', 'diamond_boots', 'iron_boots', 'chainmail_boots', 'golden_boots', 'leather_boots']
};

class SurvivalManager {
  constructor(bot, logger = console.log) {
    this.bot = bot;
    this.log = logger;
    this.isEating = false;
    this.lastAteTime = 0;
    this.posHistory = [];
    this.stuckCount = 0;
    this.isStuck = false;
    this.isUnstucking = false;
    this.lastUnstuckTime = 0;
    this.stopReason = null;
    this.status = 'IDLE'; // 'IDLE' | 'ACTIVE' | 'EATING' | 'STUCK' | 'DEAD' | 'COMBAT'
  }

  // Update position history and check whether bot is physically stuck
  checkStuck(intendedAction, f) {
    if (!this.bot.entity || this.isUnstucking) return;

    const curPos = this.bot.entity.position.clone();
    this.posHistory.push({ pos: curPos, time: Date.now(), action: intendedAction });
    if (this.posHistory.length > 20) this.posHistory.shift();

    if (this.bot.health <= 0) {
      this.status = 'DEAD';
      this.stopReason = '💀 Bot died in survival mode';
      return;
    }

    if (this.isEating) {
      this.status = 'EATING';
      this.stopReason = '🍗 Consuming food to restore health/hunger';
      return;
    }

    // If bot is in melee reach, it is intentionally in combat, NOT stuck!
    if (f.inRange) {
      this.status = 'COMBAT';
      this.stopReason = `⚔️ Attacking ${f.targetName} in melee range`;
      this.isStuck = false;
      this.stuckCount = 0;
      return;
    }

    // Check if bot was trying to advance (forward / jump) but barely moved
    if (this.posHistory.length >= 12) {
      const oldest = this.posHistory[0];
      const newest = this.posHistory[this.posHistory.length - 1];
      const dx = newest.pos.x - oldest.pos.x;
      const dz = newest.pos.z - oldest.pos.z;
      const distMoved = Math.hypot(dx, dz);

      const wasTryingToMove = this.posHistory.filter(h => h.action === 'forward' || h.action === 'jump').length >= 8;

      if (wasTryingToMove && distMoved < 0.25) {
        this.stuckCount++;
        if (this.stuckCount >= 2) {
          this.isStuck = true;
          this.status = 'STUCK';

          if (f.wall && f.wallHead) {
            this.stopReason = `⛔ Blocked by 2-block high wall (${f.aheadBlock})`;
          } else if (f.liquid) {
            this.stopReason = `⛔ Trapped or halted by liquid (${f.aheadBlock})`;
          } else if (f.drop) {
            this.stopReason = '⛔ Stopped: Cliff or drop hazard ahead';
          } else {
            this.stopReason = `⛔ Stuck against obstacle at (${curPos.x.toFixed(1)}, ${curPos.y.toFixed(1)}, ${curPos.z.toFixed(1)})`;
          }
          return;
        }
      } else {
        this.stuckCount = 0;
        this.isStuck = false;
      }
    }

    // Normal active or waiting status
    if (!f.hasTarget) {
      this.status = 'SEARCHING';
      this.stopReason = '🔍 Searching for targets';
    } else {
      this.status = 'ACTIVE';
      this.stopReason = null;
    }
  }

  // Multi-phase evasive un-stuck maneuver (debounced with 3s cooldown)
  tryUnstuck(f = {}) {
    if (this.isUnstucking || !this.bot.entity) return;
    const now = Date.now();
    if (now - this.lastUnstuckTime < 3000) return; // Prevent spam: at least 3 seconds between attempts

    this.isUnstucking = true;
    this.lastUnstuckTime = now;
    this.status = 'STUCK';
    this.log(`[SURVIVAL] Bot is stuck (${this.stopReason || 'obstacle'}). Executing evasive un-stuck maneuver...`);

    const bot = this.bot;
    bot.clearControlStates();

    // Turn ~115 degrees to the left or right to escape the obstacle
    const currentYaw = bot.entity.yaw;
    const side = Math.random() < 0.5 ? 1 : -1;
    const escapeYaw = currentYaw + side * (Math.PI * 0.65);

    // Phase 1: Step back for 350ms to create clearance from the wall
    bot.setControlState('back', true);
    bot.setControlState('forward', false);

    setTimeout(() => {
      if (!bot.entity) { this.isUnstucking = false; return; }
      bot.setControlState('back', false);

      // Phase 2: Look at escape angle, sprint forward and jump over hurdles
      bot.look(escapeYaw, 0, true);
      bot.setControlState('forward', true);
      bot.setControlState('jump', true);

      setTimeout(() => {
        if (!bot.entity) { this.isUnstucking = false; return; }
        bot.setControlState('jump', false);

        // Phase 3: Continue advancing on new heading for 600ms
        setTimeout(() => {
          bot.clearControlStates();
          this.posHistory = [];
          this.stuckCount = 0;
          this.isStuck = false;
          this.isUnstucking = false;
          this.status = 'ACTIVE';
          this.log('[SURVIVAL] Un-stuck maneuver finished. Resuming normal navigation.');
        }, 600);
      }, 500);
    }, 350);
  }

  // Autonomously scan inventory, equip best food, and consume
  async checkAndEat() {
    if (this.isEating || !this.bot.entity) return false;
    const hp = this.bot.health;
    const food = this.bot.food;

    // Eat if hungry, or if damaged and food is below 20
    const needsFood = food <= 16 || (hp < 18 && food < 20);
    if (!needsFood) return false;

    const items = this.bot.inventory.items();
    let bestFood = null;
    for (const name of FOOD_PRIORITY) {
      const found = items.find(i => i.name === name);
      if (found) {
        bestFood = found;
        break;
      }
    }

    if (!bestFood) return false;

    try {
      this.isEating = true;
      this.status = 'EATING';
      this.stopReason = `🍗 Consuming ${bestFood.name} (HP: ${hp}/20, Food: ${food}/20)`;
      this.log(`[SURVIVAL] Low hunger/health (${hp} HP, ${food} Food). Eating ${bestFood.name}...`);

      // Cease movement while eating
      this.bot.clearControlStates();
      await this.bot.equip(bestFood, 'hand');
      await this.bot.consume();
      this.lastAteTime = Date.now();
      this.log(`[SURVIVAL] Finished eating ${bestFood.name}. Health: ${this.bot.health}, Food: ${this.bot.food}`);
    } catch (err) {
      this.log(`[SURVIVAL] Failed to eat: ${err.message}`);
    } finally {
      this.isEating = false;
      this.autoEquipGear();
    }
    return true;
  }

  // Auto-equip best weapon, shield, and armor from inventory
  async autoEquipGear() {
    if (this.isEating || !this.bot.entity) return;

    try {
      const items = this.bot.inventory.items();

      // 1. Equip best weapon to main hand
      for (const w of WEAPON_PRIORITY) {
        const found = items.find(i => i.name === w);
        if (found) {
          if (this.bot.heldItem?.name !== found.name) {
            await this.bot.equip(found, 'hand');
          }
          break;
        }
      }

      // 2. Equip shield to off-hand
      const shield = items.find(i => i.name === 'shield');
      if (shield) {
        await this.bot.equip(shield, 'off-hand').catch(() => {});
      }

      // 3. Equip armor
      for (const [slot, priorityList] of Object.entries(ARMOR_SLOTS)) {
        for (const armorName of priorityList) {
          const piece = items.find(i => i.name === armorName);
          if (piece) {
            await this.bot.equip(piece, slot).catch(() => {});
            break;
          }
        }
      }
    } catch {
      // Ignore equip errors during active combat transitions
    }
  }

  // Clutch reflex: negate fall damage using water bucket
  async checkWaterClutch() {
    if (!this.bot.entity) return;
    const vy = this.bot.entity.velocity.y;
    if (vy < -0.65) {
      const waterBucket = this.bot.inventory.items().find(i => i.name === 'water_bucket');
      if (waterBucket) {
        this.log('[SURVIVAL] Fast fall detected! Attempting MLG water bucket clutch...');
        try {
          await this.bot.look(this.bot.entity.yaw, -Math.PI / 2, true);
          await this.bot.equip(waterBucket, 'hand');
          const refBlock = this.bot.blockAt(this.bot.entity.position.offset(0, -1, 0));
          if (refBlock) {
            await this.bot.activateBlock(refBlock);
            setTimeout(async () => {
              const emptyBucket = this.bot.inventory.items().find(i => i.name === 'bucket');
              if (emptyBucket) {
                await this.bot.equip(emptyBucket, 'hand');
                const waterBlock = this.bot.blockAt(this.bot.entity.position);
                if (waterBlock && waterBlock.name.includes('water')) {
                  await this.bot.activateBlock(waterBlock).catch(() => {});
                }
              }
            }, 600);
          }
        } catch (e) {
          this.log(`[SURVIVAL] Water clutch error: ${e.message}`);
        }
      }
    }
  }
}

module.exports = {
  SurvivalManager,
  FOOD_PRIORITY,
  WEAPON_PRIORITY
};
