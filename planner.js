'use strict';
// planner.js - Survival Goal Planner & Tech Tree Knowledge Engine for Minecraft Bot.
// Encodes the complete early-game survival progression:
// Trees -> Logs -> Planks -> Crafting Table -> Sticks -> Wooden Sword/Pickaxe -> Stone -> Stone Gear

class SurvivalPlanner {
  constructor(bot, onLog = console.log) {
    this.bot = bot;
    this.log = msg => onLog(`[PLANNER] ${msg}`);
    this.currentGoal = null;
    this.craftingInProgress = false;
    this.lastCraftAttempt = 0;
  }

  // Count items matching a regex or string in inventory
  countItem(pattern) {
    if (!this.bot || !this.bot.inventory) return 0;
    const re = typeof pattern === 'string' ? new RegExp(pattern, 'i') : pattern;
    return this.bot.inventory.items()
      .filter(it => re.test(it.name))
      .reduce((s, it) => s + it.count, 0);
  }

  // Check if inventory has any item matching pattern
  hasItem(pattern) {
    return this.countItem(pattern) > 0;
  }

  // Inspect the tech tree milestones
  getMilestones() {
    const logs = this.countItem(/log|wood|stem/);
    const planks = this.countItem(/planks?/);
    const sticks = this.countItem(/stick/);
    const hasTable = this.hasItem(/crafting_table/);
    const hasWoodPickaxe = this.hasItem(/wooden_pickaxe/);
    const hasStonePickaxe = this.hasItem(/stone_pickaxe|iron_pickaxe|diamond_pickaxe/);
    const hasWoodSword = this.hasItem(/wooden_sword/);
    const hasStoneSword = this.hasItem(/stone_sword|iron_sword|diamond_sword/);
    const stone = this.countItem(/cobblestone|blackstone|cobbled_deepslate/);

    return [
      {
        id: 'gather_wood',
        name: 'Gather Wood Logs',
        textureUrl: '/textures/blocks/oak_log.png',
        target: 4,
        current: logs,
        done: logs >= 4 || planks >= 8
      },
      {
        id: 'craft_planks',
        name: 'Craft Wooden Planks',
        textureUrl: '/textures/blocks/oak_planks.png',
        target: 8,
        current: planks,
        done: planks >= 8 || hasTable
      },
      {
        id: 'craft_table',
        name: 'Craft Crafting Table',
        textureUrl: '/textures/blocks/crafting_table_front.png',
        target: 1,
        current: hasTable ? 1 : 0,
        done: hasTable || hasWoodPickaxe || hasWoodSword
      },
      {
        id: 'craft_sticks',
        name: 'Craft Sticks',
        textureUrl: '/textures/items/stick.png',
        target: 4,
        current: sticks,
        done: sticks >= 4 || hasWoodPickaxe || hasWoodSword
      },
      {
        id: 'craft_wooden_sword',
        name: 'Craft Wooden Sword',
        textureUrl: '/textures/items/wooden_sword.png',
        target: 1,
        current: (hasWoodSword || hasStoneSword) ? 1 : 0,
        done: hasWoodSword || hasStoneSword
      },
      {
        id: 'craft_wooden_pickaxe',
        name: 'Craft Wooden Pickaxe',
        textureUrl: '/textures/items/wooden_pickaxe.png',
        target: 1,
        current: (hasWoodPickaxe || hasStonePickaxe) ? 1 : 0,
        done: hasWoodPickaxe || hasStonePickaxe
      },
      {
        id: 'mine_cobblestone',
        name: 'Mine Cobblestone',
        textureUrl: '/textures/blocks/cobblestone.png',
        target: 8,
        current: stone,
        done: stone >= 8 || hasStonePickaxe || hasStoneSword
      },
      {
        id: 'craft_stone_gear',
        name: 'Upgrade to Stone Weapons & Tools',
        textureUrl: '/textures/items/stone_sword.png',
        target: 1,
        current: (hasStoneSword && hasStonePickaxe) ? 1 : 0,
        done: hasStoneSword && hasStonePickaxe
      }
    ];
  }

  // Assess what goal takes highest precedence right now
  getUrgentGoal(f = {}) {
    if (!this.bot || !this.bot.entity) return null;

    const hp = this.bot.health ?? 20;
    const food = this.bot.food ?? 20;
    const logs = this.countItem(/log|wood|stem/);
    const planks = this.countItem(/planks?/);
    const sticks = this.countItem(/stick/);
    const hasTable = this.hasItem(/crafting_table/);
    const hasWoodPickaxe = this.hasItem(/wooden_pickaxe/);
    const hasStonePickaxe = this.hasItem(/stone_pickaxe|iron_pickaxe|diamond_pickaxe/);
    const hasSword = this.hasItem(/sword/);
    const hasPickaxe = hasWoodPickaxe || hasStonePickaxe;
    const stone = this.countItem(/cobblestone|blackstone/);
    const isNight = this.bot.time ? this.bot.time.timeOfDay > 13000 : false;

    // Priority 1: Critical Survival - Low Health Flee or Low Hunger Eat
    if (hp <= 6 && f.mobNearby > 0) {
      return {
        id: 'flee',
        name: '🏃 Flee to Safety',
        action: 'sprint',
        description: 'Health critical! Evading hostile mobs',
        priority: 100
      };
    }

    if (food <= 12 && f.invFood > 0) {
      return {
        id: 'eat',
        name: '🍖 Eat Food',
        action: 'use_item',
        description: 'Hunger below 12. Restoring nutrition',
        priority: 90
      };
    }

    // Priority 2: Armed Self-Defense against Close Hostile Mobs
    if (f.mobNearby > 0 && f.inRange) {
      return {
        id: 'combat',
        name: '⚔️ Engage Hostile Mob',
        action: 'attack',
        description: `Defending against ${f.targetName || 'hostile mob'} with sword`,
        priority: 85
      };
    }

    // Priority 3: Early-Game Tech Tree Progression
    // Step A: Need wood logs
    if (logs < 4 && planks < 8 && !hasTable) {
      return {
        id: 'get_wood',
        name: '🪓 Chop Wood Logs',
        action: 'mine',
        targetBlock: 'log',
        description: `Chop trees (${logs}/4 logs collected)`,
        priority: 70
      };
    }

    // Step B: Craft planks from logs
    if (logs > 0 && planks < 8) {
      return {
        id: 'craft_planks',
        name: '🪵 Craft Wooden Planks',
        action: 'craft',
        recipe: 'planks',
        description: 'Converting raw logs into 2x2 planks',
        priority: 65
      };
    }

    // Step C: Craft Crafting Table if not present
    if (!hasTable && planks >= 4) {
      return {
        id: 'craft_table',
        name: '🔨 Craft Crafting Table',
        action: 'craft',
        recipe: 'crafting_table',
        description: 'Crafting table required for 3x3 weapon/tool crafting',
        priority: 60
      };
    }

    // Step D: Craft sticks
    if (sticks < 4 && planks >= 2) {
      return {
        id: 'craft_sticks',
        name: '🥢 Craft Sticks',
        action: 'craft',
        recipe: 'stick',
        description: 'Sticks required for tool handles',
        priority: 55
      };
    }

    // Step E: Craft Wooden Sword for self-defense
    if (!hasSword && planks >= 2 && sticks >= 1) {
      return {
        id: 'craft_sword',
        name: '🗡️ Craft Wooden Sword',
        action: 'craft',
        recipe: 'wooden_sword',
        description: 'Equip sword for combat with zombies & skeletons',
        priority: 50
      };
    }

    // Step F: Craft Wooden Pickaxe for mining stone
    if (!hasPickaxe && planks >= 3 && sticks >= 2) {
      return {
        id: 'craft_pickaxe',
        name: '⛏️ Craft Wooden Pickaxe',
        action: 'craft',
        recipe: 'wooden_pickaxe',
        description: 'Pickaxe required to mine cobblestone & ores',
        priority: 45
      };
    }

    // Step G: Mine cobblestone
    if (hasPickaxe && stone < 8 && !hasStonePickaxe) {
      return {
        id: 'mine_stone',
        name: '🪨 Mine Cobblestone',
        action: 'mine',
        targetBlock: 'stone',
        description: `Mining cobblestone (${stone}/8 stone collected)`,
        priority: 40
      };
    }

    // Step H: Nighttime caution
    if (isNight && !hasSword) {
      return {
        id: 'shelter',
        name: '⛺ Take Shelter / Bunker',
        action: 'sneak',
        description: 'Nightfall without weapon. Maintaining low profile',
        priority: 35
      };
    }

    return {
      id: 'explore_survive',
      name: '🧭 Explore & Survive',
      action: 'forward',
      description: 'Roaming world, scouting resources and maintaining vigilance',
      priority: 10
    };
  }

  // Find the nearest block matching a name/regex
  findNearestBlock(pattern, maxDist = 24) {
    if (!this.bot || !this.bot.findBlock) return null;
    const re = typeof pattern === 'string' ? new RegExp(pattern, 'i') : pattern;

    return this.bot.findBlock({
      matching: b => b && re.test(b.name),
      maxDistance: maxDist
    });
  }

  // Attempt to craft a recipe
  async tryCraft(recipeName) {
    if (this.craftingInProgress || Date.now() - this.lastCraftAttempt < 2000) return false;
    this.craftingInProgress = true;
    this.lastCraftAttempt = Date.now();

    try {
      let mcData;
      try {
        mcData = require('minecraft-data')(this.bot.version);
      } catch {
        mcData = require('minecraft-data')('1.20.1');
      }

      if (!mcData || !mcData.itemsByName) {
        this.craftingInProgress = false;
        return false;
      }

      // Find recipe ID
      const targetItem = mcData.itemsByName[recipeName];
      if (!targetItem) {
        this.log(`Unknown craft target item: ${recipeName}`);
        this.craftingInProgress = false;
        return false;
      }

      const recipes = this.bot.recipesFor(targetItem.id, null, 1, true);
      if (!recipes || recipes.length === 0) {
        // Check if a crafting table nearby makes recipe possible
        const tableBlock = this.findNearestBlock(/crafting_table/, 4);
        const tableRecipes = tableBlock ? this.bot.recipesFor(targetItem.id, null, 1, tableBlock) : [];
        if (tableRecipes.length > 0) {
          this.log(`Crafting ${recipeName} using Crafting Table at ${tableBlock.position}...`);
          await this.bot.craft(tableRecipes[0], 1, tableBlock);
          this.log(`✨ Successfully crafted ${recipeName}!`);
          this.craftingInProgress = false;
          return true;
        }
        this.craftingInProgress = false;
        return false;
      }

      // Craft in 2x2 player inventory
      this.log(`Crafting ${recipeName} in 2x2 player inventory...`);
      await this.bot.craft(recipes[0], 1, null);
      this.log(`✨ Successfully crafted ${recipeName}!`);
      this.craftingInProgress = false;
      return true;
    } catch (err) {
      this.log(`Crafting ${recipeName} notice: ${err.message}`);
      this.craftingInProgress = false;
      return false;
    }
  }

  // Full state report for web dashboard
  getStatus(f = {}) {
    const milestones = this.getMilestones();
    const urgent = this.getUrgentGoal(f);
    const completedCount = milestones.filter(m => m.done).length;

    return {
      currentGoal: urgent,
      milestones,
      progressPercent: Math.round((completedCount / milestones.length) * 100),
      inventory: {
        wood: this.countItem(/log|wood|stem/),
        planks: this.countItem(/planks?/),
        sticks: this.countItem(/stick/),
        food: this.countItem(/bread|apple|beef|porkchop|chicken|mutton|cookie|carrot|potato|melon|stew/),
        hasSword: this.hasItem(/sword/),
        hasPickaxe: this.hasItem(/pickaxe/),
        hasCraftingTable: this.hasItem(/crafting_table/)
      }
    };
  }
}

module.exports = { SurvivalPlanner };
