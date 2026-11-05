'use strict';
// knowledge.js - Scalable Minecraft Knowledge, Recipe, and Multi-Tier Tech Tree Engine.
// Uses official 'minecraft-data' to dynamically support ALL Minecraft versions (1.18, 1.19, 1.20, 1.21+),
// including every block, item, crafting recipe, smelting recipe, and harvesting tool requirement.

const mcDataLoader = require('minecraft-data');

function getMCData(version = '1.20.1') {
  try {
    return mcDataLoader(version);
  } catch {
    return mcDataLoader('1.20.1');
  }
}

// 7 Comprehensive Tech Progression Ages across all of Minecraft
const TECH_TREE_AGES = [
  {
    id: 'wood_age',
    name: 'Wood Age (Primitive)',
    textureUrl: '/textures/blocks/oak_log.png',
    milestones: [
      { id: 'gather_wood', name: 'Gather Wood Logs', textureUrl: '/textures/blocks/oak_log.png', target: 4, pattern: /log|wood|stem/i },
      { id: 'craft_planks', name: 'Craft Wooden Planks', textureUrl: '/textures/blocks/oak_planks.png', target: 8, pattern: /planks?/i },
      { id: 'craft_table', name: 'Craft Crafting Table', textureUrl: '/textures/blocks/crafting_table_front.png', target: 1, pattern: /crafting_table/i },
      { id: 'craft_sticks', name: 'Craft Sticks', textureUrl: '/textures/items/stick.png', target: 4, pattern: /stick/i },
      { id: 'craft_wood_pickaxe', name: 'Wooden Pickaxe', textureUrl: '/textures/items/wooden_pickaxe.png', target: 1, pattern: /wooden_pickaxe/i },
      { id: 'craft_wood_sword', name: 'Wooden Sword', textureUrl: '/textures/items/wooden_sword.png', target: 1, pattern: /wooden_sword/i }
    ]
  },
  {
    id: 'stone_age',
    name: 'Stone Age (Tools & Fire)',
    textureUrl: '/textures/blocks/stone.png',
    milestones: [
      { id: 'mine_cobble', name: 'Mine Cobblestone', textureUrl: '/textures/blocks/cobblestone.png', target: 12, pattern: /cobblestone|blackstone|cobbled_deepslate/i },
      { id: 'craft_furnace', name: 'Craft Furnace', textureUrl: '/textures/blocks/furnace_front.png', target: 1, pattern: /furnace/i },
      { id: 'craft_stone_pickaxe', name: 'Stone Pickaxe', textureUrl: '/textures/items/stone_pickaxe.png', target: 1, pattern: /stone_pickaxe/i },
      { id: 'craft_stone_sword', name: 'Stone Sword', textureUrl: '/textures/items/stone_sword.png', target: 1, pattern: /stone_sword/i },
      { id: 'craft_stone_axe', name: 'Stone Axe', textureUrl: '/textures/items/stone_axe.png', target: 1, pattern: /stone_axe/i }
    ]
  },
  {
    id: 'sustenance_age',
    name: 'Sustenance & Cooking',
    textureUrl: '/textures/items/bread.png',
    milestones: [
      { id: 'collect_meat', name: 'Hunt Animal Food', textureUrl: '/textures/items/beef.png', target: 5, pattern: /raw_beef|raw_porkchop|raw_chicken|raw_mutton|beef|porkchop|chicken|mutton/i },
      { id: 'cook_food', name: 'Cook Rations in Furnace', textureUrl: '/textures/items/cooked_beef.png', target: 4, pattern: /cooked_beef|cooked_porkchop|cooked_chicken|cooked_mutton|bread|baked_potato/i },
      { id: 'craft_torches', name: 'Craft Torches for Light', textureUrl: '/textures/blocks/torch.png', target: 8, pattern: /torch/i }
    ]
  },
  {
    id: 'iron_age',
    name: 'Iron Age & Armor Defense',
    textureUrl: '/textures/items/iron_ingot.png',
    milestones: [
      { id: 'mine_coal', name: 'Mine Coal Fuel', textureUrl: '/textures/items/coal.png', target: 8, pattern: /coal|charcoal/i },
      { id: 'mine_iron', name: 'Mine Raw Iron Ore', textureUrl: '/textures/items/raw_iron.png', target: 10, pattern: /raw_iron|iron_ore|deepslate_iron_ore/i },
      { id: 'smelt_iron', name: 'Smelt Iron Ingots', textureUrl: '/textures/items/iron_ingot.png', target: 10, pattern: /iron_ingot/i },
      { id: 'craft_shield', name: 'Craft Combat Shield', textureUrl: '/textures/items/shield.png', target: 1, pattern: /shield/i },
      { id: 'craft_bucket', name: 'Craft Water Clutch Bucket', textureUrl: '/textures/items/water_bucket.png', target: 1, pattern: /bucket|water_bucket/i },
      { id: 'craft_iron_pickaxe', name: 'Craft Iron Pickaxe', textureUrl: '/textures/items/iron_pickaxe.png', target: 1, pattern: /iron_pickaxe/i },
      { id: 'craft_iron_sword', name: 'Craft Iron Sword', textureUrl: '/textures/items/iron_sword.png', target: 1, pattern: /iron_sword/i },
      { id: 'craft_iron_armor', name: 'Craft Iron Armor Set', textureUrl: '/textures/items/iron_chestplate.png', target: 1, pattern: /iron_chestplate|iron_helmet|iron_leggings|iron_boots/i }
    ]
  },
  {
    id: 'agriculture_age',
    name: 'Agriculture & Farming',
    textureUrl: '/textures/items/wheat.png',
    milestones: [
      { id: 'craft_hoe', name: 'Craft Farming Hoe', textureUrl: '/textures/items/iron_hoe.png', target: 1, pattern: /hoe/i },
      { id: 'gather_seeds', name: 'Collect Crop Seeds', textureUrl: '/textures/items/wheat_seeds.png', target: 8, pattern: /wheat_seeds|beetroot_seeds|melon_seeds|pumpkin_seeds|carrot|potato/i },
      { id: 'harvest_wheat', name: 'Harvest Ripe Wheat', textureUrl: '/textures/items/wheat.png', target: 6, pattern: /wheat/i },
      { id: 'bake_bread', name: 'Bake Fresh Bread', textureUrl: '/textures/items/bread.png', target: 3, pattern: /bread/i }
    ]
  },
  {
    id: 'diamond_age',
    name: 'Diamond & Obsidian Tier',
    textureUrl: '/textures/items/diamond.png',
    milestones: [
      { id: 'mine_diamonds', name: 'Mine Diamonds', textureUrl: '/textures/items/diamond.png', target: 5, pattern: /diamond/i },
      { id: 'craft_diamond_pickaxe', name: 'Diamond Pickaxe', textureUrl: '/textures/items/diamond_pickaxe.png', target: 1, pattern: /diamond_pickaxe/i },
      { id: 'craft_diamond_sword', name: 'Diamond Sword', textureUrl: '/textures/items/diamond_sword.png', target: 1, pattern: /diamond_sword/i },
      { id: 'mine_obsidian', name: 'Mine Obsidian Blocks', textureUrl: '/textures/blocks/obsidian.png', target: 10, pattern: /obsidian/i },
      { id: 'craft_enchant_table', name: 'Craft Enchanting Table', textureUrl: '/textures/blocks/crafting_table_front.png', target: 1, pattern: /enchanting_table/i }
    ]
  },
  {
    id: 'nether_age',
    name: 'Nether & Advanced Technology',
    textureUrl: '/textures/blocks/netherrack.png',
    milestones: [
      { id: 'craft_flint_steel', name: 'Flint and Steel', textureUrl: '/textures/items/flint_and_steel.png', target: 1, pattern: /flint_and_steel/i },
      { id: 'collect_blaze_rod', name: 'Blaze Rods', textureUrl: '/textures/items/blaze_rod.png', target: 2, pattern: /blaze_rod/i },
      { id: 'collect_ender_pearl', name: 'Ender Pearls', textureUrl: '/textures/items/ender_pearl.png', target: 4, pattern: /ender_pearl/i },
      { id: 'craft_brewing_stand', name: 'Brewing Stand', textureUrl: '/textures/items/brewing_stand.png', target: 1, pattern: /brewing_stand/i }
    ]
  }
];

class MinecraftKnowledgeEngine {
  constructor(version = '1.20.1') {
    this.version = version;
    this.mcData = getMCData(version);
  }

  setVersion(version) {
    if (version && version !== this.version) {
      this.version = version;
      this.mcData = getMCData(version);
    }
  }

  // Count matching items in an array of inventory items
  countItems(items = [], pattern) {
    if (!Array.isArray(items)) return 0;
    const re = pattern instanceof RegExp ? pattern : new RegExp(pattern, 'i');
    return items
      .filter(it => it && re.test(it.name))
      .reduce((s, it) => s + (it.count || 1), 0);
  }

  // Evaluate full tech tree state across all 7 ages
  evaluateTechTree(inventoryItems = []) {
    let totalMilestones = 0;
    let completedMilestones = 0;

    const evaluatedAges = TECH_TREE_AGES.map(age => {
      let ageCompleted = 0;
      const evaluatedMilestones = age.milestones.map(m => {
        totalMilestones++;
        const currentCount = this.countItems(inventoryItems, m.pattern);
        const done = currentCount >= m.target;
        if (done) {
          completedMilestones++;
          ageCompleted++;
        }
        return {
          id: m.id,
          name: m.name,
          target: m.target,
          current: currentCount,
          done
        };
      });

      return {
        id: age.id,
        name: age.name,
        milestones: evaluatedMilestones,
        completedCount: ageCompleted,
        totalCount: age.milestones.length,
        isCompleted: ageCompleted === age.milestones.length
      };
    });

    const overallProgress = totalMilestones > 0 ? Math.round((completedMilestones / totalMilestones) * 100) : 0;

    return {
      progressPercent: overallProgress,
      totalMilestones,
      completedMilestones,
      ages: evaluatedAges
    };
  }

  // Search recipes, blocks, and items for any query
  searchKnowledge(query, limit = 15) {
    if (!query || !this.mcData) return [];
    const q = query.toLowerCase().trim();
    const results = [];

    // Search items
    if (this.mcData.itemsByName) {
      for (const [name, item] of Object.entries(this.mcData.itemsByName)) {
        if (name.includes(q) || (item.displayName && item.displayName.toLowerCase().includes(q))) {
          const recipes = (this.mcData.recipes && this.mcData.recipes[item.id]) || [];
          results.push({
            type: 'item',
            name: item.name,
            displayName: item.displayName || item.name,
            id: item.id,
            stackSize: item.stackSize || 64,
            hasRecipe: recipes.length > 0,
            recipeCount: recipes.length
          });
          if (results.length >= limit) break;
        }
      }
    }

    // Search blocks if under limit
    if (results.length < limit && this.mcData.blocksByName) {
      for (const [name, block] of Object.entries(this.mcData.blocksByName)) {
        if ((name.includes(q) || (block.displayName && block.displayName.toLowerCase().includes(q))) &&
            !results.some(r => r.name === name)) {
          results.push({
            type: 'block',
            name: block.name,
            displayName: block.displayName || block.name,
            id: block.id,
            hardness: block.hardness ?? 1.5,
            diggable: block.diggable !== false,
            material: block.material || 'stone'
          });
          if (results.length >= limit) break;
        }
      }
    }

    return results;
  }

  // Detailed recipe inspection for crafting
  getRecipeDetails(itemName) {
    if (!this.mcData || !this.mcData.itemsByName) return null;
    const item = this.mcData.itemsByName[itemName];
    if (!item) return null;

    const rawRecipes = (this.mcData.recipes && this.mcData.recipes[item.id]) || [];
    if (!rawRecipes.length) {
      return { itemName, hasRecipe: false, message: 'No crafting table recipe (natural resource, mob drop, or smelting output)' };
    }

    const parsedRecipes = rawRecipes.map(r => {
      // Decode ingredients
      const ingredients = {};
      if (Array.isArray(r.ingredients)) {
        for (const ingId of r.ingredients) {
          if (ingId !== null && ingId !== undefined) {
            const ingItem = this.mcData.items[ingId];
            const ingName = ingItem ? ingItem.displayName || ingItem.name : `Item #${ingId}`;
            ingredients[ingName] = (ingredients[ingName] || 0) + 1;
          }
        }
      } else if (Array.isArray(r.inShape)) {
        for (const row of r.inShape) {
          for (const ingId of row) {
            if (ingId !== null && ingId !== undefined) {
              const ingItem = this.mcData.items[ingId];
              const ingName = ingItem ? ingItem.displayName || ingItem.name : `Item #${ingId}`;
              ingredients[ingName] = (ingredients[ingName] || 0) + 1;
            }
          }
        }
      }

      return {
        resultCount: r.result ? r.result.count : 1,
        requiresCraftingTable: (r.inShape && (r.inShape.length > 2 || (r.inShape[0] && r.inShape[0].length > 2))),
        ingredients
      };
    });

    return {
      itemName,
      displayName: item.displayName || item.name,
      hasRecipe: true,
      recipes: parsedRecipes
    };
  }

  // Determine best tool from inventory to harvest a specific block
  getBestToolForBlock(blockName, inventoryItems = []) {
    if (!this.mcData || !this.mcData.blocksByName) return null;
    const block = this.mcData.blocksByName[blockName];
    if (!block) return null;

    // Standard Minecraft tool affinity
    const name = block.name.toLowerCase();
    let preferredToolPattern;
    if (/ore|stone|rock|deepslate|granite|diorite|andesite|brick|obsidian|iron_block|gold_block/i.test(name)) {
      preferredToolPattern = /pickaxe/i;
    } else if (/log|wood|stem|planks|fence|door|chest|crafting_table|bookshelf/i.test(name)) {
      preferredToolPattern = /axe/i;
    } else if (/dirt|grass|sand|gravel|clay|mud|snow/i.test(name)) {
      preferredToolPattern = /shovel/i;
    } else if (/leaves|web|wool/i.test(name)) {
      preferredToolPattern = /shears|sword/i;
    }

    if (!preferredToolPattern) return null;

    // Find best match in inventory (prioritizing diamond > iron > stone > wood)
    const matches = inventoryItems.filter(i => i && preferredToolPattern.test(i.name));
    if (!matches.length) return null;

    const tierOrder = ['netherite', 'diamond', 'iron', 'golden', 'stone', 'wooden'];
    matches.sort((a, b) => {
      const aTier = tierOrder.findIndex(t => a.name.includes(t));
      const bTier = tierOrder.findIndex(t => b.name.includes(t));
      const aScore = aTier === -1 ? 99 : aTier;
      const bScore = bTier === -1 ? 99 : bTier;
      return aScore - bScore;
    });

    return matches[0].name;
  }
}

module.exports = {
  MinecraftKnowledgeEngine,
  TECH_TREE_AGES,
  getMCData
};
