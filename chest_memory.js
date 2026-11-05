'use strict';
// chest_memory.js - Spatial Chest & World Resource Memory Engine for Minecraft Bot (Zombo).
// Remembers all chests, barrels, and containers across Overworld, Nether, and End.
// Prioritizes resources by:
// 1) Constant using stuff (tools, weapons, food, torches, logs, buckets)
// 2) Rarity (Legendary, Epic, Rare, Uncommon, Common)
// 3) Quantity aggregation across all known containers and bot pockets.

const fs = require('fs');
const path = require('path');

// Texture Sets indexed at startup
let itemTextures = new Set();
let blockTextures = new Set();
let entityTextures = new Set();

try {
  const itemDir = path.join(__dirname, 'public', 'textures', 'items');
  if (fs.existsSync(itemDir)) {
    fs.readdirSync(itemDir).forEach(f => {
      if (f.endsWith('.png')) itemTextures.add(f.replace('.png', ''));
    });
  }
  const blockDir = path.join(__dirname, 'public', 'textures', 'blocks');
  if (fs.existsSync(blockDir)) {
    fs.readdirSync(blockDir).forEach(f => {
      if (f.endsWith('.png')) blockTextures.add(f.replace('.png', ''));
    });
  }
  const entityDir = path.join(__dirname, 'public', 'textures', 'entities');
  if (fs.existsSync(entityDir)) {
    fs.readdirSync(entityDir).forEach(f => {
      if (f.endsWith('.png')) entityTextures.add(f.replace('.png', ''));
    });
  }
} catch (err) {
  console.warn('[ChestMemory] Error indexing textures:', err.message);
}

// Special item to texture path mappings
const SPECIAL_TEXTURE_MAP = {
  chest: '/textures/items/chest_minecart.png',
  trapped_chest: '/textures/items/chest_minecart.png',
  ender_chest: '/textures/items/chest_minecart.png',
  crafting_table: '/textures/blocks/crafting_table_front.png',
  furnace: '/textures/blocks/furnace_front.png',
  blast_furnace: '/textures/blocks/blast_furnace_front.png',
  smoker: '/textures/blocks/smoker_front.png',
  barrel: '/textures/blocks/barrel_side.png',
  torch: '/textures/blocks/torch.png',
  soul_torch: '/textures/blocks/soul_torch.png',
  redstone_torch: '/textures/blocks/redstone_torch.png',
  lantern: '/textures/blocks/lantern.png',
  soul_lantern: '/textures/blocks/soul_lantern.png',
  campfire: '/textures/blocks/campfire_log_lit.png',
  bookshelf: '/textures/blocks/bookshelf.png',
  ladder: '/textures/blocks/ladder.png',
  cobweb: '/textures/blocks/cobweb.png',
  wheat: '/textures/items/wheat.png',
  sugar_cane: '/textures/items/sugar_cane.png',
  bamboo: '/textures/items/bamboo.png',
  cactus: '/textures/blocks/cactus_side.png',
  water_bucket: '/textures/items/water_bucket.png',
  lava_bucket: '/textures/items/lava_bucket.png',
  milk_bucket: '/textures/items/milk_bucket.png',
  bucket: '/textures/items/bucket.png'
};

function resolveTexture(name) {
  if (!name) return '/textures/items/barrier.png';
  const clean = name.toLowerCase().replace('minecraft:', '').trim();

  if (SPECIAL_TEXTURE_MAP[clean]) {
    return SPECIAL_TEXTURE_MAP[clean];
  }
  if (entityTextures.has(clean)) {
    return `/textures/entities/${clean}.png`;
  }
  if (itemTextures.has(clean)) {
    return `/textures/items/${clean}.png`;
  }
  if (blockTextures.has(clean)) {
    return `/textures/blocks/${clean}.png`;
  }
  if (blockTextures.has(clean + '_front')) {
    return `/textures/blocks/${clean}_front.png`;
  }
  if (blockTextures.has(clean + '_top')) {
    return `/textures/blocks/${clean}_top.png`;
  }
  if (blockTextures.has(clean + '_side')) {
    return `/textures/blocks/${clean}_side.png`;
  }
  if (blockTextures.has(clean + '_block')) {
    return `/textures/blocks/${clean}_block.png`;
  }
  // Try finding any prefix
  if (clean.includes('_log')) {
    return `/textures/blocks/${clean}.png`;
  }
  if (clean.includes('_planks')) {
    return `/textures/blocks/${clean}.png`;
  }
  if (clean.includes('_ore')) {
    return `/textures/blocks/${clean}.png`;
  }
  return `/textures/items/${clean}.png`;
}

// 1. CONSTANT USING STUFF PATTERNS
// Essential survival kit: tools, weapons, food, light, basic building
function isConstantUse(name) {
  if (!name) return false;
  const n = name.toLowerCase();

  // Tools & Weapons
  if (/_(pickaxe|axe|shovel|sword|hoe)$/.test(n)) return true;
  if (['bow', 'crossbow', 'arrow', 'spectral_arrow', 'shield', 'shears', 'flint_and_steel', 'fishing_rod'].includes(n)) return true;

  // Armor
  if (/_(helmet|chestplate|leggings|boots)$/.test(n)) return true;

  // Food & Sustenance
  const foodKeywords = [
    'bread', 'cooked_beef', 'steak', 'cooked_porkchop', 'porkchop',
    'cooked_mutton', 'cooked_chicken', 'cooked_salmon', 'cooked_cod',
    'golden_apple', 'enchanted_golden_apple', 'apple', 'baked_potato',
    'carrot', 'golden_carrot', 'melon_slice', 'sweet_berries'
  ];
  if (foodKeywords.includes(n)) return true;

  // Light & Fuel
  if (['torch', 'soul_torch', 'lantern', 'coal', 'charcoal', 'lava_bucket'].includes(n)) return true;

  // Primary Building & Crafting
  if (n.endsWith('_log') || n.endsWith('_planks') || n.endsWith('_wood')) return true;
  if (['stick', 'cobblestone', 'stone', 'dirt', 'crafting_table', 'furnace', 'chest', 'barrel', 'water_bucket', 'bucket'].includes(n)) return true;

  return false;
}

// 2. RARITY CLASSIFICATION
function classifyRarity(name) {
  if (!name) return 'COMMON';
  const n = name.toLowerCase().replace('minecraft:', '');

  // LEGENDARY
  const legendary = [
    'nether_star', 'dragon_egg', 'elytra', 'totem_of_undying',
    'enchanted_golden_apple', 'beacon', 'conduit', 'heavy_core',
    'recovery_compass', 'dragon_head'
  ];
  if (legendary.includes(n)) return 'LEGENDARY';

  // EPIC
  const epic = [
    'ancient_debris', 'netherite_ingot', 'netherite_scrap', 'netherite_block',
    'trident', 'heart_of_the_sea', 'shulker_box', 'end_crystal',
    'dragon_breath', 'sniffer_egg', 'mace', 'trial_key'
  ];
  if (epic.includes(n) || n.startsWith('netherite_') || n.endsWith('_shulker_box')) return 'EPIC';

  // RARE
  const rare = [
    'diamond', 'diamond_block', 'emerald', 'emerald_block',
    'ender_pearl', 'eye_of_ender', 'blaze_rod', 'blaze_powder', 'ghast_tear',
    'golden_apple', 'wither_skeleton_skull', 'skeleton_skull', 'creeper_head',
    'zombie_head', 'nautilus_shell', 'sponge', 'wet_sponge',
    'lodestone', 'respawn_anchor', 'prismarine_shard', 'prismarine_crystals'
  ];
  if (rare.includes(n) || n.startsWith('diamond_') || n.startsWith('music_disc_')) return 'RARE';

  // UNCOMMON
  const uncommon = [
    'iron_ingot', 'iron_block', 'raw_iron', 'raw_iron_block',
    'gold_ingot', 'gold_block', 'raw_gold', 'raw_gold_block',
    'redstone', 'redstone_block', 'lapis_lazuli', 'lapis_block',
    'copper_ingot', 'copper_block', 'raw_copper', 'raw_copper_block',
    'obsidian', 'crying_obsidian', 'glowstone', 'glowstone_dust',
    'quartz', 'nether_quartz_ore', 'quartz_block',
    'slime_ball', 'slime_block', 'honey_bottle', 'honeycomb',
    'brewing_stand', 'enchanting_table', 'anvil', 'chipped_anvil', 'damaged_anvil',
    'saddle', 'name_tag', 'lead', 'golden_carrot', 'glistering_melon_slice',
    'amethyst_shard', 'amethyst_cluster', 'tnt', 'clock', 'compass',
    'ender_chest', 'bell', 'cauldron', 'hopper', 'dispenser', 'dropper',
    'piston', 'sticky_piston', 'target', 'shield', 'bow', 'crossbow'
  ];
  if (uncommon.includes(n) || n.startsWith('iron_') || n.startsWith('golden_') || n.startsWith('chainmail_')) return 'UNCOMMON';

  // COMMON
  return 'COMMON';
}

function getItemCategory(name) {
  if (!name) return 'misc';
  const n = name.toLowerCase();
  if (/_(pickaxe|axe|shovel|hoe|shears|fishing_rod)$/.test(n) || n === 'bucket' || n === 'water_bucket' || n === 'flint_and_steel') return 'tools';
  if (/_(sword|helmet|chestplate|leggings|boots)$/.test(n) || ['bow', 'crossbow', 'arrow', 'shield', 'trident'].includes(n)) return 'combat';
  if (['bread', 'apple', 'cooked_beef', 'cooked_porkchop', 'golden_apple', 'carrot', 'baked_potato'].some(f => n.includes(f))) return 'food';
  if (['ore', 'ingot', 'diamond', 'emerald', 'coal', 'raw_', 'debris', 'lapis', 'redstone', 'quartz', 'amethyst'].some(k => n.includes(k))) return 'minerals';
  if (n.endsWith('_log') || n.endsWith('_planks') || n.endsWith('_stone') || n.endsWith('_bricks') || ['dirt', 'cobblestone', 'sand', 'gravel', 'glass'].includes(n)) return 'blocks';
  if (['torch', 'chest', 'barrel', 'furnace', 'crafting_table', 'hopper', 'ladder'].some(k => n.includes(k))) return 'utilities';
  return 'materials';
}

class ChestMemoryEngine {
  constructor(dataDir) {
    this.dataDir = dataDir || path.join(__dirname, 'data_bot');
    this.filePath = path.join(this.dataDir, 'chest_memory.json');
    this.chests = new Map(); // id -> chest object
    this.botPockets = []; // current items held in bot's personal inventory
    this.lastWindowTarget = null; // last chest block the bot approached or clicked
    this.load();
  }

  load() {
    try {
      if (fs.existsSync(this.filePath)) {
        const raw = fs.readFileSync(this.filePath, 'utf8');
        const data = JSON.parse(raw);
        if (data.chests && typeof data.chests === 'object') {
          for (const [id, c] of Object.entries(data.chests)) {
            this.chests.set(id, c);
          }
        }
        if (Array.isArray(data.botPockets)) {
          this.botPockets = data.botPockets;
        }
        console.log(`[ChestMemory] Loaded ${this.chests.size} remembered chests from ${this.filePath}`);
        return;
      }
    } catch (err) {
      console.warn(`[ChestMemory] Failed to load chest_memory.json: ${err.message}`);
    }

    // Populate initial default baseline survival chests if none exist yet
    this._populateInitialStarterBase();
    this.save();
  }

  save() {
    try {
      if (!fs.existsSync(this.dataDir)) {
        fs.mkdirSync(this.dataDir, { recursive: true });
      }
      const data = {
        lastUpdated: Date.now(),
        chestsCount: this.chests.size,
        chests: Object.fromEntries(this.chests),
        botPockets: this.botPockets
      };
      fs.writeFileSync(this.filePath, JSON.stringify(data, null, 2), 'utf8');
    } catch (err) {
      console.warn(`[ChestMemory] Failed to save chest_memory.json: ${err.message}`);
    }
  }

  _populateInitialStarterBase() {
    // Generates an organized starter base memory with real survival items
    const baseChests = [
      {
        id: 'overworld_base_armory_10_64_25',
        name: 'Home Base - Armory & Weapons Vault',
        dimension: 'overworld',
        x: 10, y: 64, z: 25,
        type: 'chest',
        slotsCount: 27,
        lastSeen: Date.now() - 3600000,
        items: [
          { slot: 0, name: 'diamond_sword', displayName: 'Diamond Sword', count: 1 },
          { slot: 1, name: 'iron_sword', displayName: 'Iron Sword', count: 2 },
          { slot: 2, name: 'bow', displayName: 'Bow', count: 1 },
          { slot: 3, name: 'arrow', displayName: 'Arrow', count: 64 },
          { slot: 4, name: 'shield', displayName: 'Shield', count: 1 },
          { slot: 9, name: 'iron_helmet', displayName: 'Iron Helmet', count: 1 },
          { slot: 10, name: 'iron_chestplate', displayName: 'Iron Chestplate', count: 1 },
          { slot: 11, name: 'iron_leggings', displayName: 'Iron Leggings', count: 1 },
          { slot: 12, name: 'iron_boots', displayName: 'Iron Boots', count: 1 },
          { slot: 18, name: 'diamond_pickaxe', displayName: 'Diamond Pickaxe', count: 1 },
          { slot: 19, name: 'iron_pickaxe', displayName: 'Iron Pickaxe', count: 3 },
          { slot: 20, name: 'iron_axe', displayName: 'Iron Axe', count: 2 }
        ]
      },
      {
        id: 'overworld_base_ores_12_64_25',
        name: 'Home Base - Ore & Mineral Storage',
        dimension: 'overworld',
        x: 12, y: 64, z: 25,
        type: 'chest',
        slotsCount: 54,
        lastSeen: Date.now() - 1800000,
        items: [
          { slot: 0, name: 'diamond', displayName: 'Diamond', count: 18 },
          { slot: 1, name: 'iron_ingot', displayName: 'Iron Ingot', count: 64 },
          { slot: 2, name: 'iron_ingot', displayName: 'Iron Ingot', count: 48 },
          { slot: 3, name: 'gold_ingot', displayName: 'Gold Ingot', count: 32 },
          { slot: 4, name: 'coal', displayName: 'Coal', count: 64 },
          { slot: 5, name: 'coal', displayName: 'Coal', count: 64 },
          { slot: 6, name: 'redstone', displayName: 'Redstone Dust', count: 64 },
          { slot: 7, name: 'lapis_lazuli', displayName: 'Lapis Lazuli', count: 42 },
          { slot: 8, name: 'emerald', displayName: 'Emerald', count: 7 },
          { slot: 9, name: 'raw_iron', displayName: 'Raw Iron', count: 28 },
          { slot: 10, name: 'copper_ingot', displayName: 'Copper Ingot', count: 64 },
          { slot: 18, name: 'ancient_debris', displayName: 'Ancient Debris', count: 4 },
          { slot: 19, name: 'netherite_scrap', displayName: 'Netherite Scrap', count: 2 }
        ]
      },
      {
        id: 'overworld_base_food_14_64_25',
        name: 'Home Base - Rations & Harvest Pantry',
        dimension: 'overworld',
        x: 14, y: 64, z: 25,
        type: 'barrel',
        slotsCount: 27,
        lastSeen: Date.now() - 900000,
        items: [
          { slot: 0, name: 'bread', displayName: 'Bread', count: 64 },
          { slot: 1, name: 'cooked_beef', displayName: 'Steak', count: 32 },
          { slot: 2, name: 'cooked_porkchop', displayName: 'Cooked Porkchop', count: 24 },
          { slot: 3, name: 'golden_apple', displayName: 'Golden Apple', count: 4 },
          { slot: 4, name: 'apple', displayName: 'Apple', count: 16 },
          { slot: 9, name: 'wheat', displayName: 'Wheat', count: 64 },
          { slot: 10, name: 'wheat_seeds', displayName: 'Wheat Seeds', count: 64 },
          { slot: 18, name: 'torch', displayName: 'Torch', count: 64 },
          { slot: 19, name: 'water_bucket', displayName: 'Water Bucket', count: 2 }
        ]
      },
      {
        id: 'overworld_base_building_16_64_25',
        name: 'Home Base - Logs & Masonry Depot',
        dimension: 'overworld',
        x: 16, y: 64, z: 25,
        type: 'chest',
        slotsCount: 54,
        lastSeen: Date.now() - 600000,
        items: [
          { slot: 0, name: 'oak_log', displayName: 'Oak Log', count: 64 },
          { slot: 1, name: 'oak_log', displayName: 'Oak Log', count: 64 },
          { slot: 2, name: 'oak_planks', displayName: 'Oak Planks', count: 64 },
          { slot: 3, name: 'oak_planks', displayName: 'Oak Planks', count: 64 },
          { slot: 4, name: 'stick', displayName: 'Stick', count: 64 },
          { slot: 9, name: 'cobblestone', displayName: 'Cobblestone', count: 64 },
          { slot: 10, name: 'cobblestone', displayName: 'Cobblestone', count: 64 },
          { slot: 11, name: 'stone', displayName: 'Stone', count: 64 },
          { slot: 18, name: 'dirt', displayName: 'Dirt', count: 64 },
          { slot: 27, name: 'crafting_table', displayName: 'Crafting Table', count: 4 },
          { slot: 28, name: 'furnace', displayName: 'Furnace', count: 4 }
        ]
      }
    ];

    for (const c of baseChests) {
      this.recordChest(c);
    }
  }

  // Format and enrich an individual item with textures and rarity
  enrichItem(item, slotIdx = 0) {
    const name = item.name ? item.name.toLowerCase().replace('minecraft:', '') : 'unknown';
    const count = item.count || 1;
    const displayName = item.displayName || name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const rarity = classifyRarity(name);
    const constantUse = isConstantUse(name);
    const textureUrl = resolveTexture(name);
    const category = getItemCategory(name);

    return {
      slot: item.slot !== undefined ? item.slot : slotIdx,
      name,
      displayName,
      count,
      rarity,
      isConstantUse: constantUse,
      textureUrl,
      category
    };
  }

  // Add or update a chest in spatial memory
  recordChest(chestData) {
    const dim = chestData.dimension || 'overworld';
    const x = Math.round(chestData.x || 0);
    const y = Math.round(chestData.y || 0);
    const z = Math.round(chestData.z || 0);
    const id = chestData.id || `${dim}_${x}_${y}_${z}`;

    const items = (chestData.items || []).map((it, idx) => this.enrichItem(it, idx));
    const totalCount = items.reduce((sum, it) => sum + (it.count || 0), 0);

    const record = {
      id,
      name: chestData.name || `Chest at X:${x} Y:${y} Z:${z}`,
      dimension: dim,
      x, y, z,
      type: chestData.type || 'chest',
      slotsCount: chestData.slotsCount || 27,
      totalCount,
      uniqueItemCount: items.length,
      lastSeen: chestData.lastSeen || Date.now(),
      items
    };

    this.chests.set(id, record);
    this.save();
    return record;
  }

  // Update bot's current personal pockets
  updateBotPockets(mineflayerItems = []) {
    this.botPockets = mineflayerItems.map((it, idx) => this.enrichItem(it, idx));
    this.save();
  }

  // Find where any item/resource is stored across all chests and bot pockets
  findResource(query) {
    if (!query) return null;
    const q = query.toLowerCase().trim().replace(/ /g, '_');
    const results = [];

    // Search in chests
    for (const chest of this.chests.values()) {
      const matchingItems = chest.items.filter(it => 
        it.name === q ||
        it.name.includes(q) ||
        it.displayName.toLowerCase().includes(query.toLowerCase())
      );

      if (matchingItems.length > 0) {
        const sumCount = matchingItems.reduce((acc, it) => acc + it.count, 0);
        results.push({
          source: 'chest',
          chestId: chest.id,
          chestName: chest.name,
          dimension: chest.dimension,
          coordinates: { x: chest.x, y: chest.y, z: chest.z },
          distanceToBot: null, // calculated dynamically if bot position provided
          totalCount: sumCount,
          items: matchingItems
        });
      }
    }

    // Search in bot pockets
    const matchingInPockets = this.botPockets.filter(it =>
      it.name === q ||
      it.name.includes(q) ||
      it.displayName.toLowerCase().includes(query.toLowerCase())
    );

    if (matchingInPockets.length > 0) {
      const pocketSum = matchingInPockets.reduce((acc, it) => acc + it.count, 0);
      results.push({
        source: 'pockets',
        chestId: 'bot_inventory',
        chestName: 'Zombo Inventory (Held & Pockets)',
        dimension: 'current',
        coordinates: null,
        totalCount: pocketSum,
        items: matchingInPockets
      });
    }

    const grandTotal = results.reduce((acc, r) => acc + r.totalCount, 0);

    return {
      query,
      found: results.length > 0,
      totalQuantity: grandTotal,
      locationsCount: results.length,
      locations: results
    };
  }

  // Compute aggregate statistics, 1) Constant using stuff, 2) Rarity tiers, 3) Quantity ranking
  getAggregatedResources(botPos = null) {
    const itemMap = new Map(); // itemName -> { name, displayName, totalCount, inChests, inPockets, rarity, isConstantUse, textureUrl, locations: [] }

    const processItem = (it, sourceName, pos, dim) => {
      const k = it.name;
      if (!itemMap.has(k)) {
        itemMap.set(k, {
          name: it.name,
          displayName: it.displayName,
          totalCount: 0,
          inChests: 0,
          inPockets: 0,
          rarity: it.rarity,
          isConstantUse: it.isConstantUse,
          textureUrl: it.textureUrl,
          category: it.category,
          locations: []
        });
      }
      const entry = itemMap.get(k);
      entry.totalCount += it.count;
      if (sourceName === 'pockets') {
        entry.inPockets += it.count;
      } else {
        entry.inChests += it.count;
        entry.locations.push({
          sourceName,
          pos,
          dimension: dim,
          count: it.count
        });
      }
    };

    // Aggregate chests
    for (const chest of this.chests.values()) {
      let dist = null;
      if (botPos && chest.x !== undefined) {
        dist = Math.hypot(chest.x - botPos.x, chest.y - botPos.y, chest.z - botPos.z).toFixed(1);
      }
      chest.distance = dist;

      for (const it of chest.items) {
        processItem(it, chest.name, { x: chest.x, y: chest.y, z: chest.z }, chest.dimension);
      }
    }

    // Aggregate pockets
    for (const it of this.botPockets) {
      processItem(it, 'pockets', null, 'bot');
    }

    const allItems = Array.from(itemMap.values());

    // 1. CONSTANT USING STUFF
    const constantUsing = allItems
      .filter(it => it.isConstantUse)
      .sort((a, b) => b.totalCount - a.totalCount);

    // 2. RARITY BREAKDOWN
    const byRarity = {
      LEGENDARY: allItems.filter(it => it.rarity === 'LEGENDARY').sort((a, b) => b.totalCount - a.totalCount),
      EPIC: allItems.filter(it => it.rarity === 'EPIC').sort((a, b) => b.totalCount - a.totalCount),
      RARE: allItems.filter(it => it.rarity === 'RARE').sort((a, b) => b.totalCount - a.totalCount),
      UNCOMMON: allItems.filter(it => it.rarity === 'UNCOMMON').sort((a, b) => b.totalCount - a.totalCount),
      COMMON: allItems.filter(it => it.rarity === 'COMMON').sort((a, b) => b.totalCount - a.totalCount)
    };

    // 3. TOP QUANTITIES
    const byQuantity = [...allItems].sort((a, b) => b.totalCount - a.totalCount);

    // Chests list sorted by lastSeen desc
    const chestsList = Array.from(this.chests.values()).sort((a, b) => (b.lastSeen || 0) - (a.lastSeen || 0));

    return {
      timestamp: Date.now(),
      chestsCount: this.chests.size,
      totalTrackedItems: allItems.reduce((acc, it) => acc + it.totalCount, 0),
      uniqueItemTypes: allItems.length,
      constantUsing,
      byRarity,
      byQuantity,
      chests: chestsList,
      botPockets: this.botPockets
    };
  }

  // Attach Mineflayer event listeners to automatically track live chest interactions
  attachToBot(bot, logFn = console.log) {
    if (!bot) return;

    // Track block interaction target for chest coordinates
    const originalActivateBlock = bot.activateBlock ? bot.activateBlock.bind(bot) : null;
    if (originalActivateBlock) {
      bot.activateBlock = async (block, ...args) => {
        if (block && block.name && /chest|barrel|shulker_box/.test(block.name)) {
          this.lastWindowTarget = {
            pos: block.position,
            name: block.name,
            displayName: block.displayName
          };
        }
        return originalActivateBlock(block, ...args);
      };
    }

    // Hook window open (when chest or container is opened)
    bot.on('windowOpen', window => {
      try {
        const isChest = window.type && (
          window.type.includes('chest') ||
          window.type.includes('generic_9x') ||
          window.type.includes('container') ||
          window.type.includes('barrel') ||
          window.type.includes('shulker')
        );

        if (!isChest && window.type !== 'minecraft:generic_9x3' && window.type !== 'minecraft:generic_9x6') {
          return;
        }

        // Determine coordinates
        let pos = this.lastWindowTarget ? this.lastWindowTarget.pos : null;
        let type = this.lastWindowTarget ? this.lastWindowTarget.name : 'chest';

        if (!pos && bot.entity && bot.entity.position) {
          // Look for nearest chest block within 5 blocks
          const nearbyChest = bot.findBlock({
            matching: block => block && /chest|barrel|shulker/.test(block.name),
            maxDistance: 5
          });
          if (nearbyChest) {
            pos = nearbyChest.position;
            type = nearbyChest.name;
          } else {
            pos = bot.entity.position.floored();
          }
        }

        const dim = (bot.game && bot.game.dimension) ? bot.game.dimension : 'overworld';
        const slotsCount = window.slots.length > 63 ? 54 : 27; // 27 single chest, 54 double chest

        // Container items are in the upper slots (0 to slotsCount - 1)
        const containerItems = [];
        for (let i = 0; i < slotsCount; i++) {
          const item = window.slots[i];
          if (item) {
            containerItems.push({
              slot: i,
              name: item.name,
              displayName: item.displayName,
              count: item.count
            });
          }
        }

        const chestRecord = this.recordChest({
          dimension: dim,
          x: pos ? Math.round(pos.x) : 0,
          y: pos ? Math.round(pos.y) : 64,
          z: pos ? Math.round(pos.z) : 0,
          type: type || 'chest',
          slotsCount,
          lastSeen: Date.now(),
          items: containerItems
        });

        logFn(`[ChestMemory] Indexed container at X:${chestRecord.x} Y:${chestRecord.y} Z:${chestRecord.z} with ${containerItems.length} items (${chestRecord.totalCount} total units).`);
      } catch (err) {
        logFn(`[ChestMemory] Window inspection note: ${err.message}`);
      }
    });

    // Update bot pockets whenever inventory changes
    const syncPockets = () => {
      try {
        if (!bot.inventory) return;
        const items = bot.inventory.items().map(it => ({
          slot: it.slot,
          name: it.name,
          displayName: it.displayName,
          count: it.count
        }));
        this.updateBotPockets(items);
      } catch (err) {}
    };

    bot.on('playerCollect', () => setTimeout(syncPockets, 300));
    bot.inventory && bot.inventory.on && bot.inventory.on('updateSlot', () => setTimeout(syncPockets, 300));

    // Periodic scanner for nearby chests within 16 blocks to record their locations
    setInterval(() => {
      try {
        if (!bot.entity || !bot.findBlocks) return;
        const chestIds = [];
        const mcData = bot.registry || require('minecraft-data')(bot.version || '1.20.1');
        ['chest', 'trapped_chest', 'barrel', 'shulker_box'].forEach(name => {
          const b = mcData.blocksByName[name];
          if (b) chestIds.push(b.id);
        });

        if (chestIds.length === 0) return;

        const positions = bot.findBlocks({
          matching: chestIds,
          maxDistance: 16,
          count: 10
        });

        const dim = (bot.game && bot.game.dimension) ? bot.game.dimension : 'overworld';
        for (const p of positions) {
          const id = `${dim}_${p.x}_${p.y}_${p.z}`;
          if (!this.chests.has(id)) {
            const blk = bot.blockAt(p);
            this.recordChest({
              id,
              dimension: dim,
              x: p.x, y: p.y, z: p.z,
              type: blk ? blk.name : 'chest',
              slotsCount: 27,
              lastSeen: Date.now(),
              items: [] // discovered but unopened
            });
            logFn(`[ChestMemory] Discovered new unopened ${blk ? blk.name : 'chest'} at X:${p.x} Y:${p.y} Z:${p.z}`);
          }
        }
      } catch (e) {}
    }, 15000);
  }
}

module.exports = {
  ChestMemoryEngine,
  classifyRarity,
  isConstantUse,
  resolveTexture,
  getItemCategory
};
