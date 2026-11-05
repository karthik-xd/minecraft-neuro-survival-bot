'use strict';
// prompt_commander.js - Natural Language AI Prompt Parser & Execution Engine for Minecraft Bot.
// Parses English prompts like:
// "hai Zombo can u mine diamond"
// "hai Zombo can u mine coal"
// "Zombo kill zombie" / "kill player Steve"
// "Zombo collect food for me"
// "Zombo build a farm" / "build shelter"
// "Zombo follow and protect me"
// "Zombo go to 100 64 -200"
// "Zombo craft crafting table"
// "Zombo stop"

// Canonical Minecraft block name aliases
const BLOCK_ALIASES = {
  diamond: 'diamond_ore',
  diamonds: 'diamond_ore',
  diamond_ore: 'diamond_ore',
  deepslate_diamond: 'deepslate_diamond_ore',
  coal: 'coal_ore',
  coals: 'coal_ore',
  coal_ore: 'coal_ore',
  iron: 'iron_ore',
  irons: 'iron_ore',
  iron_ore: 'iron_ore',
  raw_iron: 'iron_ore',
  gold: 'gold_ore',
  gold_ore: 'gold_ore',
  copper: 'copper_ore',
  copper_ore: 'copper_ore',
  redstone: 'redstone_ore',
  redstone_ore: 'redstone_ore',
  lapis: 'lapis_ore',
  lapis_ore: 'lapis_ore',
  emerald: 'emerald_ore',
  emerald_ore: 'emerald_ore',
  netherite: 'ancient_debris',
  debris: 'ancient_debris',
  ancient_debris: 'ancient_debris',
  wood: 'oak_log',
  log: 'oak_log',
  logs: 'oak_log',
  oak: 'oak_log',
  tree: 'oak_log',
  trees: 'oak_log',
  stone: 'stone',
  cobblestone: 'cobblestone',
  cobble: 'cobblestone',
  dirt: 'dirt',
  sand: 'sand',
  gravel: 'gravel',
  obsidian: 'obsidian',
  netherrack: 'netherrack',
  quartz: 'nether_quartz_ore',
  glowstone: 'glowstone',
  glass: 'glass',
  wool: 'white_wool'
};

// Mob aliases
const MOB_ALIASES = {
  zombie: 'zombie',
  zombies: 'zombie',
  skeleton: 'skeleton',
  skeletons: 'skeleton',
  creeper: 'creeper',
  creepers: 'creeper',
  spider: 'spider',
  spiders: 'spider',
  witch: 'witch',
  witches: 'witch',
  slime: 'slime',
  enderman: 'enderman',
  endermen: 'enderman',
  blaze: 'blaze',
  ghast: 'ghast',
  wither: 'wither',
  wither_skeleton: 'wither_skeleton',
  magma_cube: 'magma_cube',
  piglin: 'piglin',
  hoglin: 'hoglin',
  drowned: 'drowned',
  phantom: 'phantom',
  pillager: 'pillager',
  warden: 'warden',
  dragon: 'ender_dragon',
  ender_dragon: 'ender_dragon',
  cow: 'cow',
  cows: 'cow',
  pig: 'pig',
  pigs: 'pig',
  chicken: 'chicken',
  chickens: 'chicken',
  sheep: 'sheep'
};

// Number word to integer mapping
const NUMBER_WORDS = {
  one: 1, two: 2, three: 3, four: 4, five: 5,
  six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  twelve: 12, sixteen: 16, twenty: 20, thirty: 30,
  sixty: 64, stack: 64
};

class PromptCommander {
  constructor(defaultBotName = 'Zombo', chestMemory = null) {
    this.defaultBotName = defaultBotName;
    this.chestMemory = chestMemory;
  }

  setChestMemory(chestMemory) {
    this.chestMemory = chestMemory;
  }

  // Extract count from text (digits or number words)
  extractCount(text, defaultCount = 10) {
    const digitMatch = text.match(/\b(\d+)\b/);
    if (digitMatch) {
      return parseInt(digitMatch[1], 10);
    }
    for (const [word, val] of Object.entries(NUMBER_WORDS)) {
      const regex = new RegExp(`\\b${word}\\b`, 'i');
      if (regex.test(text)) return val;
    }
    return defaultCount;
  }

  // Parse natural language prompt into executable bot command
  parsePrompt(rawText, botUsername = this.defaultBotName, sender = null, onlinePlayers = []) {
    if (!rawText || typeof rawText !== 'string') {
      return {
        success: false,
        error: 'Empty prompt provided'
      };
    }

    const original = rawText.trim();
    let text = original.toLowerCase();

    // 1. Clean out greetings & bot nametag
    // Examples: "hai zombo", "hi zombo", "hey zombo", "hello zombo", "zombo,", "zombo:"
    const botName = (botUsername || 'zombo').toLowerCase();
    const botNamePattern = new RegExp(`\\b(hai|hi|hey|hello|yo|ok|please)\\s+(${botName}|zombo|survivalbot|bot)\\b`, 'i');
    text = text.replace(botNamePattern, '').trim();

    // Also remove standalone nametag if at beginning or end
    text = text.replace(new RegExp(`^(${botName}|zombo|survivalbot|bot)[,:\\s]*`, 'i'), '').trim();
    text = text.replace(new RegExp(`[,:\\s]*(${botName}|zombo|survivalbot|bot)$`, 'i'), '').trim();

    // Remove filler phrases like "can u", "can you", "could you", "would you", "please", "i want you to", "go and"
    text = text.replace(/\b(can\s+u|can\s+you|could\s+you|would\s+you|please|i\s+want\s+you\s+to|go\s+and|will\s+you)\b/gi, '').trim();

    // Clean punctuation
    const cleanText = text.replace(/[?!.,;]/g, ' ').replace(/\s+/g, ' ').trim();

    // 2. Intent matching

    // A0. SUICIDE / TERMINATE BOT (Supreme Master only)
    if (/\b(kill\s+(your\s*self|ur\s*self|self)|commit\s+suicide|die\b|suicide\b|end\s+your\s+life|terminate\s+your\s*self|destroy\s+your\s*self|drown\s+your\s*self|jump\s+off\s+(a\s+)?cliff|take\s+your\s+(own\s+)?life)\b/i.test(cleanText)) {
      return {
        success: true,
        intent: 'suicide',
        sender,
        rawPrompt: original,
        displayName: 'Terminate Self (Suicide Protocol)',
        reply: 'Executing self-termination command...',
        confidence: 0.99
      };
    }

    // A1. ROLE ASSIGNMENT IN CHAT (Supreme Master only)
    const roleAssignMatch = cleanText.match(/\b(?:make|set|assign|promote|grant)\s+(?:player\s+)?([a-zA-Z0-9_]+)\s+(?:as\s+|to\s+)?(supreme[- ]?master|master|small[- ]?master|[a-zA-Z0-9_-]+)\b/i);
    if (roleAssignMatch) {
      const targetPlayer = roleAssignMatch[1];
      const targetRole = roleAssignMatch[2].toLowerCase().replace(/[-_]/g, ' ');
      return {
        success: true,
        intent: 'assign_role',
        sender,
        targetPlayer,
        targetRole,
        rawPrompt: original,
        displayName: `Assign Role "${targetRole}" to ${targetPlayer}`,
        reply: `Setting role "${targetRole}" for player ${targetPlayer}...`,
        confidence: 0.98
      };
    }

    // A. STOP / CANCEL
    if (/^(stop|halt|cancel|freeze|hold\s*on|wait\s*here|stay\s*here|idle)$/i.test(cleanText) ||
        /\b(stop\s+all|stop\s+everything|stop\s+it|cancel\s+mission)\b/i.test(cleanText)) {
      return {
        success: true,
        intent: 'stop',
        sender,
        rawPrompt: original,
        displayName: 'Stop All Actions',
        reply: `Stopping current mission! Standing by for orders.`,
        confidence: 0.99
      };
    }

    // B. BODYGUARD / FOLLOW / PROTECT
    if (/\b(protect|guard|defend|follow|stay\s+with|come\s+with)\b/i.test(cleanText)) {
      const target = sender || 'master';
      return {
        success: true,
        intent: 'protect',
        sender,
        rawPrompt: original,
        target,
        displayName: `Protect & Follow ${target}`,
        reply: `Yes Sir! Following and protecting ${target}!`,
        confidence: 0.98
      };
    }

    // B2. CHEST & SPATIAL RESOURCE MEMORY QUERY
    if (/\b(where\s+(is|are)|where\s+can\s+i\s+find|find\s+resource|locate\s+resource|what('s|\s+is|\s+are)\s+in\s+(the\s+)?chest|check\s+chests?|list\s+chests?|show\s+chests?|inventory\s+of\s+chest|do\s+we\s+have|how\s+many|how\s+much)\b/i.test(cleanText)) {
      let itemQuery = null;
      for (const [alias, blockId] of Object.entries(BLOCK_ALIASES)) {
        if (cleanText.includes(alias)) {
          itemQuery = alias;
          break;
        }
      }
      if (!itemQuery) {
        const miscKeywords = ['iron', 'gold', 'diamond', 'coal', 'wood', 'bread', 'food', 'torch', 'torches', 'sword', 'pickaxe', 'armor', 'arrow', 'bow', 'cobblestone', 'stone', 'dirt'];
        for (const kw of miscKeywords) {
          if (cleanText.includes(kw)) {
            itemQuery = kw;
            break;
          }
        }
      }

      return {
        success: true,
        intent: 'chest_query',
        sender,
        rawPrompt: original,
        itemQuery: itemQuery || null,
        displayName: itemQuery ? `Locate ${itemQuery} in Chest Memory` : `Query Known Chest Network`,
        reply: itemQuery ? `Searching chest memory for ${itemQuery}...` : `Scanning all remembered chests across the world...`,
        confidence: 0.97
      };
    }

    // C. MINING / CHOPPING / DIGGING
    if (/\b(mine|dig|chop|harvest|extract|quarry|get\s+some|gather\s+wood|chop\s+wood)\b/i.test(cleanText)) {
      const count = this.extractCount(cleanText, 10);
      let targetBlock = 'iron_ore';
      let foundAlias = null;

      // Match block alias
      for (const [alias, blockId] of Object.entries(BLOCK_ALIASES)) {
        const re = new RegExp(`\\b${alias.replace(/_/g, '\\s*')}\\b`, 'i');
        if (re.test(cleanText)) {
          targetBlock = blockId;
          foundAlias = alias;
          break;
        }
      }

      const displayBlock = targetBlock.replace(/_/g, ' ');
      return {
        success: true,
        intent: 'mine',
        sender,
        rawPrompt: original,
        target: targetBlock,
        count,
        displayName: `Mine ${count} ${displayBlock}`,
        reply: `Roger that! Mining ${count} ${displayBlock} for you!`,
        confidence: 0.97
      };
    }

    // D. COMBAT & KILLING
    if (/\b(kill|attack|slay|hunt|fight|eliminate|destroy)\b/i.test(cleanText)) {
      let target = 'zombie';
      let isPlayer = false;

      // Check if target is a known player
      const playerMatch = cleanText.match(/\b(player|user)\s+([a-zA-Z0-9_]+)\b/i);
      if (playerMatch) {
        target = playerMatch[2];
        isPlayer = true;
      } else {
        // Check online players list
        for (const p of onlinePlayers) {
          if (cleanText.includes(p.toLowerCase())) {
            target = p;
            isPlayer = true;
            break;
          }
        }
      }

      // If not player, check mob aliases
      if (!isPlayer) {
        for (const [alias, mobId] of Object.entries(MOB_ALIASES)) {
          const re = new RegExp(`\\b${alias.replace(/_/g, '\\s*')}\\b`, 'i');
          if (re.test(cleanText)) {
            target = mobId;
            break;
          }
        }
      }

      const targetTitle = isPlayer ? `Player ${target}` : target.replace(/_/g, ' ');
      return {
        success: true,
        intent: 'kill',
        sender,
        rawPrompt: original,
        target,
        isPlayer,
        displayName: `Engage & Eliminate ${targetTitle}`,
        reply: `Combat engaged! Tracking down and eliminating ${targetTitle}!`,
        confidence: 0.96
      };
    }

    // E. FOOD & SUSTENANCE
    if (/\b(food|eat|hungry|meat|hunger|apples?|bread|rations)\b/i.test(cleanText) ||
        /\b(collect\s+food|gather\s+food|find\s+food|get\s+food)\b/i.test(cleanText)) {
      return {
        success: true,
        intent: 'food',
        sender,
        rawPrompt: original,
        displayName: 'Collect & Cook Food Rations',
        reply: `Hunting animal sustenance and collecting food for the rations stockpile!`,
        confidence: 0.95
      };
    }

    // F. FARMING & AGRICULTURE
    if (/\b(farm|farming|plant|crops?|wheat|harvest\s+wheat|seeds?)\b/i.test(cleanText)) {
      return {
        success: true,
        intent: 'farm',
        sender,
        rawPrompt: original,
        displayName: 'Establish Wheat Farm & Harvest Crops',
        reply: `Setting up agriculture! Tilling soil, planting seeds, and harvesting crops!`,
        confidence: 0.95
      };
    }

    // G. BUILDING & SHELTER
    if (/\b(build|make|construct)\b/i.test(cleanText) && /\b(shelter|house|base|wall|hut|box)\b/i.test(cleanText)) {
      return {
        success: true,
        intent: 'build',
        sender,
        structure: 'shelter',
        rawPrompt: original,
        displayName: 'Construct Survival Shelter',
        reply: `Gathering building materials and constructing survival shelter walls!`,
        confidence: 0.94
      };
    }

    // H. NAVIGATION & COORDINATES
    const coordMatch = cleanText.match(/\b(?:go\s*to|walk\s*to|move\s*to|run\s*to|coord|at)\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\b/i);
    if (coordMatch) {
      const x = parseInt(coordMatch[1], 10);
      const y = parseInt(coordMatch[2], 10);
      const z = parseInt(coordMatch[3], 10);
      return {
        success: true,
        intent: 'goto',
        sender,
        rawPrompt: original,
        x, y, z,
        displayName: `Navigate to [X: ${x}, Y: ${y}, Z: ${z}]`,
        reply: `Pathfinding initiated. Navigating to coordinates X:${x} Y:${y} Z:${z}!`,
        confidence: 0.98
      };
    }

    // I. FIND VILLAGE / EXPLORE
    if (/\b(village|town|villager|explore|find\s+village|locate\s+village)\b/i.test(cleanText)) {
      return {
        success: true,
        intent: 'find_village',
        sender,
        rawPrompt: original,
        displayName: 'Explore World for Village',
        reply: `Scouting terrain and searching for nearby NPC villages!`,
        confidence: 0.93
      };
    }

    // J. CRAFTING
    if (/\b(craft|make|create|forge)\b/i.test(cleanText)) {
      const count = this.extractCount(cleanText, 1);
      let craftItem = 'crafting_table';
      if (/pickaxe/i.test(cleanText)) craftItem = 'wooden_pickaxe';
      else if (/sword/i.test(cleanText)) craftItem = 'wooden_sword';
      else if (/axe/i.test(cleanText)) craftItem = 'wooden_axe';
      else if (/furnace/i.test(cleanText)) craftItem = 'furnace';
      else if (/torch/i.test(cleanText)) craftItem = 'torch';
      else if (/planks?/i.test(cleanText)) craftItem = 'oak_planks';
      else if (/sticks?/i.test(cleanText)) craftItem = 'stick';
      else if (/shield/i.test(cleanText)) craftItem = 'shield';

      return {
        success: true,
        intent: 'craft',
        sender,
        rawPrompt: original,
        item: craftItem,
        count,
        displayName: `Craft ${count} ${craftItem.replace(/_/g, ' ')}`,
        reply: `Opening crafting station to craft ${count} ${craftItem.replace(/_/g, ' ')}!`,
        confidence: 0.92
      };
    }

    // Fallback: If user mentioned an ore/item alone
    for (const [alias, blockId] of Object.entries(BLOCK_ALIASES)) {
      if (cleanText.includes(alias)) {
        const count = this.extractCount(cleanText, 10);
        return {
          success: true,
          intent: 'mine',
          rawPrompt: original,
          target: blockId,
          count,
          displayName: `Mine ${count} ${blockId.replace(/_/g, ' ')}`,
          reply: `Inferred mining objective. Mining ${count} ${blockId.replace(/_/g, ' ')}!`,
          confidence: 0.85
        };
      }
    }

    // Unknown prompt
    return {
      success: false,
      rawPrompt: original,
      error: `Could not determine objective from "${original}". Try: "hai Zombo can u mine diamond", "hai Zombo kill zombie", "hai Zombo build farm", or "hai Zombo protect me".`
    };
  }

  // Execute the parsed command on active ObjectiveEngine with Role & Permission Enforcement
  executeCommand(parsed, objectiveEngine, chatCallback = null, roleManager = null) {
    if (!parsed || !parsed.success) {
      if (chatCallback && parsed?.error) {
        chatCallback(`[Zombo] ${parsed.error}`);
      }
      return { success: false, error: parsed?.error || 'Invalid command' };
    }

    if (!objectiveEngine) {
      return { success: false, error: 'Survival Bot Objective Engine is offline. Spawn bot first!' };
    }

    // Role & Permission Access Control
    const rManager = roleManager || objectiveEngine.roleManager;
    if (rManager && parsed.sender) {
      const targetParam = parsed.isPlayer ? parsed.target : (parsed.intent === 'assign_role' ? parsed.targetPlayer : null);
      const permCheck = rManager.canExecuteTask(parsed.sender, parsed.intent, targetParam);
      if (!permCheck.allowed) {
        const denyReason = permCheck.reason || 'Permission denied: Your role does not allow this command.';
        if (chatCallback) {
          chatCallback(`[Zombo] ${denyReason}`);
        }
        return { success: false, error: denyReason, permissionDenied: true };
      }
    }

    // Special Intent: Suicide / Self-Termination
    if (parsed.intent === 'suicide') {
      if (typeof objectiveEngine.executeSuicide === 'function') {
        objectiveEngine.executeSuicide(chatCallback);
        return { success: true, intent: 'suicide', message: 'Executed self-termination command.' };
      }
    }

    // Special Intent: In-Game Role Assignment (Supreme Master)
    if (parsed.intent === 'assign_role') {
      if (rManager) {
        const assignRes = rManager.setPlayerRole(parsed.targetPlayer, parsed.targetRole, parsed.sender, false);
        if (assignRes.success) {
          const successMsg = `👑 Granted role "${assignRes.roleName}" (Priority: ${assignRes.priority}) to player ${assignRes.player}!`;
          if (chatCallback) chatCallback(`[Zombo] ${successMsg}`);
          return { success: true, message: successMsg, data: assignRes };
        } else {
          const failMsg = `Permission error: ${assignRes.error}`;
          if (chatCallback) chatCallback(`[Zombo] ${failMsg}`);
          return { success: false, error: assignRes.error };
        }
      }
    }

    if (chatCallback && parsed.reply) {
      chatCallback(`[Zombo] ${parsed.reply}`);
    }

    switch (parsed.intent) {
      case 'mine':
        objectiveEngine.startMining(parsed.target, parsed.count);
        break;
      case 'kill':
        objectiveEngine.startCombat(parsed.target, parsed.isPlayer);
        break;
      case 'protect':
        objectiveEngine.startBodyguard(parsed.target);
        break;
      case 'chest_query': {
        const mem = this.chestMemory || (objectiveEngine && objectiveEngine.chestMemory);
        if (mem) {
          if (parsed.itemQuery) {
            const res = mem.findResource(parsed.itemQuery);
            if (res && res.found) {
              const topLocs = res.locations.slice(0, 2).map(l => {
                const coordStr = l.coordinates ? ` at X:${l.coordinates.x} Y:${l.coordinates.y} Z:${l.coordinates.z}` : '';
                return `${l.chestName}${coordStr} (${l.totalCount}x)`;
              }).join('; ');
              const reply = `📦 Found ${res.totalQuantity}x ${parsed.itemQuery}! Stored in: ${topLocs}`;
              if (chatCallback) chatCallback(`[Zombo] ${reply}`);
              return { success: true, message: reply, data: res };
            } else {
              const reply = `📦 No ${parsed.itemQuery} found in known chests or pockets!`;
              if (chatCallback) chatCallback(`[Zombo] ${reply}`);
              return { success: true, message: reply };
            }
          } else {
            const agg = mem.getAggregatedResources();
            const topConst = agg.constantUsing.slice(0, 4).map(c => `${c.displayName}: ${c.totalCount}`).join(', ');
            const reply = `📦 Remembering ${agg.chestsCount} chests (${agg.totalTrackedItems} total items). Constant supplies: ${topConst}`;
            if (chatCallback) chatCallback(`[Zombo] ${reply}`);
            return { success: true, message: reply, data: agg };
          }
        }
        break;
      }
      case 'food':
        objectiveEngine.startCollectFood();
        break;
      case 'farm':
        objectiveEngine.startFarming();
        break;
      case 'goto':
        objectiveEngine.startGoTo(parsed.x, parsed.y, parsed.z);
        break;
      case 'find_village':
        objectiveEngine.startFindVillage();
        break;
      case 'build':
        objectiveEngine.startFarming(); // or shelter routine
        break;
      case 'craft':
        // If craft objective available
        objectiveEngine.startMining('oak_log', 4);
        break;
      case 'stop':
        objectiveEngine.stopCurrentObjective();
        break;
      default:
        objectiveEngine.stopCurrentObjective();
        break;
    }

    return {
      success: true,
      command: parsed
    };
  }
}

module.exports = {
  PromptCommander,
  BLOCK_ALIASES,
  MOB_ALIASES
};
