'use strict';
// server.js - Web Terminal & Central Control Hub for Minecraft Survival Bot.
// Serves the Minecraft UI Web Terminal, manages the Survival Bot, Player Recorder,
// real-time telemetry streaming (SSE), and incremental neuron evolution.

const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const { SurvivalBot } = require('./bot');
const { DemonstrationRecorder } = require('./record');
const { Brain } = require('./brain');
const { generateExpertDemos } = require('./generate_demos');
const { spawn } = require('child_process');
const { MinecraftKnowledgeEngine } = require('./knowledge');
const { TreeEngine } = require('./tree_engine');
const { PromptCommander } = require('./prompt_commander');
const { ChestMemoryEngine } = require('./chest_memory');
const { RoleManager, SYSTEM_ROLES } = require('./roles');
const knowledgeEngine = new MinecraftKnowledgeEngine('1.20.1');
const chestMemoryEngine = new ChestMemoryEngine(path.join(__dirname, 'data_bot'));
const roleManager = new RoleManager(path.join(__dirname, 'data_bot'));
const promptCommander = new PromptCommander('Zombo', chestMemoryEngine, roleManager);

const PORT = parseInt(process.env.PORT || '3000', 10);
const VIEWER_PORT = parseInt(process.env.VIEWER_PORT || '3007', 10);
const DATA_DIR = path.join(__dirname, 'data_bot');
const DEMO_DIR = path.join(__dirname, 'demos');
const YOUTUBE_DIR = path.join(__dirname, 'youtube_stream_bot');
const LINKS_FILE = path.join(YOUTUBE_DIR, 'links.txt');

// Protect server process from unhandled async socket errors (e.g. viewer port reuse)
process.on('uncaughtException', err => {
  if (err.code === 'EADDRINUSE') {
    console.warn(`[SERVER NOTICE] Port ${err.port || 'busy'} is in use. Skipping duplicate bind.`);
    return;
  }
  console.error('[SERVER UNCAUGHT EXCEPTION]', err.message);
});

fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(DEMO_DIR, { recursive: true });

let activeYoutubeProcess = null;
let youtubeQueue = [];
let youtubeLogs = [
  '[STREAM READY] In-memory YouTube stream engine initialized. 0 bytes written to disk.'
];

// Load existing links from links.txt if exists
try {
  if (fs.existsSync(LINKS_FILE)) {
    const rawLinks = fs.readFileSync(LINKS_FILE, 'utf8').split('\n');
    youtubeQueue = rawLinks
      .map(l => l.trim())
      .filter(l => l && !l.startsWith('#') && (l.includes('youtube.com') || l.includes('youtu.be')));
  }
} catch (e) {
  console.warn('[YOUTUBE INIT] Could not read links.txt:', e.message);
}

let youtubeTelemetry = {
  isRunning: false,
  currentVideo: null,
  bufferCount: 0,
  bufferMax: 50000,
  step: 0,
  totalSteps: 3000,
  btnLoss: 0,
  camLoss: 0,
  speed: 0,
  lastFrame: 0,
  lastKeys: 'IDLE',
  lastMouse: { dx: '+0.00', dy: '+0.00' }
};

const brain = new Brain(DATA_DIR);
const treeEngine = new TreeEngine({ brain, demosDir: DEMO_DIR, dataDir: DATA_DIR });

let activeBot = null;
let activeRecorder = null;
let sseClients = [];
let currentSkinUrl = '/textures/skins/steve.png';

// Baseline 46-slot inventory array matching authentic Minecraft container slots (0-45)
function createDefaultInventorySlots() {
  const slots = Array(46).fill(null);
  // Add initial starter equipment preview
  slots[36] = { slot: 36, name: 'iron_sword', displayName: 'Iron Sword', count: 1 };
  slots[37] = { slot: 37, name: 'diamond_pickaxe', displayName: 'Diamond Pickaxe', count: 1 };
  slots[38] = { slot: 38, name: 'iron_axe', displayName: 'Iron Axe', count: 1 };
  slots[39] = { slot: 39, name: 'water_bucket', displayName: 'Water Bucket', count: 1 };
  slots[40] = { slot: 40, name: 'bread', displayName: 'Bread', count: 16 };
  slots[41] = { slot: 41, name: 'torch', displayName: 'Torch', count: 32 };
  slots[42] = { slot: 42, name: 'oak_log', displayName: 'Oak Log', count: 64 };
  slots[43] = { slot: 43, name: 'cobblestone', displayName: 'Cobblestone', count: 64 };
  slots[44] = { slot: 44, name: 'shield', displayName: 'Shield', count: 1 };
  slots[5] = { slot: 5, name: 'iron_helmet', displayName: 'Iron Helmet', count: 1 };
  slots[6] = { slot: 6, name: 'iron_chestplate', displayName: 'Iron Chestplate', count: 1 };
  slots[7] = { slot: 7, name: 'iron_leggings', displayName: 'Iron Leggings', count: 1 };
  slots[8] = { slot: 8, name: 'iron_boots', displayName: 'Iron Boots', count: 1 };
  slots[45] = { slot: 45, name: 'shield', displayName: 'Shield', count: 1 };
  return slots;
}

let latestTelemetry = {
  botOnline: false,
  recorderActive: false,
  health: 20,
  food: 20,
  oxygen: 20,
  armor: 15,
  pos: { x: 0, y: 0, z: 0 },
  yaw: 0,
  pitch: 0,
  currentAction: 'idle',
  actionProb: 0,
  stuckStatus: 'IDLE',
  stopReason: null,
  task: 'survive',
  target: { name: 'none', dist: 0, inRange: false },
  brainGen: brain.generation,
  hiddenNeurons: brain.net.H,
  isRNN: true,
  rnnMemoryActive: 0,
  recentLogs: ['[SYSTEM] Neural terminal initialized (RNN Temporal Memory enabled). Ready for recording or testing.'],
  recentDecisions: [],
  recorderStats: { count: 0, lastAction: 'idle', target: 'none' },
  viewerPort: VIEWER_PORT,
  heldItem: 'iron_sword',
  skinUrl: currentSkinUrl,
  inventorySlots: createDefaultInventorySlots(),
  plannerStatus: null
};

function broadcastTelemetry() {
  if (!sseClients.length) return;
  latestTelemetry.brainGen = brain.generation;
  latestTelemetry.hiddenNeurons = brain.net.H;
  latestTelemetry.youtube = {
    ...youtubeTelemetry,
    isRunning: !!activeYoutubeProcess,
    latestLog: youtubeLogs.length ? youtubeLogs[youtubeLogs.length - 1] : ''
  };
  const payload = `data: ${JSON.stringify(latestTelemetry)}\n\n`;
  for (let i = sseClients.length - 1; i >= 0; i--) {
    try {
      sseClients[i].write(payload);
    } catch {
      sseClients.splice(i, 1);
    }
  }
}

// 10 Hz broadcast
setInterval(broadcastTelemetry, 100);

// Load all demos from disk
function getAllDemos() {
  const demos = [];
  if (!fs.existsSync(DEMO_DIR)) return demos;
  const files = fs.readdirSync(DEMO_DIR).filter(f => f.endsWith('.jsonl'));
  for (const f of files) {
    const lines = fs.readFileSync(path.join(DEMO_DIR, f), 'utf8').split('\n');
    for (const line of lines) {
      if (line.trim()) {
        try {
          demos.push(JSON.parse(line));
        } catch {}
      }
    }
  }
  return demos;
}

const server = http.createServer((req, res) => {
  const parsedUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const pathname = parsedUrl.pathname;

  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // 1. SSE Real-Time Telemetry Stream
  if (pathname === '/api/telemetry') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    });
    res.write(`data: ${JSON.stringify(latestTelemetry)}\n\n`);
    sseClients.push(res);
    req.on('close', () => {
      sseClients = sseClients.filter(c => c !== res);
    });
    return;
  }

  // 2. Status API
  if (pathname === '/api/status' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      botRunning: !!activeBot && activeBot.isRunning,
      recorderActive: !!activeRecorder && activeRecorder.isRecording,
      architecture: 'GRU (Gated Recurrent Unit with Long-Term Memory Gates)',
      brainGen: brain.generation,
      hiddenNeurons: brain.net.H,
      demosCount: getAllDemos().length,
      viewerPort: VIEWER_PORT
    }));
    return;
  }

  // 3. Start Player Observer Recording
  if (pathname === '/api/record/start' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const playerName = data.playerName || 'Player';
        const task = data.task || 'kill_skeleton';
        const host = data.host || 'localhost';
        const port = +(data.port || 25565);

        if (activeRecorder && activeRecorder.isRecording) {
          activeRecorder.stop();
        }

        activeRecorder = new DemonstrationRecorder({
          host,
          port,
          playerName,
          task,
          demoDir: DEMO_DIR,
          onSample: stats => {
            latestTelemetry.recorderStats = stats;
          },
          onLog: msg => {
            latestTelemetry.recentLogs.push(msg);
            if (latestTelemetry.recentLogs.length > 50) latestTelemetry.recentLogs.shift();
          }
        });

        activeRecorder.start();
        latestTelemetry.recorderActive = true;
        latestTelemetry.task = task;

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, message: `Watching ${playerName}` }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 4. Stop Recording & Trigger Neurogenesis ("Neuron forming starts")
  if (pathname === '/api/record/stop' && req.method === 'POST') {
    if (!activeRecorder || !activeRecorder.isRecording) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ message: 'No active recording to stop' }));
      return;
    }

    const recResult = activeRecorder.stop();
    latestTelemetry.recorderActive = false;
    activeRecorder = null;

    let evolutionReport = null;
    if (recResult.samples && recResult.samples.length >= 15) {
      latestTelemetry.recentLogs.push(`[EXPERIENCE REPLAY] Blending ${recResult.samples.length} new samples with historical memory buffer...`);
      const allHistorical = getAllDemos();
      let trainingBatch = recResult.samples.slice();
      // Shuffle in up to 1000 past demonstrations to prevent catastrophic forgetting
      if (allHistorical.length > 0) {
        const replaySample = allHistorical.sort(() => 0.5 - Math.random()).slice(0, Math.min(1000, allHistorical.length));
        trainingBatch = trainingBatch.concat(replaySample);
      }
      evolutionReport = brain.evolveFromDemos(trainingBatch);
      latestTelemetry.recentLogs.push(`[EVOLUTION] Generation ${brain.generation} formed! Kept: ${evolutionReport.kept}, Cut: ${evolutionReport.cut}, Added: ${evolutionReport.added} neurons. Accuracy: ${evolutionReport.accuracyAfter}%`);
    } else {
      latestTelemetry.recentLogs.push(`[RECORDER] Demo recorded with ${recResult.samples.length} samples (need 15+ to evolve).`);
    }

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      success: true,
      samplesSaved: recResult.count,
      evolution: evolutionReport
    }));
    return;
  }

  // 5. Retrain / Evolve from all saved demos
  if (pathname === '/api/train' && req.method === 'POST') {
    const demos = getAllDemos();
    if (demos.length < 15) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: `Need at least 15 demonstration samples (found ${demos.length})` }));
      return;
    }

    latestTelemetry.recentLogs.push(`[TRAIN] Evolving brain from ${demos.length} total demonstration samples...`);
    const report = brain.evolveFromDemos(demos);
    latestTelemetry.recentLogs.push(`[EVOLUTION] Generation ${brain.generation} ready! Hidden neurons: ${report.hiddenAfter} (Accuracy: ${report.accuracyAfter}%)`);

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true, report }));
    return;
  }

  // 5b. Auto-Generate Expert Demonstration Data & Evolve Neurons
  if (pathname === '/api/demos/generate' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const count = data.count || 250;
        latestTelemetry.recentLogs.push(`[DATA GEN] Synthesizing ${count} expert Minecraft combat & movement samples...`);
        const { samples } = generateExpertDemos(count, DEMO_DIR);

        latestTelemetry.recentLogs.push(`[NEUROGENESIS] Scoring neurons & sprouting fresh plastic neurons (+4)...`);
        const report = brain.evolveFromDemos(samples);
        latestTelemetry.brainGen = brain.generation;
        latestTelemetry.hiddenNeurons = brain.net.H;
        latestTelemetry.recentLogs.push(`✨ [NEURON EVOLUTION] Gen ${report.generation}: Neurons increased to ${report.hiddenAfter} (Kept: ${report.kept}, Cut: ${report.cut}, Added: ${report.added})!`);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, report }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 6. Start Survival Bot (Test Mode)
  if (pathname === '/api/bot/start' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const username = data.username || 'SurvivalBot';
        const task = data.task || 'survive';
        const host = data.host || 'localhost';
        const port = +(data.port || 25565);

        if (activeBot && activeBot.isRunning) {
          activeBot.stop();
        }

        activeBot = new SurvivalBot({
          host,
          port,
          username,
          task,
          dataDir: DATA_DIR,
          viewerPort: VIEWER_PORT,
          onTelemetry: telem => {
            latestTelemetry = {
              ...latestTelemetry,
              ...telem,
              botOnline: true
            };
          },
          onLog: msg => {
            latestTelemetry.recentLogs.push(msg);
            if (latestTelemetry.recentLogs.length > 50) latestTelemetry.recentLogs.shift();
          }
        });

        activeBot.start();
        latestTelemetry.botOnline = true;
        latestTelemetry.task = task;

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, message: `Bot ${username} launched.` }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 7. Stop Survival Bot
  if (pathname === '/api/bot/stop' && req.method === 'POST') {
    if (activeBot) {
      activeBot.stop();
      activeBot = null;
    }
    latestTelemetry.botOnline = false;
    latestTelemetry.stuckStatus = 'STOPPED';
    latestTelemetry.stopReason = 'Bot manually stopped by user.';
    latestTelemetry.recentLogs.push('[SYSTEM] Survival bot stopped.');

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true, message: 'Bot stopped.' }));
    return;
  }

  // 8. Brain History API
  if (pathname === '/api/history' && req.method === 'GET') {
    const historyFile = path.join(DATA_DIR, 'history.json');
    if (fs.existsSync(historyFile)) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(fs.readFileSync(historyFile, 'utf8'));
    } else {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify([]));
    }
    return;
  }

  // 8b. Brain Inspector Report API
  if (pathname === '/api/brain/report' && req.method === 'GET') {
    try {
      const net = brain.net;
      const H = net.H;
      // h_state snapshot
      const hState = net.h_state || new Array(H).fill(0);
      const activeMemCells = hState.filter(v => Math.abs(v) > 0.2).length;
      // Recurrent memory matrix stats (Uh for GRU, Wh for legacy RNN)
      const recMatrix = net.Uh || net.Wh || [];
      let whAbsSum = 0;
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < H; j++) {
          whAbsSum += Math.abs((recMatrix[i] && recMatrix[i][j]) || 0);
        }
      }
      const whMeanAbs = H > 0 ? whAbsSum / (H * H) : 0;
      // Recurrent diagonal (self-connections = memory persistence per neuron)
      const whDiag = Array.from({ length: H }, (_, i) => (recMatrix[i] && recMatrix[i][i]) || 0);
      // History
      const historyFile = path.join(DATA_DIR, 'history.json');
      const history = fs.existsSync(historyFile)
        ? JSON.parse(fs.readFileSync(historyFile, 'utf8'))
        : [];
      // Demo sample count
      let demoCount = 0;
      if (fs.existsSync(DEMO_DIR)) {
        for (const f of fs.readdirSync(DEMO_DIR)) {
          if (f.endsWith('.jsonl')) {
            const lines = fs.readFileSync(path.join(DEMO_DIR, f), 'utf8').split('\n').filter(Boolean);
            demoCount += lines.length;
          }
        }
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        generation: brain.generation,
        hiddenNeurons: H,
        maxHidden: 128,
        isRNN: true,
        hState,
        activeMemCells,
        whMeanAbs: +whMeanAbs.toFixed(6),
        whDiag,
        demoCount,
        history,
        rnWorking: activeMemCells > 0 && whMeanAbs > 0.001
      }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8c. Survival Planner & Tech Tree Status API
  if (pathname === '/api/planner/status' && req.method === 'GET') {
    try {
      if (activeBot && activeBot.planner) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          active: true,
          ...activeBot.planner.getStatus()
        }));
      } else {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          active: false,
          currentGoal: {
            id: 'awaiting_spawn',
            name: '⏳ Waiting for Survival Bot',
            description: 'Launch Bot in Test Mode to activate Tech Tree & Crafting Planner'
          },
          milestones: [
            { id: 'gather_wood', name: 'Gather Wood Logs', textureUrl: '/textures/blocks/oak_log.png', target: 4, current: 0, done: false },
            { id: 'craft_planks', name: 'Craft Wooden Planks', textureUrl: '/textures/blocks/oak_planks.png', target: 8, current: 0, done: false },
            { id: 'craft_table', name: 'Craft Crafting Table', textureUrl: '/textures/blocks/crafting_table_front.png', target: 1, current: 0, done: false },
            { id: 'craft_sticks', name: 'Craft Sticks', textureUrl: '/textures/items/stick.png', target: 4, current: 0, done: false },
            { id: 'craft_wooden_sword', name: 'Craft Wooden Sword', textureUrl: '/textures/items/wooden_sword.png', target: 1, current: 0, done: false },
            { id: 'craft_wooden_pickaxe', name: 'Craft Wooden Pickaxe', textureUrl: '/textures/items/wooden_pickaxe.png', target: 1, current: 0, done: false },
            { id: 'mine_cobblestone', name: 'Mine Cobblestone', textureUrl: '/textures/blocks/cobblestone.png', target: 8, current: 0, done: false },
            { id: 'craft_stone_gear', name: 'Upgrade to Stone Weapons & Tools', textureUrl: '/textures/items/stone_sword.png', target: 1, current: 0, done: false }
          ],
          progressPercent: 0,
          inventory: { wood: 0, planks: 0, sticks: 0, food: 0, hasSword: false, hasPickaxe: false, hasCraftingTable: false }
        }));
      }
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8c0. Role & Permission Management API (Supreme-Master, Master, Small Master, Custom Roles)
  if (pathname === '/api/roles' && req.method === 'GET') {
    const snapshot = roleManager.getStateSnapshot();
    snapshot.onlinePlayers = (activeBot && activeBot.bot && activeBot.bot.players) ? Object.keys(activeBot.bot.players) : [];
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(snapshot));
    return;
  }

  if (pathname === '/api/roles/assign' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const resData = roleManager.setPlayerRole(data.player, data.role, null, true);
        if (resData.success) {
          if (data.role === SYSTEM_ROLES.SUPREME_MASTER || data.role === SYSTEM_ROLES.MASTER) {
            if (activeBot && activeBot.objectives) {
              activeBot.objectives.setMaster(data.player);
            }
          }
          latestTelemetry.recentLogs.push(`[ROLE ASSIGNED] Player "${data.player}" -> ${data.role.toUpperCase()}`);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(resData));
        } else {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(resData));
        }
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  if (pathname === '/api/roles/unassign' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const resData = roleManager.removePlayerRole(data.player);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(resData));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  if (pathname === '/api/roles/custom' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const resData = roleManager.createCustomRole(data.roleKey, data);
        if (resData.success) {
          latestTelemetry.recentLogs.push(`[CUSTOM ROLE] Created role "${data.name || data.roleKey}" (Priority: ${data.priority || 30})`);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(resData));
        } else {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(resData));
        }
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  if (pathname === '/api/roles/custom' && req.method === 'DELETE') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const parsedUrl = url.parse(req.url, true);
        const roleKey = parsedUrl.query.key || (JSON.parse(body || '{}').roleKey);
        const resData = roleManager.deleteCustomRole(roleKey);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(resData));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 8c1. Bot Role & Master Commander Assignment
  if (pathname === '/api/bot/role' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const role = data.role || 'autonomous';
        const master = data.master || null;
        if (activeBot && activeBot.objectives) {
          activeBot.objectives.setRole(role, master);
        }
        latestTelemetry.recentLogs.push(`[ROLE ASSIGNED] Role: ${role.toUpperCase()} | Commander: ${master || 'None'}`);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, role, master }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 8c2. Bot Objective Dispatcher (Mine, Combat, Bodyguard, GoTo, Village, Food, Farm)
  if (pathname === '/api/bot/objective' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', async () => {
      try {
        const data = JSON.parse(body || '{}');
        const type = data.type;
        if (!activeBot || !activeBot.isRunning || !activeBot.objectives) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Bot is offline. Spawn bot in Survival Mode first!' }));
          return;
        }

        if (type === 'mine') {
          activeBot.objectives.startMining(data.target || 'iron_ore', data.count || 10);
        } else if (type === 'kill') {
          activeBot.objectives.startCombat(data.target || 'zombie', !!data.isPlayer);
        } else if (type === 'protect') {
          activeBot.objectives.startBodyguard(data.target || data.master);
        } else if (type === 'goto') {
          activeBot.objectives.startGoTo(data.x, data.y, data.z);
        } else if (type === 'find_village') {
          activeBot.objectives.startFindVillage();
        } else if (type === 'food') {
          activeBot.objectives.startCollectFood();
        } else if (type === 'farm') {
          activeBot.objectives.startFarming();
        } else {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: `Unknown objective type: ${type}` }));
          return;
        }

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, objective: type, target: data.target }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 8c3. Stop Current Active Objective
  if (pathname === '/api/bot/stop_objective' && req.method === 'POST') {
    if (activeBot && activeBot.objectives) {
      activeBot.objectives.stopCurrentObjective();
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true, message: 'Objective stopped' }));
    return;
  }

  // 8c3b. Natural Language AI Prompt Commander (Mine, Combat, Farming, Building, Follow, Navigation)
  if (pathname === '/api/bot/prompt' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', async () => {
      try {
        const data = JSON.parse(body || '{}');
        const rawPrompt = data.prompt || '';
        const botName = (activeBot && activeBot.username) || 'Zombo';
        const sender = data.sender || (activeBot && activeBot.objectives && activeBot.objectives.masterPlayer) || 'Commander';
        const onlinePlayers = (activeBot && activeBot.bot && activeBot.bot.players) ? Object.keys(activeBot.bot.players) : [];

        const parsed = promptCommander.parsePrompt(rawPrompt, botName, sender, onlinePlayers);

        if (!parsed.success) {
          res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({ success: false, error: parsed.error, parsed }));
          return;
        }

        if (!activeBot || !activeBot.isRunning || !activeBot.objectives) {
          res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({
            success: true,
            botOnline: false,
            parsed,
            warning: 'Bot is offline! Launch Survival Bot in Test Mode to execute live in Minecraft.',
            reply: parsed.reply
          }));
          return;
        }

        // Execute parsed command on active bot
        const execResult = promptCommander.executeCommand(parsed, activeBot.objectives, (chatMsg) => {
          if (activeBot && activeBot.bot && typeof activeBot.bot.chat === 'function') {
            try { activeBot.bot.chat(chatMsg); } catch {}
          }
        }, roleManager);

        latestTelemetry.recentLogs.push(`[AI PROMPT] "${rawPrompt}" -> ${parsed.displayName}`);
        latestTelemetry.recentLogs.push(`[ZOMBO SPEECH] "${parsed.reply}"`);

        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({
          success: true,
          botOnline: true,
          parsed,
          reply: parsed.reply,
          execResult
        }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 8c3c. Spatial Chest & World Resource Memory (Constant-use, Rarity, Quantities)
  if (pathname === '/api/bot/chest_memory' && req.method === 'GET') {
    try {
      const mem = (activeBot && activeBot.chestMemory) ? activeBot.chestMemory : chestMemoryEngine;
      const botPos = (activeBot && activeBot.bot && activeBot.bot.entity) ? activeBot.bot.entity.position : null;
      const agg = mem.getAggregatedResources(botPos);
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(agg));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8c3d. Find Specific Resource in Chest Network
  if (pathname === '/api/bot/chest_find' && req.method === 'GET') {
    try {
      const q = parsedUrl.searchParams.get('q') || '';
      const mem = (activeBot && activeBot.chestMemory) ? activeBot.chestMemory : chestMemoryEngine;
      const results = mem.findResource(q);
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(results));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8c3e. Reset Neural Brain to 0 (Clear previous synthetic demo checkpoints)
  if (pathname === '/api/bot/reset_brain' && req.method === 'POST') {
    try {
      if (activeBot && activeBot.brain) {
        activeBot.brain.net = new (require('./brain').NeuroNet)();
        activeBot.brain.generation = 0;
        activeBot.brain.bestScore = -999999;
      }
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({
        success: true,
        message: 'Brain reset to generation 0 with clean state!'
      }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8c3f. Get Current Bot Skin & Presets
  if (pathname === '/api/bot/skin' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({
      currentSkin: currentSkinUrl,
      presets: ['steve', 'alex', 'ari', 'efe', 'kai', 'makena', 'noor', 'sunny', 'zuri']
    }));
    return;
  }

  // 8c3g. Upload Custom Minecraft Skin (.png) or Select Preset
  if (pathname === '/api/bot/skin' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        if (data.preset) {
          const valid = ['steve', 'alex', 'ari', 'efe', 'kai', 'makena', 'noor', 'sunny', 'zuri'];
          const presetName = data.preset.toLowerCase().trim();
          if (valid.includes(presetName)) {
            currentSkinUrl = `/textures/skins/${presetName}.png`;
            latestTelemetry.skinUrl = currentSkinUrl;
            res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(JSON.stringify({ success: true, url: currentSkinUrl, preset: presetName }));
            return;
          }
        }

        if (data.skinBase64) {
          const raw = data.skinBase64.replace(/^data:image\/\w+;base64,/, '');
          const buf = Buffer.from(raw, 'base64');
          const skinsDir = path.join(__dirname, 'public', 'textures', 'skins');
          fs.mkdirSync(skinsDir, { recursive: true });
          const skinPath = path.join(skinsDir, 'custom_skin.png');
          fs.writeFileSync(skinPath, buf);
          currentSkinUrl = `/textures/skins/custom_skin.png?t=${Date.now()}`;
          latestTelemetry.skinUrl = currentSkinUrl;
          res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({ success: true, url: currentSkinUrl, custom: true }));
          return;
        }

        res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: 'Missing skinBase64 or preset' }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 8c4. Dynamic Minecraft Knowledge Search
  if (pathname === '/api/minecraft/knowledge' && req.method === 'GET') {
    const q = parsedUrl.searchParams.get('q') || '';
    const results = knowledgeEngine.searchKnowledge(q);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ query: q, results }));
    return;
  }

  // 8c5. Detailed Recipe Details
  if (pathname === '/api/minecraft/recipe' && req.method === 'GET') {
    const item = parsedUrl.searchParams.get('item') || '';
    const details = knowledgeEngine.getRecipeDetails(item);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(details || { error: 'Item not found' }));
    return;
  }

  // 8c6. Hierarchical Survival Learning Tree (Overworld, Nether, End)
  if (pathname === '/api/learning-tree' && req.method === 'GET') {
    try {
      const tree = treeEngine.buildTree();
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(tree));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8c7. Learning Tree Attribute Search & Autocomplete
  if (pathname === '/api/learning-tree/search' && req.method === 'GET') {
    try {
      const q = parsedUrl.searchParams.get('q') || '';
      const limit = parseInt(parsedUrl.searchParams.get('limit') || '30', 10);
      const results = treeEngine.search(q, limit);
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ query: q, results }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8c8. Learning Tree Node Detail & Bidirectional Traversal (Ancestors, Children, Uses)
  if (pathname === '/api/learning-tree/node' && req.method === 'GET') {
    try {
      const id = parsedUrl.searchParams.get('id') || '';
      const nodeData = treeEngine.findNodeWithTraversal(id);
      if (!nodeData) {
        res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: `Node ${id} not found` }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(nodeData));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8d. YouTube In-Memory Stream Bot API: Get Queue & Status
  if (pathname === '/api/youtube/status' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      queue: youtubeQueue,
      ...youtubeTelemetry,
      isRunning: !!activeYoutubeProcess,
      logs: youtubeLogs
    }));
    return;
  }

  // 8e. YouTube In-Memory Stream Bot API: Add Link to Queue
  if (pathname === '/api/youtube/queue' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const urlStr = (data.url || '').trim();
        if (!urlStr || (!urlStr.includes('youtube.com') && !urlStr.includes('youtu.be'))) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Valid YouTube URL required' }));
          return;
        }

        if (!youtubeQueue.includes(urlStr)) {
          youtubeQueue.push(urlStr);
          fs.appendFileSync(LINKS_FILE, '\n' + urlStr);
        }

        // Dynamically pipe link into python process stdin if actively running
        if (activeYoutubeProcess && activeYoutubeProcess.stdin && !activeYoutubeProcess.stdin.destroyed) {
          activeYoutubeProcess.stdin.write(urlStr + '\n');
        }

        latestTelemetry.recentLogs.push(`[YOUTUBE QUEUE] Enqueued: ${urlStr}`);
        if (latestTelemetry.recentLogs.length > 50) latestTelemetry.recentLogs.shift();

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, queue: youtubeQueue }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 8f. YouTube In-Memory Stream Bot API: Remove Link
  if (pathname === '/api/youtube/remove' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const idx = typeof data.index === 'number' ? data.index : -1;
        if (idx >= 0 && idx < youtubeQueue.length) {
          youtubeQueue.splice(idx, 1);
          fs.writeFileSync(LINKS_FILE, youtubeQueue.join('\n') + '\n');
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, queue: youtubeQueue }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // 8g. YouTube In-Memory Stream Bot API: Start Streaming Pipeline
  if (pathname === '/api/youtube/start' && req.method === 'POST') {
    if (activeYoutubeProcess) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, message: 'Pipeline already running' }));
      return;
    }

    try {
      latestTelemetry.recentLogs.push('[YOUTUBE BOT] Spawning Python In-Memory Stream & Imitation Pipeline...');
      youtubeTelemetry.isRunning = true;

      activeYoutubeProcess = spawn('python', ['-u', 'train_policy.py', '--links-file', 'links.txt'], {
        cwd: YOUTUBE_DIR,
        shell: true
      });

      activeYoutubeProcess.stdout.on('data', chunk => {
        const text = chunk.toString();
        const lines = text.split('\n').filter(Boolean);
        for (const line of lines) {
          console.log('[PY STREAM]', line);
          youtubeLogs.push(line);
          if (youtubeLogs.length > 120) youtubeLogs.shift();

          latestTelemetry.recentLogs.push(line);
          if (latestTelemetry.recentLogs.length > 50) latestTelemetry.recentLogs.shift();

          if (line.includes('---> Streaming from:')) {
            const parts = line.split('---> Streaming from:');
            if (parts[1]) youtubeTelemetry.currentVideo = parts[1].trim();
          }

          if (line.includes('IDM Inferred Keys:')) {
            const keysMatch = line.match(/IDM Inferred Keys: \[(.*?)\]/);
            const mouseMatch = line.match(/Mouse: \(dx:([+-]?[\d\.]+),\s*dy:([+-]?[\d\.]+)\)/);
            const frameMatch = line.match(/Frame #(\d+)/);
            if (frameMatch) youtubeTelemetry.lastFrame = parseInt(frameMatch[1], 10);
            if (keysMatch) youtubeTelemetry.lastKeys = keysMatch[1];
            if (mouseMatch) youtubeTelemetry.lastMouse = { dx: mouseMatch[1], dy: mouseMatch[2] };
          }

          if (line.includes('Total Loss:')) {
            const stepMatch = line.match(/Step \[(\d+)\/(\d+)\]/);
            const lossMatch = line.match(/Total Loss: ([\d\.]+)/);
            const btnMatch = line.match(/Btn Loss: ([\d\.]+)/);
            const camMatch = line.match(/Cam Loss: ([\d\.]+)/);
            const bufMatch = line.match(/Buffer: (\d+)/);
            const spdMatch = line.match(/Speed: ([\d\.]+)/);

            if (stepMatch) {
              youtubeTelemetry.step = parseInt(stepMatch[1], 10);
              youtubeTelemetry.totalSteps = parseInt(stepMatch[2], 10);
            }
            if (lossMatch) youtubeTelemetry.totalLoss = parseFloat(lossMatch[1]);
            if (btnMatch) youtubeTelemetry.btnLoss = parseFloat(btnMatch[1]);
            if (camMatch) youtubeTelemetry.camLoss = parseFloat(camMatch[1]);
            if (bufMatch) youtubeTelemetry.bufferCount = parseInt(bufMatch[1], 10);
            if (spdMatch) youtubeTelemetry.speed = parseFloat(spdMatch[1]);
          }
        }
      });

      activeYoutubeProcess.stderr.on('data', chunk => {
        const errText = chunk.toString().trim();
        if (errText) {
          latestTelemetry.recentLogs.push(`[PY WARN] ${errText.slice(0, 100)}`);
          if (latestTelemetry.recentLogs.length > 50) latestTelemetry.recentLogs.shift();
        }
      });

      activeYoutubeProcess.on('close', code => {
        console.log(`[YOUTUBE BOT] Python process exited with code ${code}`);
        latestTelemetry.recentLogs.push(`[YOUTUBE BOT] Pipeline process stopped (code ${code}).`);
        activeYoutubeProcess = null;
        youtubeTelemetry.isRunning = false;
      });

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, message: 'Streaming pipeline started' }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8h. YouTube In-Memory Stream Bot API: Stop Pipeline
  if (pathname === '/api/youtube/stop' && req.method === 'POST') {
    if (activeYoutubeProcess) {
      activeYoutubeProcess.kill();
      activeYoutubeProcess = null;
      youtubeTelemetry.isRunning = false;
      latestTelemetry.recentLogs.push('[YOUTUBE BOT] Termination signal sent to Python pipeline.');
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true, message: 'Pipeline stopped' }));
    return;
  }

  // 9. Static Web Files (Minecraft UI Dashboard)
  let filePath = path.join(__dirname, 'public', pathname === '/' ? 'index.html' : pathname);
  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const ext = path.extname(filePath);
    const mimeTypes = {
      '.html': 'text/html; charset=utf-8',
      '.css': 'text/css',
      '.js': 'text/javascript',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.svg': 'image/svg+xml',
      '.json': 'application/json'
    };
    res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
    fs.createReadStream(filePath).pipe(res);
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('404 Not Found');
  }
});

server.listen(PORT, () => {
  console.log(`========================================================================`);
  console.log(`🟩 MINECRAFT SURVIVAL BOT & NEURAL TERMINAL ACTIVE!`);
  console.log(`👉 Open Web Terminal: http://localhost:${PORT}`);
  console.log(`🎮 3D First-Person Viewer port: ${VIEWER_PORT}`);
  console.log(`🧠 Brain Generation: ${brain.generation} | Hidden Neurons: ${brain.net.H}`);
  console.log(`========================================================================`);
});
