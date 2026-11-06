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
const { Brain, newbieNet } = require('./brain');
const { generateExpertDemos } = require('./generate_demos');
const { spawn, execSync } = require('child_process');
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
const LOCAL_VIDEOS_DIR = path.join(DATA_DIR, 'local_videos');
fs.mkdirSync(LOCAL_VIDEOS_DIR, { recursive: true });

let activeYoutubeProcess = null;
let youtubeQueue = [];
let youtubeLogs = [
  '[STREAM READY] In-memory Video & Stream Engine active. Zero disk writes during streaming.'
];

function isVideoSource(str) {
  if (!str) return false;
  const s = str.trim().replace(/^['"]|['"]$/g, '');
  if (!s || s.startsWith('#')) return false;
  if (s.startsWith('http://') || s.startsWith('https://')) return true;
  const exts = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'];
  return exts.some(ext => s.toLowerCase().endsWith(ext)) || (fs.existsSync(s) && fs.statSync(s).isFile());
}

// Load existing links from links.txt if exists
try {
  if (fs.existsSync(LINKS_FILE)) {
    const rawLinks = fs.readFileSync(LINKS_FILE, 'utf8').split('\n');
    youtubeQueue = rawLinks
      .map(l => l.trim().replace(/^['"]|['"]$/g, ''))
      .filter(l => isVideoSource(l));
  }
} catch (e) {
  console.warn('[VIDEO INIT] Could not read links.txt:', e.message);
}

let youtubeTelemetry = {
  isRunning: false,
  isPaused: false,
  checkpoint: null,
  resumedFrom: null,
  currentVideo: null,
  videoTitle: 'Minecraft Gameplay',
  bufferCount: 0,
  bufferMax: 50000,
  step: 0,
  totalSteps: 3000,
  btnLoss: 0,
  camLoss: 0,
  speed: 0,
  lastFrame: 0,
  totalFrames: 0,
  videoPercent: 0,
  currentTimeStr: '00:00',
  totalTimeStr: '00:00',
  currentSeconds: 0,
  totalSeconds: 0,
  speedFps: 0,
  speedFactor: 1.0,
  videoEta: '--:--',
  videoEtaSeconds: 0,
  setProgress: 0,
  setEta: '--:--',
  setEtaSeconds: 0,
  queueIndex: 1,
  queueTotal: 1,
  lastKeys: 'IDLE',
  lastMouse: { dx: '+0.00', dy: '+0.00' }
};

const CHECKPOINT_META_PATH = path.join(YOUTUBE_DIR, 'checkpoint_meta.json');
function loadCheckpointMeta(retries = 3) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      if (fs.existsSync(CHECKPOINT_META_PATH)) {
        const raw = fs.readFileSync(CHECKPOINT_META_PATH, 'utf8').trim();
        if (raw.startsWith('{') && raw.endsWith('}')) {
          const data = JSON.parse(raw);
          youtubeTelemetry.checkpoint = data;
          return data;
        }
      }
    } catch (err) {
      if (attempt === retries - 1) {
        console.warn('[CHECKPOINT] Failed to read checkpoint metadata:', err.message);
      }
    }
  }
  youtubeTelemetry.checkpoint = null;
  return null;
}
loadCheckpointMeta();

const POLICY_PATH = path.join(YOUTUBE_DIR, 'minecraft_policy.pth');
function getDeepPolicyStats() {
  const hasModel = fs.existsSync(POLICY_PATH);
  const cp = loadCheckpointMeta(1);
  return {
    hasModel,
    totalParameters: cp ? (cp.total_parameters || 1188683) : 1109355,
    gruUnits: cp ? (cp.gru_neurons || 288) : 256,
    cnnChannels: 480,
    trainedFrames: cp ? (cp.frame || 0) : 0,
    trainedSteps: cp ? (cp.step || 0) : 0,
    videoDuration: cp ? (cp.video_duration || 0) : 0,
    videoSeconds: cp ? (cp.video_seconds || 0) : 0,
    videoTitle: cp ? (cp.video_title || '') : ''
  };
}

const brain = new Brain(DATA_DIR);
const treeEngine = new TreeEngine({ brain, demosDir: DEMO_DIR, dataDir: DATA_DIR, youtubeDir: YOUTUBE_DIR });

let activeBot = null;
let activeRecorder = null;
let sseClients = [];
let currentSkinUrl = '/textures/skins/steve.png';

// Baseline 46-slot inventory array matching authentic Minecraft container slots (0-45)
function createDefaultInventorySlots() {
  // Real-time pure empty inventory by default. Populated ONLY by live Mineflayer bot.inventory.slots
  return Array(46).fill(null);
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
  deepNeurons: 1109355,
  deepPolicy: getDeepPolicyStats(),
  isRNN: true,
  rnnMemoryActive: 0,
  recentLogs: ['[SYSTEM] Neural terminal initialized (RNN Temporal Memory enabled). Ready for recording or testing.'],
  recentDecisions: [],
  recorderStats: { count: 0, lastAction: 'idle', target: 'none' },
  viewerPort: VIEWER_PORT,
  heldItem: null,
  skinUrl: currentSkinUrl,
  inventorySlots: createDefaultInventorySlots(),
  plannerStatus: null
};

function broadcastTelemetry() {
  if (!sseClients.length) return;
  latestTelemetry.brainGen = brain.generation;
  latestTelemetry.hiddenNeurons = brain.net.H;
  latestTelemetry.deepNeurons = 1109355;
  latestTelemetry.deepPolicy = getDeepPolicyStats();
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

  // 8c3e. Nuclear Purge: Delete ALL Learned Neural Weights, Checkpoints, and Models
  function performCompletePurge() {
    console.log('[PURGE] Executing complete purge of all learned neural models, weights, and checkpoints...');

    // 1. Terminate running bot or youtube pipeline
    if (activeBot) {
      try { activeBot.stop(); } catch {}
      activeBot = null;
    }
    if (activeYoutubeProcess) {
      try {
        if (process.platform === 'win32' && activeYoutubeProcess.pid) {
          spawn('taskkill', ['/pid', activeYoutubeProcess.pid.toString(), '/T', '/F']);
        } else {
          activeYoutubeProcess.kill('SIGKILL');
        }
      } catch {}
      activeYoutubeProcess = null;
      youtubeTelemetry.isRunning = false;
      youtubeTelemetry.isPaused = false;
    }

    // 2. Reset Brain in-memory and write fresh Gen 0 weights
    brain.net = newbieNet();
    brain.generation = 0;
    brain.bestScore = -999999;
    brain.save();

    // 3. Delete past generation JSON snapshots and history
    if (fs.existsSync(DATA_DIR)) {
      const files = fs.readdirSync(DATA_DIR);
      for (const f of files) {
        if ((f.startsWith('gen_') || f.startsWith('brain_legacy_')) && f.endsWith('.json')) {
          try { fs.unlinkSync(path.join(DATA_DIR, f)); } catch {}
        }
      }
      try {
        fs.writeFileSync(path.join(DATA_DIR, 'history.json'), '[]');
      } catch {}
    }

    // 4. Delete YouTube Checkpoint and reset imitation policy weights to clean random init
    const checkpointMetaPath = path.join(YOUTUBE_DIR, 'checkpoint_meta.json');
    if (fs.existsSync(checkpointMetaPath)) {
      try { fs.unlinkSync(checkpointMetaPath); } catch {}
    }
    const policyPath = path.join(YOUTUBE_DIR, 'minecraft_policy.pth');
    try {
      const resetPy = `import torch; from models import MinecraftPolicy; torch.save(MinecraftPolicy().state_dict(), 'minecraft_policy.pth')`;
      execSync(`python -c "${resetPy}"`, { cwd: YOUTUBE_DIR, timeout: 8000 });
    } catch (e) {
      console.warn('[PURGE] Policy reinit via python failed, unlinking file:', e.message);
      if (fs.existsSync(policyPath)) {
        try { fs.unlinkSync(policyPath); } catch {}
      }
    }

    // 5. Reset YouTube Telemetry
    youtubeTelemetry.isRunning = false;
    youtubeTelemetry.isPaused = false;
    youtubeTelemetry.step = 0;
    youtubeTelemetry.lastFrame = 0;
    youtubeTelemetry.bufferCount = 0;
    youtubeTelemetry.btnLoss = 0;
    youtubeTelemetry.camLoss = 0;
    youtubeTelemetry.totalLoss = 0;
    youtubeTelemetry.speed = 0;
    youtubeTelemetry.videoPercent = 0;
    youtubeTelemetry.currentTimeStr = '00:00';
    youtubeTelemetry.totalTimeStr = '00:00';
    youtubeTelemetry.currentSeconds = 0;
    youtubeTelemetry.videoEta = '--:--';
    youtubeTelemetry.setProgress = 0;
    youtubeTelemetry.setEta = '--:--';
    youtubeTelemetry.checkpoint = null;
    youtubeTelemetry.resumedFrom = null;

    // 6. Reset general telemetry
    latestTelemetry.botOnline = false;
    latestTelemetry.brainGen = 0;
    latestTelemetry.hiddenNeurons = brain.net.H;
    latestTelemetry.recentLogs.push('[PURGE] All learned models, checkpoints, weights, and histories wiped. System reset to Generation 0 with pristine random weights.');
    broadcastTelemetry();

    return {
      success: true,
      message: 'All learned neural models, checkpoints, and histories completely wiped. System reset to Generation 0 with clean random weights.',
      brainGen: 0,
      hiddenNeurons: brain.net.H
    };
  }

  if ((pathname === '/api/bot/reset_brain' || pathname === '/api/purge_all_learned') && req.method === 'POST') {
    try {
      const result = performCompletePurge();
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(result));
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

  // 8e. Video Stream Bot API: Add YouTube URL or Local File Path to Queue
  if (pathname === '/api/youtube/queue' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body || '{}');
        const urlStr = (data.url || data.path || '').trim().replace(/^['"]|['"]$/g, '');
        if (!urlStr || !isVideoSource(urlStr)) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Valid YouTube URL or local video file path (.mp4, .mkv, .webm) required' }));
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

        latestTelemetry.recentLogs.push(`[VIDEO QUEUE] Enqueued: ${path.basename(urlStr)}`);
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

  // 8e-2. Upload Local Video File Directly to Queue
  if (pathname === '/api/video/upload' && req.method === 'POST') {
    const rawFilename = req.headers['x-filename'] || `local_${Date.now()}.mp4`;
    const safeFilename = path.basename(decodeURIComponent(rawFilename)).replace(/[^a-zA-Z0-9_\-\.]/g, '_');
    const destPath = path.join(LOCAL_VIDEOS_DIR, safeFilename);

    const writeStream = fs.createWriteStream(destPath);
    req.pipe(writeStream);

    writeStream.on('finish', () => {
      if (!youtubeQueue.includes(destPath)) {
        youtubeQueue.push(destPath);
        fs.appendFileSync(LINKS_FILE, '\n' + destPath);
      }
      if (activeYoutubeProcess && activeYoutubeProcess.stdin && !activeYoutubeProcess.stdin.destroyed) {
        activeYoutubeProcess.stdin.write(destPath + '\n');
      }
      latestTelemetry.recentLogs.push(`[LOCAL VIDEO] Enqueued uploaded video: ${safeFilename}`);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, path: destPath, queue: youtubeQueue }));
    });

    writeStream.on('error', (err) => {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    });
    return;
  }

  // 8e-3. Local Video HTTP Streaming (Partial Content 206 for Browser Video Player)
  if (pathname === '/api/video/stream' && req.method === 'GET') {
    const filePath = req.url.split('?file=')[1] ? decodeURIComponent(req.url.split('?file=')[1]) : null;
    if (!filePath || !fs.existsSync(filePath)) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Local video not found');
      return;
    }
    const stat = fs.statSync(filePath);
    const fileSize = stat.size;
    const range = req.headers.range;

    if (range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunksize = (end - start) + 1;
      const file = fs.createReadStream(filePath, { start, end });
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunksize,
        'Content-Type': 'video/mp4'
      });
      file.pipe(res);
    } else {
      res.writeHead(200, {
        'Content-Length': fileSize,
        'Content-Type': 'video/mp4'
      });
      fs.createReadStream(filePath).pipe(res);
    }
    return;
  }

  // 8f. YouTube / Local Video API: Remove Video from Queue
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

    if (!youtubeQueue || youtubeQueue.length === 0) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Queue is empty. Please enqueue a YouTube video URL first before starting learning.' }));
      return;
    }

    try {
      latestTelemetry.recentLogs.push('[YOUTUBE BOT] Spawning Python In-Memory Stream & Imitation Pipeline...');
      youtubeTelemetry.isRunning = true;

      activeYoutubeProcess = spawn('python', ['-u', 'train_policy.py', '--links-file', 'links.txt', '--steps', '0'], {
        cwd: YOUTUBE_DIR
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

          if (line.includes('[PIPELINE PAUSED]')) {
            youtubeTelemetry.isPaused = true;
          }
          if (line.includes('[PIPELINE RESUMED]')) {
            youtubeTelemetry.isPaused = false;
          }
          if (line.includes('CHECKPOINT SAVED]')) {
            loadCheckpointMeta();
          }
          if (line.includes('[RESUMING VIDEO]')) {
            const resMatch = line.match(/from ([\d\.]+)s/);
            if (resMatch) youtubeTelemetry.resumedFrom = parseFloat(resMatch[1]);
          }

          if (line.includes('---> Streaming from:')) {
            const parts = line.split('---> Streaming from:');
            if (parts[1]) youtubeTelemetry.currentVideo = parts[1].trim();
          }

          if (line.includes('[STREAMING PROGRESS]')) {
            const frameMatch = line.match(/Frame #(\d+)\/(\d+) \((\d+\.?\d*)%\)/);
            const timeMatch = line.match(/Time: ([^\s]+)\/([^\s]+) \((\d+\.?\d*)s\/(\d+\.?\d*)s\)/);
            const speedMatch = line.match(/Speed: ([\d\.]+)fps \(([\d\.]+)x\)/);
            const vEtaMatch = line.match(/VideoETA: ([^\s]+) \((\d+)s\)/);
            const sProgMatch = line.match(/SetProgress: ([\d\.]+)%/);
            const sEtaMatch = line.match(/SetETA: ([^\s]+) \((\d+)s\)/);
            const qMatch = line.match(/Queue: (\d+)\/(\d+)/);
            const keysMatch = line.match(/Keys: \[(.*?)\]/);
            const mouseMatch = line.match(/Mouse: \(dx:([+-]?[\d\.]+),\s*dy:([+-]?[\d\.]+)\)/);
            const bufMatch = line.match(/Buffer: (\d+)\/(\d+)/);
            const titleMatch = line.match(/Title: (.*)$/);

            if (frameMatch) {
              youtubeTelemetry.lastFrame = parseInt(frameMatch[1], 10);
              youtubeTelemetry.totalFrames = parseInt(frameMatch[2], 10);
              youtubeTelemetry.videoPercent = parseFloat(frameMatch[3]);
            }
            if (timeMatch) {
              youtubeTelemetry.currentTimeStr = timeMatch[1];
              youtubeTelemetry.totalTimeStr = timeMatch[2];
              youtubeTelemetry.currentSeconds = parseFloat(timeMatch[3]);
              youtubeTelemetry.totalSeconds = parseFloat(timeMatch[4]);
            }
            if (speedMatch) {
              youtubeTelemetry.speedFps = parseFloat(speedMatch[1]);
              youtubeTelemetry.speedFactor = parseFloat(speedMatch[2]);
            }
            if (vEtaMatch) {
              youtubeTelemetry.videoEta = vEtaMatch[1];
              youtubeTelemetry.videoEtaSeconds = parseInt(vEtaMatch[2], 10);
            }
            if (sProgMatch) {
              youtubeTelemetry.setProgress = parseFloat(sProgMatch[1]);
            }
            if (sEtaMatch) {
              youtubeTelemetry.setEta = sEtaMatch[1];
              youtubeTelemetry.setEtaSeconds = parseInt(sEtaMatch[2], 10);
            }
            if (qMatch) {
              youtubeTelemetry.queueIndex = parseInt(qMatch[1], 10);
              youtubeTelemetry.queueTotal = parseInt(qMatch[2], 10);
            }
            if (keysMatch) youtubeTelemetry.lastKeys = keysMatch[1];
            if (mouseMatch) youtubeTelemetry.lastMouse = { dx: mouseMatch[1], dy: mouseMatch[2] };
            if (bufMatch) youtubeTelemetry.bufferCount = parseInt(bufMatch[1], 10);
            if (titleMatch) youtubeTelemetry.videoTitle = titleMatch[1].trim();
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
            const stepMatch = line.match(/Step \[(\d+)\/(\d+|CONT)\]/);
            const lossMatch = line.match(/Total Loss: ([\d\.]+)/);
            const btnMatch = line.match(/Btn Loss: ([\d\.]+)/);
            const camMatch = line.match(/Cam Loss: ([\d\.]+)/);
            const bufMatch = line.match(/Buffer: (\d+)/);
            const spdMatch = line.match(/Speed: ([\d\.]+)/);

            if (stepMatch) {
              youtubeTelemetry.step = parseInt(stepMatch[1], 10);
              youtubeTelemetry.totalSteps = (stepMatch[2] === 'CONT') ? '∞' : parseInt(stepMatch[2], 10);
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

  // 8h. YouTube In-Memory Stream Bot API: Stop Pipeline (Gracefully Saves Exact Checkpoint)
  if (pathname === '/api/youtube/stop' && req.method === 'POST') {
    if (activeYoutubeProcess) {
      try {
        if (activeYoutubeProcess.stdin && !activeYoutubeProcess.stdin.destroyed) {
          activeYoutubeProcess.stdin.write('STOP\n');
        }
      } catch (e) {}

      const procToKill = activeYoutubeProcess;
      activeYoutubeProcess = null;
      youtubeTelemetry.isRunning = false;
      youtubeTelemetry.isPaused = false;
      latestTelemetry.recentLogs.push('[YOUTUBE BOT] Graceful stop requested. Saving checkpoint...');

      setTimeout(() => {
        try {
          if (procToKill && procToKill.pid) {
            if (process.platform === 'win32') {
              spawn('taskkill', ['/pid', procToKill.pid.toString(), '/T', '/F']);
            } else {
              procToKill.kill('SIGKILL');
            }
          }
        } catch (e) {}
      }, 1500);
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true, message: 'Pipeline stopping gracefully with checkpoint' }));
    return;
  }

  // 8i. Video Stream Learning API: Pause / Resume Pipeline
  if (pathname === '/api/youtube/pause' && req.method === 'POST') {
    if (!activeYoutubeProcess) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Pipeline is not running.' }));
      return;
    }
    const targetState = !youtubeTelemetry.isPaused;
    const cmd = targetState ? 'PAUSE\n' : 'RESUME\n';
    try {
      if (activeYoutubeProcess.stdin && !activeYoutubeProcess.stdin.destroyed) {
        activeYoutubeProcess.stdin.write(cmd);
        youtubeTelemetry.isPaused = targetState;
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, isPaused: youtubeTelemetry.isPaused }));
      } else {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Python stdin pipe unavailable.' }));
      }
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8j. Video Stream Learning API: Manual Checkpoint Placement
  if (pathname === '/api/youtube/checkpoint' && req.method === 'POST') {
    if (!activeYoutubeProcess) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Pipeline is not running. Cannot place checkpoint while idle.' }));
      return;
    }
    try {
      if (activeYoutubeProcess.stdin && !activeYoutubeProcess.stdin.destroyed) {
        activeYoutubeProcess.stdin.write('CHECKPOINT\n');
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true, message: 'Checkpoint triggered.' }));
      } else {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Python stdin pipe unavailable.' }));
      }
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8j2. Video Stream Learning API: Trigger Dynamic GRU Neurogenesis (+32 neurons)
  if (pathname === '/api/youtube/grow' && req.method === 'POST') {
    try {
      if (activeYoutubeProcess && activeYoutubeProcess.stdin && !activeYoutubeProcess.stdin.destroyed) {
        activeYoutubeProcess.stdin.write('GROW\n');
        latestTelemetry.recentLogs.push('✨ [NEUROGENESIS] Sent GROW command to training pipeline (+32 GRU neurons).');
      } else {
        const growPy = `import torch, json, os; from models import MinecraftPolicy; sd = torch.load('minecraft_policy.pth', weights_only=True); gru_dim = sd['gru.weight_hh_l0'].shape[1] if 'gru.weight_hh_l0' in sd else 256; p = MinecraftPolicy(gru_hidden_dim=gru_dim); p.load_state_dict(sd); new_gru = p.grow_gru_neurons(32); torch.save(p.state_dict(), 'minecraft_policy.pth'); meta = json.load(open('checkpoint_meta.json', 'r', encoding='utf-8')) if os.path.exists('checkpoint_meta.json') else {}; meta['gru_neurons'] = new_gru; meta['total_parameters'] = sum(x.numel() for x in p.parameters()); json.dump(meta, open('checkpoint_meta.json', 'w', encoding='utf-8'), indent=2); print(new_gru, meta['total_parameters'])`;
        execSync(`python -c "${growPy}"`, { cwd: YOUTUBE_DIR, timeout: 10000 });
        loadCheckpointMeta(1);
        latestTelemetry.recentLogs.push(`✨ [NEUROGENESIS] Policy expanded: GRU memory now has ${youtubeTelemetry.checkpoint?.gru_neurons || 288} recurrent neurons!`);
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, gruNeurons: youtubeTelemetry.checkpoint?.gru_neurons || 288 }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 8k. Video Stream Learning API: Get / Reset Checkpoint Metadata
  if (pathname === '/api/youtube/checkpoint' && req.method === 'GET') {
    const cp = loadCheckpointMeta();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ checkpoint: cp }));
    return;
  }

  if (pathname === '/api/youtube/clear_checkpoint' && req.method === 'POST') {
    try {
      if (fs.existsSync(CHECKPOINT_META_PATH)) {
        fs.unlinkSync(CHECKPOINT_META_PATH);
      }
      youtubeTelemetry.checkpoint = null;
      youtubeTelemetry.resumedFrom = null;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: true, message: 'Checkpoint cleared.' }));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
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
