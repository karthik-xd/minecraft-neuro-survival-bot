'use strict';
// clone.js - trains the brain to copy your demonstrations with neurogenesis.
// Evaluates existing neurons, cuts detrimental ones, sprouts new neurons, and learns your tactics.
//
// CLI Usage:
//   node clone.js [brainDir] [epochs]
//   e.g.: node clone.js data_bot 35

const fs = require('fs');
const path = require('path');
const { Brain, ACTIONS } = require('./brain');

const dir = path.join(__dirname, process.argv[2] || 'data_bot');
const epochs = +(process.argv[3] || 35);
const demoDir = path.join(__dirname, 'demos');

function loadDemos() {
  const demos = [];
  if (!fs.existsSync(demoDir)) return demos;
  const files = fs.readdirSync(demoDir).filter(f => f.endsWith('.jsonl'));
  for (const f of files) {
    const lines = fs.readFileSync(path.join(demoDir, f), 'utf8').split('\n');
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

const demos = loadDemos();
console.log(`========================================================================`);
console.log(`🧠 NEURON FORMATION & BEHAVIORAL CLONING`);
console.log(`Loaded ${demos.length} demonstration samples from ${demoDir}`);

if (demos.length < 15) {
  console.log(`⚠️ Need at least 15 demonstration samples to evaluate neurons. Please play and record first!`);
  process.exit(1);
}

const actionSummary = ACTIONS.map((name, k) => `${name}: ${demos.filter(d => d.a === k).length}`).join(' | ');
console.log(`Action breakdown: ${actionSummary}`);
console.log(`------------------------------------------------------------------------`);

const brain = new Brain(dir);
console.log(`Current Brain: Generation ${brain.generation} with ${brain.net.H} hidden neurons.`);
console.log(`Scoring neuron utility, pruning dead pathways, and sprouting fresh plastic neurons...`);

const report = brain.evolveFromDemos(demos, { epochs });

console.log(`========================================================================`);
console.log(`✨ NEURON FORMING COMPLETE!`);
console.log(`- New Generation: ${report.generation}`);
console.log(`- Hidden Neurons: ${report.hiddenBefore} -> ${report.hiddenAfter} (Kept: ${report.kept}, Pruned: ${report.cut}, Sprouted: ${report.added})`);
console.log(`- Imitation Accuracy: ${report.accuracyBefore}% -> ${report.accuracyAfter}%`);
console.log(`- Brain file saved: ${brain.file}`);
console.log(`========================================================================`);
