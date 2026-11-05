# 🧟 Minecraft Neuro-Evolutionary Survival Bot & Minecraft UI Terminal

A Minecraft Survival Mode AI Agent featuring **Behavioral Cloning through Player Observation**, **Plastic Neurogenesis (Neuron Forming & Pruning)**, **Explainable Decision Making**, and an authentic **Minecraft UI Web Terminal** with real-time HUD (Hearts, Hunger, XP), 3D first-person game visuals, and **Stuck / Stopped Reason Diagnostics**.

---

## 🚀 Quick Start

### 1. Start the Minecraft UI Web Terminal
```bash
npm start
```
Then open your browser at:
👉 **[http://localhost:3000](http://localhost:3000)**

---

## 🎮 How to Teach the Bot (Observation ➔ Neurogenesis ➔ Testing)

The bot learns by watching you play, extracting spatial facts and tactical decisions, reasoning about neuron utility, and growing new neural pathways.

### Step 1: Teach the Bot (Watch Me Play)
1. Join your Minecraft server with your normal Minecraft client.
2. In the Web Terminal under **1. TEACH BOT**:
   - Enter your Minecraft username in **PLAYER NAME** (e.g. `Karthik`).
   - Select your task (e.g., `survive`, `kill_skeleton`, `kill_zombie`, `chase_player`).
   - Click **🎥 WATCH ME PLAY**.
3. An `Observer` bot will join in spectator mode and follow your actions.
4. Play normally: run, jump over blocks, swing weapons, attack enemies.
5. The terminal streams your live samples in real time.

### Step 2: Stop Recording & Evolve Neurons
1. When you finish playing, click **⏹️ STOP & EVOLVE**.
2. Observation stops immediately, and the **Neurogenesis Engine** begins:
   - **Evaluates Current Neurons**: Ablation scores every hidden neuron against your demonstrations.
   - **Prunes Bad Pathways**: Neurons that conflicted with your actions are eliminated.
   - **Sprouts New Neurons**: Generates `+4` fresh plastic neurons into the hidden layer.
   - **Trains the Evolved Brain**: Optimizes the new generation on your play style.
   - **Saves & Archives**: Model A is safely archived (`gen_X_A.json`), and the new network becomes the active brain.

### Step 3: Test the Bot in Survival Mode
1. Click **⚔️ SPAWN IN SURVIVAL**.
2. The bot spawns into the world in survival mode with its newly formed neural network.
3. Observe live in the Web Terminal:
   - **10 Health Hearts**: Dynamic full/half/empty hearts.
   - **10 Food Drumsticks**: Dynamic hunger monitoring.
   - **When He Stops & Why He Stopped**: Real-time diagnostic alert banner explaining why the bot stopped (e.g. `Blocked by 2-block obstacle`, `Consuming food to restore HP`, `Searching for targets`, `In melee combat`).
   - **3D First-Person View & Screenshots**: Watch the bot's perspective in 3D or capture instant snapshots.
   - **Neuron Reasoning**: Inspect which neurons fired and why the bot chose an action.

### Step 4: Continual Retraining (Zero Interference)
- When you want to teach the bot new skills or refine existing ones, simply click **🎥 WATCH ME PLAY** again!
- The system appends the new demonstration, executes another round of neurogenesis, forms additional neurons, and keeps all previously learned foundational skills intact without crashing or resetting.

---

## 🛠️ System Architecture

| Component | File | Description |
| :--- | :--- | :--- |
| **Neural Brain** | [`brain.js`](file:///c:/Users/karth/Downloads/DL/approch%203/brain.js) | Dynamic growing neural network with neurogenesis, neuron ablation scoring, explainable AI, and generational evolution. |
| **Survival Bot** | [`bot.js`](file:///c:/Users/karth/Downloads/DL/approch%203/bot.js) | Full survival agent with movement execution, 3D Prismarine viewer stream, and combat reflexes. |
| **Survival Manager** | [`survival.js`](file:///c:/Users/karth/Downloads/DL/approch%203/survival.js) | Autonomous eating, auto-equipping weapons/armor, water bucket clutch, and stuck detection diagnostics. |
| **Environment Sensor** | [`features.js`](file:///c:/Users/karth/Downloads/DL/approch%203/features.js) | 18-feature spatial perception engine (distance, angles, walls, cliff drops, liquids, target types). |
| **Player Recorder** | [`record.js`](file:///c:/Users/karth/Downloads/DL/approch%203/record.js) | Spectator observer bot capturing human gameplay trajectories for behavioral cloning. |
| **Control Server** | [`server.js`](file:///c:/Users/karth/Downloads/DL/approch%203/server.js) | HTTP & Server-Sent Events (SSE) server bridging game state to the web UI at 10 Hz. |
| **Minecraft UI Terminal**| [`public/index.html`](file:///c:/Users/karth/Downloads/DL/approch%203/public/index.html) | Authentic Minecraft-themed web dashboard with hearts, hunger, XP bar, 3D screen, logs, and diagnostic indicators. |

---

## ⌨️ CLI Commands

You can also run all modules directly from your terminal:

```bash
# Launch the Web Terminal
npm start

# Run Survival Bot directly
node bot.js --host localhost --port 25565 --username SurvivalBot --task survive

# Record human player demonstrations
node record.js YourMinecraftName kill_skeleton

# Train & evolve brain from saved demos
node clone.js data_bot 35

# Fast headless simulator training (no Minecraft server required)
node sim_test.js 60
```
