"""
ui_server.py - Standalone Python Web UI for YouTube Stream Learning Pipeline.
Runs on http://localhost:5000 (No extra dependencies: uses Python standard library http.server).
"""
import os
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import config
from queue_manager import LinkQueueManager
from train_policy import StreamingTrainer

PORT = 5000
trainer_instance = None
trainer_thread = None


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Minecraft In-Memory YouTube Stream Learner</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0d1117; color: #c9d1d9; font-family: monospace; }
    .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
  </style>
</head>
<body class="p-6 max-w-4xl mx-auto space-y-6">
  <div class="card p-6 border-l-4 border-l-red-500 flex justify-between items-center">
    <div>
      <h1 class="text-xl font-bold text-white flex items-center gap-2">
        <span>📺</span> MINECRAFT YOUTUBE STREAM LEARNER
      </h1>
      <p class="text-xs text-slate-400 mt-1">Direct HTTPS In-Memory RAM Stream • 0 Disk Writes • IDM Pseudo-Labeling</p>
    </div>
    <span id="statusBadge" class="px-3 py-1 bg-red-900/60 border border-red-700 text-red-300 text-xs rounded font-bold">IDLE</span>
  </div>

  <div class="card p-6 space-y-4">
    <label class="block text-sm text-slate-300 font-bold">PASTE YOUTUBE GAMEPLAY URL:</label>
    <div class="flex gap-2">
      <input id="urlInput" type="text" placeholder="https://www.youtube.com/watch?v=..." class="flex-1 bg-black border border-slate-700 p-2.5 rounded text-sm text-amber-300 outline-none focus:border-red-500">
      <button onclick="addUrl()" class="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded text-sm">
        ➕ Add to Queue
      </button>
    </div>
    <div class="flex gap-2">
      <button onclick="preset('https://www.youtube.com/watch?v=dQw4w9WgXcQ')" class="text-xs px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-300">
        ⚡ Preset 1
      </button>
      <button onclick="preset('https://www.youtube.com/watch?v=fJ9rUzIMcZQ')" class="text-xs px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-slate-300">
        ⚡ Preset 2
      </button>
    </div>
  </div>

  <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
    <div class="card p-4 text-center">
      <span class="text-xs text-slate-400 block">RAM REPLAY BUFFER</span>
      <span id="bufCount" class="text-xl font-bold text-amber-400 mt-1 block">0 / 50,000</span>
      <div class="w-full bg-slate-800 h-2 mt-2 rounded overflow-hidden">
        <div id="bufBar" class="bg-amber-500 h-full w-0 transition-all duration-300"></div>
      </div>
    </div>
    <div class="card p-4 text-center">
      <span class="text-xs text-slate-400 block">TRAINING STEP</span>
      <span id="stepCount" class="text-xl font-bold text-cyan-400 mt-1 block">0 / 3000</span>
      <span class="text-[10px] text-slate-500 block mt-1">GPU Accelerated</span>
    </div>
    <div class="card p-4 text-center">
      <span class="text-xs text-slate-400 block">DISK USAGE</span>
      <span class="text-xl font-bold text-emerald-400 mt-1 block">0 BYTES</span>
      <span class="text-[10px] text-slate-500 block mt-1">Direct FFmpeg Pipe:1 to RAM</span>
    </div>
  </div>

  <div class="card p-6 space-y-4">
    <div class="flex justify-between items-center border-b border-slate-800 pb-2">
      <span class="text-sm font-bold text-slate-300">ENQUEUED YOUTUBE VIDEOS</span>
      <span id="qCount" class="text-xs text-slate-400">0 Links</span>
    </div>
    <div id="qList" class="space-y-2 text-xs max-h-40 overflow-y-auto">
      <div class="text-slate-500 italic">No YouTube links enqueued. Paste a URL above to start!</div>
    </div>
    <div class="flex gap-4 pt-2">
      <button id="btnStart" onclick="startPipeline()" class="flex-1 py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded text-sm">
        ▶ Start Streaming &amp; Policy Training
      </button>
      <button id="btnStop" onclick="stopPipeline()" class="flex-1 py-3 bg-red-600 hover:bg-red-500 text-white font-bold rounded text-sm opacity-50" disabled>
        ⏹ Stop Pipeline
      </button>
    </div>
  </div>

  <script>
    function preset(url) { document.getElementById('urlInput').value = url; }

    async function addUrl() {
      const url = document.getElementById('urlInput').value.trim();
      if (!url) return alert('Enter URL');
      await fetch('/api/queue', { method: 'POST', body: JSON.stringify({ url }) });
      document.getElementById('urlInput').value = '';
      loadStatus();
    }

    async function startPipeline() {
      await fetch('/api/start', { method: 'POST' });
      loadStatus();
    }

    async function stopPipeline() {
      await fetch('/api/stop', { method: 'POST' });
      loadStatus();
    }

    async function loadStatus() {
      try {
        const res = await fetch('/api/status');
        const d = await res.json();
        document.getElementById('statusBadge').innerText = d.isRunning ? 'STREAMING IN RAM' : 'IDLE';
        document.getElementById('statusBadge').className = d.isRunning ?
          'px-3 py-1 bg-green-900/60 border border-green-700 text-green-300 text-xs rounded font-bold' :
          'px-3 py-1 bg-red-900/60 border border-red-700 text-red-300 text-xs rounded font-bold';

        document.getElementById('btnStart').disabled = d.isRunning;
        document.getElementById('btnStart').className = d.isRunning ?
          'flex-1 py-3 bg-slate-700 text-slate-500 font-bold rounded text-sm cursor-not-allowed' :
          'flex-1 py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded text-sm';

        document.getElementById('btnStop').disabled = !d.isRunning;
        document.getElementById('btnStop').className = !d.isRunning ?
          'flex-1 py-3 bg-slate-700 text-slate-500 font-bold rounded text-sm cursor-not-allowed' :
          'flex-1 py-3 bg-red-600 hover:bg-red-500 text-white font-bold rounded text-sm';

        document.getElementById('bufCount').innerText = `${d.bufferCount.toLocaleString()} / 50,000`;
        document.getElementById('bufBar').style.width = `${Math.min(100, (d.bufferCount / 50000) * 100)}%`;

        document.getElementById('qCount').innerText = `${d.queue.length} Links`;
        if (d.queue.length) {
          document.getElementById('qList').innerHTML = d.queue.map(u => `
            <div class="bg-black/50 p-2 rounded border border-slate-800 text-amber-300 flex justify-between">
              <span class="truncate">${u}</span>
              <span class="text-emerald-400 font-bold">READY</span>
            </div>
          `).join('');
        } else {
          document.getElementById('qList').innerHTML = '<div class="text-slate-500 italic">No YouTube links enqueued. Paste a URL above!</div>';
        }
      } catch (e) {}
    }

    loadStatus();
    setInterval(loadStatus, 2000);
  </script>
</body>
</html>
"""

class StandaloneUIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
        elif self.path == "/api/status":
            global trainer_instance
            queue_urls = []
            links_path = os.path.join(os.path.dirname(__file__), "links.txt")
            if os.path.exists(links_path):
                with open(links_path, "r") as f:
                    queue_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

            buf_count = len(trainer_instance.replay_buffer) if trainer_instance else 0
            is_running = trainer_thread is not None and trainer_thread.is_alive()

            payload = {
                "isRunning": is_running,
                "bufferCount": buf_count,
                "queue": queue_urls
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global trainer_instance, trainer_thread
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        data = json.loads(body)

        if self.path == "/api/queue":
            url = data.get("url", "").strip()
            if url:
                links_path = os.path.join(os.path.dirname(__file__), "links.txt")
                with open(links_path, "a") as f:
                    f.write("\n" + url)
                if trainer_instance:
                    trainer_instance.queue_mgr.add_link(url)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"success": true}')

        elif self.path == "/api/start":
            if trainer_thread is None or not trainer_thread.is_alive():
                trainer_instance = StreamingTrainer()
                links_path = os.path.join(os.path.dirname(__file__), "links.txt")
                if os.path.exists(links_path):
                    with open(links_path, "r") as f:
                        for l in f:
                            trainer_instance.queue_mgr.add_link(l.strip())

                def run():
                    prod = threading.Thread(target=trainer_instance.stream_producer_loop, daemon=True)
                    prod.start()
                    trainer_instance.train_policy_loop(total_steps=3000)

                trainer_thread = threading.Thread(target=run, daemon=True)
                trainer_thread.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"success": true}')

        elif self.path == "/api/stop":
            if trainer_instance:
                trainer_instance.stop_signal.set()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"success": true}')


def run_server():
    server = HTTPServer(("0.0.0.0", PORT), StandaloneUIHandler)
    print(f"\n=======================================================")
    print(f"  📺 YOUTUBE IN-MEMORY STREAM LEARNER - WEB UI RUNNING ")
    print(f"  👉 Open in your browser: http://localhost:{PORT}")
    print(f"=======================================================\n")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
