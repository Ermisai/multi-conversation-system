import asyncio
import random
import string
import time  # <--- Adicionado
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

##############################################################################
#                          MODELOS E MOCKS
##############################################################################

class PipelineTask:
    def __init__(self, conversation_id: str, turn_id: int, stage: str, data: Any):
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        self.stage = stage
        self.data = data

class STTMock:
    async def transcribe(self, audio_data: str) -> str:
        await asyncio.sleep(random.uniform(1,2))
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(5))

class LogicMock:
    async def process(self, text: str) -> str:
        await asyncio.sleep(random.uniform(1,4))
        return f"Resposta para: {text}"

class TTSMock:
    async def synthesize(self, text: str):
        # Aqui não precisamos medir tempo, pois já dividimos em incrementos.
        # Mas se quiser, podemos medir também. Veja no tts_worker.
        await asyncio.sleep(random.uniform(2,4))
        print(f"[TTS] Falando: {text}")

##############################################################################
#                          WORKERS (STT, LOGIC, TTS)
##############################################################################

async def stt_worker(stt_queue: asyncio.Queue,
                     logic_queue: asyncio.Queue,
                     stt_module: STTMock,
                     orchestrator: "Orchestrator"):
    while True:
        task: PipelineTask = await stt_queue.get()
        if not orchestrator.is_turn_valid(task.conversation_id, task.turn_id):
            stt_queue.task_done()
            continue

        orchestrator.update_stage(task.conversation_id, "STT")

        # --- Mede tempo ---
        start_time = time.time()
        text = await stt_module.transcribe(task.data)
        elapsed = time.time() - start_time
        # ------------------

        orchestrator.update_status(task.conversation_id,
            f"STT -> Texto: {text} (time: {elapsed:.2f}s)")

        # Guardamos no 'data' a info do texto + tempo
        orchestrator.record_pipeline_step(
            task.conversation_id,
            task.turn_id,
            "STT",
            f"{text} (time: {elapsed:.2f}s)"
        )

        new_task = PipelineTask(task.conversation_id, task.turn_id, "logic", text)
        await logic_queue.put(new_task)
        stt_queue.task_done()

async def logic_worker(logic_queue: asyncio.Queue,
                       tts_queue: asyncio.Queue,
                       logic_module: LogicMock,
                       orchestrator: "Orchestrator"):
    while True:
        task: PipelineTask = await logic_queue.get()
        if not orchestrator.is_turn_valid(task.conversation_id, task.turn_id):
            logic_queue.task_done()
            continue

        orchestrator.update_stage(task.conversation_id, "Logic")

        # --- Mede tempo ---
        start_time = time.time()
        reply_text = await logic_module.process(task.data)
        elapsed = time.time() - start_time
        # ------------------

        orchestrator.update_status(task.conversation_id,
            f"Lógica -> {reply_text} (time: {elapsed:.2f}s)")

        orchestrator.record_pipeline_step(
            task.conversation_id,
            task.turn_id,
            "Logic",
            f"{reply_text} (time: {elapsed:.2f}s)"
        )

        new_task = PipelineTask(task.conversation_id, task.turn_id, "tts", reply_text)
        await tts_queue.put(new_task)
        logic_queue.task_done()

async def tts_worker(tts_queue: asyncio.Queue,
                     tts_module: TTSMock,
                     orchestrator: "Orchestrator"):
    while True:
        task: PipelineTask = await tts_queue.get()
        if not orchestrator.is_turn_valid(task.conversation_id, task.turn_id):
            tts_queue.task_done()
            continue

        orchestrator.update_stage(task.conversation_id, "TTS")

        # Em vez de um único sleep, dividimos em incrementos
        total_duration = random.uniform(2, 4)
        increments = 10
        step = total_duration / increments

        orchestrator.update_status(task.conversation_id,
            f"Iniciando TTS (~{total_duration:.1f}s)")

        # --- Se quiser medir tempo total de TTS, comece aqui ---
        tts_start_time = time.time()

        interrupted = False
        for _ in range(increments):
            await asyncio.sleep(step)
            if not orchestrator.is_turn_valid(task.conversation_id, task.turn_id):
                interrupted = True
                break

        if interrupted:
            orchestrator.set_turn_status(task.conversation_id, task.turn_id, "INTERRUPTED")
            orchestrator.update_status(task.conversation_id, "TTS interrompido abruptamente!")
            orchestrator.update_stage(task.conversation_id, "VAD")
            conv = orchestrator.conversations.get(task.conversation_id)
            if conv:
                conv.pipeline_in_progress = False

        else:
            # Tempo total TTS
            tts_elapsed = time.time() - tts_start_time

            orchestrator.update_status(task.conversation_id,
                f"TTS finalizado: {task.data} (time: {tts_elapsed:.2f}s)")

            # Grava step TTS com tempo
            orchestrator.record_pipeline_step(
                task.conversation_id,
                task.turn_id,
                "TTS",
                f"{task.data} (time: {tts_elapsed:.2f}s)"
            )

            # Estado transitório "TTSUser"
            orchestrator.update_stage(task.conversation_id, "TTSUser")
            await asyncio.sleep(1.0)

            orchestrator.set_turn_status(task.conversation_id, task.turn_id, "COMPLETED")
            orchestrator.update_stage(task.conversation_id, "Idle")

            conv = orchestrator.conversations.get(task.conversation_id)
            if conv:
                conv.pipeline_finished()

        tts_queue.task_done()

##############################################################################
#                          CLASSE Conversation
##############################################################################

class Conversation:
    def __init__(self, conversation_id: str, orchestrator: "Orchestrator"):
        self.id = conversation_id
        self.orchestrator = orchestrator
        self._running = True
        self.current_turn_id = 0
        self.pipeline_in_progress = False

    async def run(self):
        while self._running:
            if not self.pipeline_in_progress:
                speech_detected = await self.simulate_vad()
                if not self._running:
                    break
                if speech_detected:
                    await self.start_new_turn()
            else:
                await asyncio.sleep(0.5)

        print(f"[Conversation {self.id}] Encerrada.")

    async def simulate_vad(self):
        await asyncio.sleep(random.uniform(2,4))
        if not self._running:
            return False
        return bool(random.getrandbits(1))

    async def start_new_turn(self):
        self.current_turn_id += 1
        turn_id = self.current_turn_id
        self.pipeline_in_progress = True

        self.orchestrator.init_turn_history(self.id, turn_id)
        self.orchestrator.update_stage(self.id, "VAD")
        self.orchestrator.update_status(self.id, f"Nova fala detectada (turn {turn_id})")
        self.orchestrator.record_pipeline_step(self.id, turn_id, "VAD", "")

        await asyncio.sleep(1.0)

        if self._running and self.pipeline_in_progress and self.current_turn_id == turn_id:
            audio_data = f"audio_chunk_{random.randint(1,100)}"
            stt_task = PipelineTask(self.id, turn_id, "stt", audio_data)
            asyncio.create_task(self.orchestrator.stt_queue.put(stt_task))

    def pipeline_finished(self):
        self.pipeline_in_progress = False

    def interrupt(self):
        if self.pipeline_in_progress:
            old_turn_id = self.current_turn_id
            self.orchestrator.set_turn_status(self.id, old_turn_id, "INTERRUPTED")
            self.orchestrator.update_status(self.id,
                f"Interrompida => invalidando turn {old_turn_id}")
            self.orchestrator.update_stage(self.id, "VAD")
            self.pipeline_in_progress = False

            asyncio.create_task(self.start_new_turn())

    def stop(self):
        self._running = False

##############################################################################
#                          ORCHESTRATOR
##############################################################################

class Orchestrator:
    def __init__(self):
        self.stt_queue = asyncio.Queue()
        self.logic_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()

        self.stt_module = STTMock()
        self.logic_module = LogicMock()
        self.tts_module = TTSMock()

        self.conversations: Dict[str, Conversation] = {}
        self.workers_tasks = []

        self.status_data: Dict[str, Dict[str, Any]] = {}
        self.turns_history: Dict[str, Dict[int, Dict[str, Any]]] = {}

    async def start_workers(self):
        stt_task = asyncio.create_task(stt_worker(self.stt_queue, self.logic_queue, self.stt_module, self))
        logic_task = asyncio.create_task(logic_worker(self.logic_queue, self.tts_queue, self.logic_module, self))
        tts_task = asyncio.create_task(tts_worker(self.tts_queue, self.tts_module, self))
        self.workers_tasks = [stt_task, logic_task, tts_task]

    def update_stage(self, conversation_id: str, stage: str):
        if conversation_id not in self.status_data:
            self.status_data[conversation_id] = {}
        self.status_data[conversation_id]["stage"] = stage

    def update_status(self, conversation_id: str, message: str):
        if conversation_id not in self.status_data:
            self.status_data[conversation_id] = {}
        self.status_data[conversation_id]["message"] = message

        conv = self.conversations.get(conversation_id)
        if conv:
            self.status_data[conversation_id]["turn_id"] = conv.current_turn_id

        print(f"[Status] {conversation_id}: {message}")

    def is_turn_valid(self, conversation_id: str, turn_id: int) -> bool:
        conv = self.conversations.get(conversation_id)
        if not conv or not conv._running:
            return False
        return turn_id == conv.current_turn_id

    def init_turn_history(self, conversation_id: str, turn_id: int):
        if conversation_id not in self.turns_history:
            self.turns_history[conversation_id] = {}
        self.turns_history[conversation_id][turn_id] = {
            "status": "IN_PROGRESS",
            "steps": []
        }

    def record_pipeline_step(self, conversation_id: str, turn_id: int, stage: str, data: str):
        if conversation_id not in self.turns_history:
            self.turns_history[conversation_id] = {}
        if turn_id not in self.turns_history[conversation_id]:
            self.init_turn_history(conversation_id, turn_id)

        self.turns_history[conversation_id][turn_id]["steps"].append({
            "stage": stage,
            "data": data
        })

    def set_turn_status(self, conversation_id: str, turn_id: int, status: str):
        if conversation_id not in self.turns_history:
            self.turns_history[conversation_id] = {}
        if turn_id not in self.turns_history[conversation_id]:
            self.init_turn_history(conversation_id, turn_id)
        self.turns_history[conversation_id][turn_id]["status"] = status

    async def start_conversation(self, conversation_id: str):
        if conversation_id in self.conversations:
            return
        conv = Conversation(conversation_id, self)
        self.conversations[conversation_id] = conv
        asyncio.create_task(conv.run())
        self.update_stage(conversation_id, "Idle")
        self.update_status(conversation_id, "Conversa iniciada")

    def interrupt_conversation(self, conversation_id: str):
        conv = self.conversations.get(conversation_id)
        if conv and conv._running:
            conv.interrupt()

    def stop_conversation(self, conversation_id: str):
        conv = self.conversations.get(conversation_id)
        if conv and conv._running:
            conv.stop()
            self.update_stage(conversation_id, "Stopped")
            self.update_status(conversation_id, "Conversa parada")
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        if conversation_id in self.status_data:
            del self.status_data[conversation_id]

    async def stop_all_conversations(self):
        for conv in list(self.conversations.values()):
            conv.stop()
        self.conversations.clear()
        self.status_data.clear()

    async def wait_for_workers(self):
        await self.stt_queue.join()
        await self.logic_queue.join()
        await self.tts_queue.join()
        for t in self.workers_tasks:
            t.cancel()
        await asyncio.gather(*self.workers_tasks, return_exceptions=True)

##############################################################################
#                    FASTAPI + INTERFACE (2 colunas 50/50)
##############################################################################

app = FastAPI()
orchestrator = Orchestrator()

@app.on_event("startup")
async def on_startup():
    await orchestrator.start_workers()

@app.on_event("shutdown")
async def on_shutdown():
    await orchestrator.stop_all_conversations()
    await orchestrator.wait_for_workers()

@app.get("/", response_class=HTMLResponse)
async def index():
    """
    - Estado "TTSUser" é adicionado após TTS. Seta TTS->User é destacada nesse estado.
    - Idle não destaca TTS->User.
    """
    html_content = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <title>Simulação: TTS->User sem Idle</title>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.24.0/dist/cytoscape.min.js"></script>
  <style>
    html, body {
      margin: 0; padding: 0;
      width: 100%; height: 100%;
      overflow: hidden;
      font-family: Arial, sans-serif;
    }
    #mainContainer {
      display: flex;
      width: 100%; height: 100%;
    }
    #leftSide, #rightSide {
      width: 50%;
      overflow: auto;
      display: flex;
      flex-direction: column;
    }
    #controls {
      padding: 10px;
      border-bottom: 1px solid #ccc;
      flex: 0 0 auto;
    }
    #cy {
      flex: 1;
      margin: 10px;
      border: 1px solid #ccc;
    }
    #statusPanel {
      flex: 0 0 auto;
      border-bottom: 1px solid #ccc;
      padding: 10px;
    }
    #turnsPanel {
      flex: 1 1 auto;
      padding: 10px;
      overflow-y: auto;
    }
    .activeNode {
      background-color: #ffeb3b;
      color: #000;
      border-width: 2px;
      border-color: #333;
    }
    .activeEdge {
      line-color: #ffeb3b;
      target-arrow-color: #ffeb3b;
      width: 4;
    }
    

    /* Classes de cor para cada estado de turn */
    .turn-inprogress {
      color: #2196f3; /* azul */
    }
    .turn-completed {
      color: #4caf50; /* verde */
    }
    .turn-interrupted {
      color: #f44336; /* vermelho */
    }
    /* Se quiser uma cor "padrão" para casos não mapeados */
    .turn-unknown {
      color: #999;
    }
    
    .conversation-status {
      margin-bottom: 10px;
    }
    .turn-history {
      margin-bottom: 5px;
    }
  </style>
</head>
<body>
<div id="mainContainer">
  <!-- Lado esquerdo -->
  <div id="leftSide">
    <div id="controls">
      <input type="text" id="conv_id" placeholder="Nome da conversa (ex: conv1)"/>
      <button onclick="startConversation()">Iniciar</button>
      <button onclick="interruptConversation()">Interromper</button>
      <button onclick="stopConversation()">Parar</button>
    </div>
    <div id="cy"></div>
  </div>

  <!-- Lado direito -->
  <div id="rightSide">
    <div id="statusPanel">
      <h3>Status das Conversas</h3>
      <div id="statusContainer"></div>
    </div>
    <div id="turnsPanel">
      <h3>Turns</h3>
      <div id="turnsContainer"></div>
    </div>
  </div>
</div>

<script>
  let cy = null;

  function initCytoscape() {
    cy = cytoscape({
      container: document.getElementById('cy'),
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#0074D9',
            'label': 'data(label)',
            'text-valign': 'center',
            'color': '#fff',
            'width': 'label',
            'height': 'label',
            'padding': '8px',
            'border-style': 'solid',
            'border-width': '1px',
            'border-color': '#555',
            'font-size': '12px'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 3,
            'line-color': '#999',
            'target-arrow-color': '#999',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '10px',
            'text-rotation': 'autorotate',
            'text-margin-y': -10
          }
        },
        {
          selector: '.activeNode',
          style: {
            'background-color': '#ffeb3b',
            'color': '#000',
            'border-width': 2,
            'border-color': '#333'
          }
        },
        {
          selector: '.activeEdge',
          style: {
            'line-color': '#ffeb3b',
            'target-arrow-color': '#ffeb3b',
            'width': 4
          }
        }
      ],
      layout: { name: 'grid' }
    });
  }

  // Adicionamos um caso "TTSUser" para destacar TTS->User
  function getEdgeInfo(stage) {
    switch(stage) {
      case "VAD":     return { edgeSuffix: "uv", nodeSuffix: "VAD"   };
      case "STT":     return { edgeSuffix: "vs", nodeSuffix: "STT"   };
      case "Logic":   return { edgeSuffix: "sl", nodeSuffix: "Logic" };
      case "TTS":     return { edgeSuffix: "lt", nodeSuffix: "TTS"   };
      case "TTSUser": return { edgeSuffix: "tu", nodeSuffix: "User"  }; // destaque TTS->User
      // Idle => sem destaque
      default:        return { edgeSuffix: null, nodeSuffix: null };
    }
  }

  async function fetchAllData() {
    const resp = await fetch("/status");
    return await resp.json();
  }

  function updateGraph(statusData) {
    cy.elements().remove();

    for (const [convId, info] of Object.entries(statusData)) {
      const userId  = `User_${convId}`;
      const vadId   = `VAD_${convId}`;
      const sttId   = `STT_${convId}`;
      const logicId = `Logic_${convId}`;
      const ttsId   = `TTS_${convId}`;

      const e_uv = `e_${convId}_uv`;
      const e_vs = `e_${convId}_vs`;
      const e_sl = `e_${convId}_sl`;
      const e_lt = `e_${convId}_lt`;
      const e_tu = `e_${convId}_tu`;

      cy.add([
        { data: { id: userId,  label: `User(${convId})`  } },
        { data: { id: vadId,   label: `VAD(${convId})`   } },
        { data: { id: sttId,   label: `STT(${convId})`   } },
        { data: { id: logicId, label: `Logic(${convId})` } },
        { data: { id: ttsId,   label: `TTS(${convId})`   } },
      ]);

      cy.add([
        { data: { id: e_uv, source: userId,  target: vadId   } },
        { data: { id: e_vs, source: vadId,   target: sttId   } },
        { data: { id: e_sl, source: sttId,   target: logicId } },
        { data: { id: e_lt, source: logicId, target: ttsId   } },
        { data: { id: e_tu, source: ttsId,   target: userId  } },
      ]);

      const stage = info.stage || "Idle";
      const turnId = info.turn_id || 0;
      const { edgeSuffix, nodeSuffix } = getEdgeInfo(stage);
      if (edgeSuffix && nodeSuffix) {
        const nodeId = `${nodeSuffix}_${convId}`;
        const nodeEle = cy.getElementById(nodeId);
        nodeEle.addClass("activeNode");

        const edgeId = `e_${convId}_${edgeSuffix}`;
        const edgeEle = cy.getElementById(edgeId);
        if (edgeEle) {
          edgeEle.addClass("activeEdge");
          edgeEle.data("label", `turn: ${convId}-${turnId}`);
        }
      }
    }

    cy.layout({ name: 'grid' }).run();
  }

  function updateStatusPanel(statusData) {
    const container = document.getElementById("statusContainer");
    container.innerHTML = "";

    for (const [convId, info] of Object.entries(statusData)) {
      const stage = info.stage || "Idle";
      const turnId = info.turn_id || 0;
      const message = info.message || "";

      const div = document.createElement("div");
      div.className = "conversation-status";
      div.innerHTML = `
        <strong>${convId}</strong><br/>
        Stage: ${stage}<br/>
        Turn: ${turnId}<br/>
        Message: ${message}
      `;
      container.appendChild(div);
    }
  }
  
  // Retorna a classe CSS de acordo com o status do turn
  function getStateClass(state) {
    switch(state) {
      case "IN_PROGRESS":  return "turn-inprogress";
      case "COMPLETED":    return "turn-completed";
      case "INTERRUPTED":  return "turn-interrupted";
      default:             return "turn-unknown";
    }
  }

  function buildTurnLine(convId, turnId, turnInfo) {
    const status = turnInfo.status;
    const steps = turnInfo.steps || [];
    let line = `(${convId})(turn ${turnId}) - `;
    line += steps.map(s => {
      if (s.data) return `${s.stage}(${s.data})`;
      return s.stage;
    }).join(" -> ");
    line += ` - ${status}`;
    return line;
  }

  function updateTurnsPanel(turnsData) {
    const container = document.getElementById("turnsContainer");
    container.innerHTML = "";

    for (const [convId, turns] of Object.entries(turnsData)) {
      const sortedTurnIds = Object.keys(turns).map(Number).sort((a,b) => a - b);
      sortedTurnIds.forEach(turnIdNum => {
        const turnInfo = turns[turnIdNum];
        const line = buildTurnLine(convId, turnIdNum, turnInfo);
        
        // Pega a classe CSS conforme status
        const cssClass = getStateClass(turnInfo.status || "unknown");

        const div = document.createElement("div");
        div.className = `turn-history ${cssClass}`;
        div.textContent = line;
        container.appendChild(div);
      });
    }
    container.scrollTop = container.scrollHeight;
  }

  async function refreshAll() {
    try {
      const data = await fetchAllData();
      updateGraph(data.status_data);
      updateStatusPanel(data.status_data);
      updateTurnsPanel(data.turns_history);
    } catch (err) {
      console.error("Erro ao atualizar:", err);
    }
  }

  function getConvId() {
    return document.getElementById("conv_id").value;
  }

  async function startConversation() {
    const convId = getConvId();
    if (!convId) return;
    await fetch("/start?conversation_id=" + convId);
    refreshAll();
  }

  async function interruptConversation() {
    const convId = getConvId();
    if (!convId) return;
    await fetch("/interrupt?conversation_id=" + convId);
    refreshAll();
  }

  async function stopConversation() {
    const convId = getConvId();
    if (!convId) return;
    await fetch("/stop?conversation_id=" + convId);
    refreshAll();
  }

  initCytoscape();
  setInterval(refreshAll, 100);
  refreshAll();
</script>
    """
    return HTMLResponse(html_content)

@app.get("/status")
async def get_status():
    return {
        "status_data": orchestrator.status_data,
        "turns_history": orchestrator.turns_history
    }

@app.get("/start")
async def start_conversation(conversation_id: str):
    asyncio.create_task(orchestrator.start_conversation(conversation_id))
    return {"status": f"Conversa {conversation_id} iniciada"}

@app.get("/interrupt")
async def interrupt_conversation(conversation_id: str):
    orchestrator.interrupt_conversation(conversation_id)
    return {"status": f"Conversa {conversation_id} interrompida"}

@app.get("/stop")
async def stop_conversation(conversation_id: str):
    orchestrator.stop_conversation(conversation_id)
    return {"status": f"Conversa {conversation_id} parada"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
