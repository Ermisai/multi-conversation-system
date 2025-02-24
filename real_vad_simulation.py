import asyncio
import random
import string
import time  # <--- Adicionado
from typing import Dict, Any

import torch
import sounddevice as sd
import numpy as np

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

##############################################################################
#                    IMPLEMENTAÇÃO DO VAD REAL (SILERO)
##############################################################################

class VADReal:
    """
    Implementa a detecção de fala usando o modelo Silero e a captação do microfone.
    """
    def __init__(self, threshold=0.5, duration=0.5):
        self.threshold = threshold
        self.duration = duration  # duração do trecho a ser analisado (em segundos)
        self.sampling_rate = 16000
        self.vad = load_silero_vad()        

    def record_audio(self, duration, fs):
        # print("[VADReal] Capturando áudio do microfone...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        return np.squeeze(audio)

    async def has_speech(self) -> bool:
        # Captura áudio de forma assíncrona para não bloquear o loop principal
        audio = await asyncio.to_thread(self.record_audio, self.duration, self.sampling_rate)
        audio_tensor = torch.tensor(audio).float()
        # speech_timestamps = self.get_speech_ts(audio_tensor, self.sampling_rate, threshold=self.threshold)
        # detected = len(speech_timestamps) > 0
        speech_timestamps = get_speech_timestamps(
          audio_tensor, 
          self.vad,
          return_seconds=True,  # Return speech timestamps in seconds (default is samples)
        )
        detected = len(speech_timestamps) > 0        
        if detected:
          print(f"[VADReal] Fala detectada")
        return detected

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

        start_time = time.time()
        text = await stt_module.transcribe(task.data)
        elapsed = time.time() - start_time

        orchestrator.update_status(task.conversation_id,
            f"STT -> Texto: {text} (time: {elapsed:.2f}s)")
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

        start_time = time.time()
        reply_text = await logic_module.process(task.data)
        elapsed = time.time() - start_time

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

        total_duration = random.uniform(2, 4)
        increments = 10
        step = total_duration / increments

        orchestrator.update_status(task.conversation_id,
            f"Iniciando TTS (~{total_duration:.1f}s)")

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
            tts_elapsed = time.time() - tts_start_time
            orchestrator.update_status(task.conversation_id,
                f"TTS finalizado: {task.data} (time: {tts_elapsed:.2f}s)")
            orchestrator.record_pipeline_step(
                task.conversation_id,
                task.turn_id,
                "TTS",
                f"{task.data} (time: {tts_elapsed:.2f}s)"
            )
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
        # Usa o VAD real (Silero) com duração menor para detecção mais ágil
        self.vad = VADReal(duration=0.5)

    async def run(self):
        while self._running:
            if not self.pipeline_in_progress:
                # Verifica se há fala para iniciar um novo turno
                if await self.vad.has_speech():
                    await self.start_new_turn()
            else:
                # Monitora continuamente se nova fala ocorre para interromper o turno atual
                if await self.vad.has_speech():
                    print("[Conversation] Nova fala detectada durante pipeline. Interrompendo...")
                    self.interrupt()
                await asyncio.sleep(0.2)
        print(f"[Conversation {self.id}] Encerrada.")

    async def start_new_turn(self):
        self.current_turn_id += 1
        turn_id = self.current_turn_id
        self.pipeline_in_progress = True

        self.orchestrator.init_turn_history(self.id, turn_id)
        self.orchestrator.update_stage(self.id, "VAD")
        self.orchestrator.update_status(self.id, f"Nova fala detectada (turn {turn_id})")
        self.orchestrator.record_pipeline_step(self.id, turn_id, "VAD", "")

        await asyncio.sleep(1.0)  # Delay antes de iniciar o STT

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
                f"Interrupção automática (turn {old_turn_id}) por nova fala")
            self.orchestrator.update_stage(self.id, "VAD")
            self.pipeline_in_progress = False
            # Inicia automaticamente um novo turno
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
#                           MAIN DA SIMULAÇÃO
##############################################################################

async def main():
    orchestrator = Orchestrator()
    await orchestrator.start_workers()
    # Inicia automaticamente a conversa única "conv1"
    await orchestrator.start_conversation("conv1")
    print("Simulação iniciada. Pressione Ctrl+C para encerrar.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Encerrando simulação...")
        await orchestrator.stop_all_conversations()
        await orchestrator.wait_for_workers()

if __name__ == "__main__":
    import sounddevice as sd
    print(sd.query_devices())
    # sd.default.device = (input_index, output_index)  # substitua pelos índices corretos
    asyncio.run(main())
