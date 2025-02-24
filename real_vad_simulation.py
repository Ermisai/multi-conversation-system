import asyncio
import random
import string
import time
from typing import Dict, Any
import os

import torch
import sounddevice as sd
import numpy as np

from silero_vad import load_silero_vad, get_speech_timestamps

##############################################################################
#                    IMPLEMENTAÇÃO DO VAD REAL (SILERO)
##############################################################################

class VADReal:
    """
    Detecta fala usando o modelo Silero e capta áudio do microfone.
    Grava continuamente pequenos trechos até que, após o início da fala,
    seja detectado um período de silêncio sustentado que sinalize o fim da frase.
    """
    def __init__(self, threshold=0.25, segment_duration=0.5):
        self.threshold = threshold
        self.segment_duration = segment_duration  # duração de cada trecho em segundos
        self.sampling_rate = 16000
        self.vad = load_silero_vad()

    def record_audio(self, duration, fs):
        # Captura um trecho de áudio do microfone
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        return np.squeeze(audio)

    def segment_has_speech(self, audio_segment) -> bool:
        # Verifica se um trecho contém fala usando get_speech_timestamps
        audio_tensor = torch.tensor(audio_segment).float()
        timestamps = get_speech_timestamps(
            audio_tensor,
            self.vad,
            return_seconds=True,
            threshold=self.threshold
        )
        return len(timestamps) > 0

    async def wait_for_initial_speech(self):
        """
        Aguarda até que um segmento contenha fala.
        Enquanto não houver fala, os segmentos são ignorados.
        """
        while True:
            segment = await asyncio.to_thread(self.record_audio, self.segment_duration, self.sampling_rate)
            if self.segment_has_speech(segment):
                print("[VADReal] Fala iniciada!")
                return
            await asyncio.sleep(0.1)

    async def record_until_silence(self, min_duration=2.0, silence_required=1.0, max_duration=20.0):
        """
        Após o início da fala, grava segmentos consecutivos até que:
         - Ao menos min_duration segundos tenham sido gravados, e
         - Um número suficiente de segmentos silenciosos consecutivos seja detectado
           (suficiente para totalizar silence_required segundos de silêncio).
        Retorna o áudio acumulado (concatenando os segmentos, removendo os segmentos silenciosos finais).
        """
        segments = []
        total_time = 0.0
        silence_counter = 0
        required_silence_segments = int(silence_required / self.segment_duration)
        while total_time < max_duration:
            print(f"[VADReal] Gravando ({total_time:.1f}s)...", end=' ')
            segment = await asyncio.to_thread(self.record_audio, self.segment_duration, self.sampling_rate)
            segments.append(segment)
            total_time += self.segment_duration
            if not self.segment_has_speech(segment):
                silence_counter += 1
                print(f"Silêncio ({silence_counter}/{required_silence_segments})")
                if total_time >= min_duration and silence_counter >= required_silence_segments:
                    # Remove os segmentos silenciosos finais
                    segments = segments[:-silence_counter]
                    print(f"[VADReal] Fim da frase detectado após {total_time:.1f}s")
                    return np.concatenate(segments) if segments else None
            else:
                silence_counter = 0
                print(f"Fala detectada ({total_time:.1f}s)")
            await asyncio.sleep(0.05)
        print(f"[VADReal] Tempo máximo atingido ({max_duration}s)")
        return np.concatenate(segments) if segments else None

    async def has_speech(self) -> bool:
        # Método rápido para verificar se um segmento contém fala
        audio = await asyncio.to_thread(self.record_audio, self.segment_duration, self.sampling_rate)
        return self.segment_has_speech(audio)

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
    async def transcribe(self, audio_data: Any) -> str:
        # Simula transcrição; exibe o comprimento do áudio recebido
        await asyncio.sleep(random.uniform(1, 2))
        return f"texto_transcrito(len={len(audio_data)})"

class LogicMock:
    async def process(self, text: str) -> str:
        await asyncio.sleep(random.uniform(1, 4))
        return f"Resposta para: {text}"

class TTSMock:
    async def synthesize(self, text: str):
        await asyncio.sleep(random.uniform(2, 4))
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
            f"STT -> {text} (time: {elapsed:.2f}s)")
        orchestrator.record_pipeline_step(
            task.conversation_id, task.turn_id, "STT", f"{text} (time: {elapsed:.2f}s)"
        )
        new_task = PipelineTask(task.conversation_id, task.turn_id, "Logic", text)
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
            f"Logic -> {reply_text} (time: {elapsed:.2f}s)")
        orchestrator.record_pipeline_step(
            task.conversation_id, task.turn_id, "Logic", f"{reply_text} (time: {elapsed:.2f}s)"
        )
        new_task = PipelineTask(task.conversation_id, task.turn_id, "TTS", reply_text)
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
                task.conversation_id, task.turn_id, "TTS", f"{task.data} (time: {tts_elapsed:.2f}s)"
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
        self.vad = VADReal(segment_duration=0.5)

    async def run(self):
        while self._running:
            if self.pipeline_in_progress:
                # Enquanto o pipeline estiver em andamento, monitora nova fala
                if await self.vad.has_speech():
                    print("[Conversation] Nova fala detectada durante pipeline. Interrompendo...")
                    await self.interrupt()
            else:
                print("[Conversation] Aguardando início da fala...")
                await self.vad.wait_for_initial_speech()
                print("[Conversation] Iniciando gravação completa da frase...")
                audio_data = await self.vad.record_until_silence(min_duration=2.0, silence_required=1.0)
                if audio_data is not None and len(audio_data) > 0:
                    await self.start_new_turn(audio_data)
            await asyncio.sleep(0.1)
        print(f"[Conversation {self.id}] Encerrada.")

    async def start_new_turn(self, audio_data):
        self.current_turn_id += 1
        turn_id = self.current_turn_id
        self.pipeline_in_progress = True

        self.orchestrator.init_turn_history(self.id, turn_id)
        self.orchestrator.update_stage(self.id, "STT")
        self.orchestrator.update_status(self.id, f"Frase completa recebida (turn {turn_id})")
        self.orchestrator.record_pipeline_step(
            self.id, turn_id, "VAD", f"Áudio recebido (len={len(audio_data)})"
        )
        stt_task = PipelineTask(self.id, turn_id, "STT", audio_data)
        asyncio.create_task(self.orchestrator.stt_queue.put(stt_task))

    async def interrupt(self):
        if self.pipeline_in_progress:
            old_turn_id = self.current_turn_id
            self.orchestrator.set_turn_status(self.id, old_turn_id, "INTERRUPTED")
            self.orchestrator.update_status(self.id,
                f"Interrupção automática (turn {old_turn_id}) por nova fala")
            self.orchestrator.update_stage(self.id, "VAD")
            self.pipeline_in_progress = False
            # Inicia a gravação de um novo turn imediatamente
            print("[Conversation] Gravando nova frase após interrupção...")
            await self.vad.wait_for_initial_speech()
            audio_data = await self.vad.record_until_silence(min_duration=2.0, silence_required=1.0)
            if audio_data is not None and len(audio_data) > 0:
                await self.start_new_turn(audio_data)

    def pipeline_finished(self):
        self.pipeline_in_progress = False

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
        self.turns_history[conversation_id][turn_id] = {"status": "IN_PROGRESS", "steps": []}

    def record_pipeline_step(self, conversation_id: str, turn_id: int, stage: str, data: str):
        if conversation_id not in self.turns_history:
            self.turns_history[conversation_id] = {}
        if turn_id not in self.turns_history[conversation_id]:
            self.init_turn_history(conversation_id, turn_id)
        self.turns_history[conversation_id][turn_id]["steps"].append({"stage": stage, "data": data})

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

    def get_active_module(self) -> str:
        conv = self.conversations.get("conv1")
        if conv:
            return self.status_data.get("conv1", {}).get("stage", "Idle")
        return "Idle"

##############################################################################
#                         MONITORAMENTO EM TEMPO REAL
##############################################################################

async def monitor_status(orchestrator: Orchestrator):
    while True:
        active = orchestrator.get_active_module()
        if active in ["Idle", "INTERRUPTED"]:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=== STATUS EM TEMPO REAL ===")
            print(f"Conversa: conv1  |  Módulo ativo: {active}")
            print("============================\n")
        for cid, status in orchestrator.status_data.items():
            print(f"{cid}: {status}")
        print("\n")
        await asyncio.sleep(1)

##############################################################################
#                           MAIN DA SIMULAÇÃO
##############################################################################

async def main(monitor=False):
    orchestrator = Orchestrator()
    await orchestrator.start_workers()
    await orchestrator.start_conversation("conv1")
    if monitor:
        asyncio.create_task(monitor_status(orchestrator))
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
    import sys
    print(sd.query_devices())
    if 'monitor' in sys.argv:
        asyncio.run(main(monitor=True))
    else:
        asyncio.run(main())
