import asyncio
import random
import string
import time

# TODO:
# [ ] Remover VADMock e iniciar o pipeline manualmente.
# [ ] Substituir STTMock, LogicMock, TTSMock por seus modelos reais.


##############################################################################
#                         MODELO DE DADOS DO PIPELINE
##############################################################################

class PipelineTask:
    """
    Representa uma tarefa em uma das etapas do pipeline.
    stage pode ser: 'stt', 'logic', 'tts'
    data pode conter áudio, texto, etc., dependendo da etapa.
    """
    def __init__(self, conversation_id, job_id, stage, data):
        self.conversation_id = conversation_id
        self.job_id = job_id
        self.stage = stage
        self.data = data

##############################################################################
#                       MÓDULOS SIMULADOS (VAD, STT, LOGIC, TTS)
##############################################################################

class VADMock:
    """Simula a detecção de fala."""
    async def has_speech(self):
        # Espera um pouco e, aleatoriamente, decide se houve fala
        await asyncio.sleep(random.uniform(0.5, 1.5))
        return bool(random.getrandbits(1))

class STTMock:
    """Simula a transcrição de áudio para texto."""
    async def transcribe(self, audio_data):
        # Aguarda para simular tempo de processamento
        await asyncio.sleep(1)
        # Gera um texto aleatório
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(5))

class LogicMock:
    """Simula a lógica de negócio (processamento do texto)."""
    async def process(self, text):
        # Aguarda para simular processamento
        await asyncio.sleep(1)
        return f"Resposta para: {text}"

class TTSMock:
    """Simula a síntese de texto para áudio, com possibilidade de interrupção."""
    async def synthesize(self, text, total_duration=2.0, increments=4):
        """
        Exemplo: divide a "fala" em 'increments' pedaços, dormindo total_duration.
        Se quiser interromper no meio, é possível checar fora desse loop.
        """
        step = total_duration / increments
        for _ in range(increments):
            await asyncio.sleep(step)
        # "Fala" concluída
        print(f"[TTS] Falando: {text}")

##############################################################################
#                      WORKERS (STT, LOGIC, TTS) - Assíncronos
##############################################################################

async def stt_worker(stt_queue, logic_queue, stt_module, orchestrator):
    """
    Pega tarefas de STT, realiza a transcrição e
    insere o resultado na fila de Lógica.
    """
    while True:
        task = await stt_queue.get()  # Espera tarefa de STT
        conversation_id = task.conversation_id
        job_id = task.job_id

        if not orchestrator.is_job_valid(conversation_id, job_id):
            stt_queue.task_done()
            continue

        # Processa STT
        start_time = time.time()
        text = await stt_module.transcribe(task.data)
        elapsed = time.time() - start_time
        print(f"[STT] Conv {conversation_id} (job {job_id}) => Texto: {text} (time: {elapsed:.2f}s)")

        # Cria nova tarefa para a fila de lógica
        new_task = PipelineTask(
            conversation_id=conversation_id,
            job_id=job_id,
            stage='logic',
            data=text
        )
        await logic_queue.put(new_task)
        stt_queue.task_done()

async def logic_worker(logic_queue, tts_queue, logic_module, orchestrator):
    """
    Pega tarefas de Lógica, processa e
    insere o resultado na fila de TTS.
    """
    while True:
        task = await logic_queue.get()
        conversation_id = task.conversation_id
        job_id = task.job_id

        if not orchestrator.is_job_valid(conversation_id, job_id):
            logic_queue.task_done()
            continue

        # Processa lógica de negócio
        start_time = time.time()
        reply_text = await logic_module.process(task.data)
        elapsed = time.time() - start_time
        print(f"[Logic] Conv {conversation_id} (job {job_id}) => {reply_text} (time: {elapsed:.2f}s)")

        # Cria nova tarefa para a fila de TTS
        new_task = PipelineTask(
            conversation_id=conversation_id,
            job_id=job_id,
            stage='tts',
            data=reply_text
        )
        await tts_queue.put(new_task)
        logic_queue.task_done()

async def tts_worker(tts_queue, tts_module, orchestrator):
    """
    Pega tarefas de TTS, sintetiza e envia ao usuário.
    Permite interrupção no meio (checando is_job_valid).
    """
    while True:
        task = await tts_queue.get()
        conversation_id = task.conversation_id
        job_id = task.job_id

        if not orchestrator.is_job_valid(conversation_id, job_id):
            tts_queue.task_done()
            continue

        # Processa TTS
        start_time = time.time()
        increments = 4
        step = 2.0 / increments
        interrupted = False
        for _ in range(increments):
            await asyncio.sleep(step)
            # Se a conversa foi interrompida no meio
            if not orchestrator.is_job_valid(conversation_id, job_id):
                interrupted = True
                break

        if not interrupted:
            elapsed = time.time() - start_time
            print(f"[TTS] Conv {conversation_id} (job {job_id}) => Fala concluída (time: {elapsed:.2f}s)")
        else:
            print(f"[TTS] Conv {conversation_id} (job {job_id}) => INTERROMPIDO no meio do TTS")

        tts_queue.task_done()

##############################################################################
#                       CLASSE Conversation
##############################################################################

class Conversation:
    """
    Faz loop de VAD + envia tarefas para a fila de STT.
    Incrementa job_id a cada fala detectada.
    """
    def __init__(self, conversation_id, orchestrator):
        self.id = conversation_id
        self.orchestrator = orchestrator
        self.vad = VADMock()
        self._running = True
        self.current_job_id = 0

    async def run(self):
        print(f"[Conversation {self.id}] Iniciada.")
        while self._running:
            # Aguarda VAD
            speech_detected = await self.vad.has_speech()
            if speech_detected and self._running:
                # Nova fala => incrementa job_id
                self.current_job_id += 1
                job_id = self.current_job_id

                # Simulamos "audio_data"
                audio_data = f"audio_chunk_{random.randint(1,100)}"

                # Envia para fila de STT
                stt_task = PipelineTask(
                    conversation_id=self.id,
                    job_id=job_id,
                    stage='stt',
                    data=audio_data
                )
                await self.orchestrator.stt_queue.put(stt_task)
                print(f"[Conversation {self.id}] Enviou tarefa STT (job {job_id}).")

            # Pequena pausa no loop
            await asyncio.sleep(0.1)

        print(f"[Conversation {self.id}] Encerrada.")

    def interrupt(self):
        """
        Se quisermos simular uma interrupção externa,
        basta incrementarmos o job_id.
        Isso faz com que qualquer pipeline anterior fique inválido.
        """
        self.current_job_id += 1
        print(f"[Conversation {self.id}] Interrompida => novo job_id = {self.current_job_id}")

    def stop(self):
        self._running = False

##############################################################################
#                       ORCHESTRATOR
##############################################################################

class Orchestrator:
    """
    Cria e gerencia conversas, filas e workers.
    """
    def __init__(self):
        # Filas para cada estágio
        self.stt_queue = asyncio.Queue()
        self.logic_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()

        self.stt_module = STTMock()
        self.logic_module = LogicMock()
        self.tts_module = TTSMock()

        # Mapeia conversation_id -> Conversation
        self.conversations = {}

        # Tarefas de workers
        self.workers_tasks = []

    def is_job_valid(self, conversation_id, job_id):
        """
        Verifica se a conversa existe, está ativa e se o job_id
        ainda é válido (não foi ultrapassado por uma nova fala).
        """
        conv = self.conversations.get(conversation_id)
        if not conv or not conv._running:
            return False
        return job_id == conv.current_job_id

    async def start_workers(self):
        """
        Inicia as tarefas de workers (STT, Logic, TTS).
        """
        stt_task = asyncio.create_task(stt_worker(
            self.stt_queue,
            self.logic_queue,
            self.stt_module,
            self
        ))
        logic_task = asyncio.create_task(logic_worker(
            self.logic_queue,
            self.tts_queue,
            self.logic_module,
            self
        ))
        tts_task = asyncio.create_task(tts_worker(
            self.tts_queue,
            self.tts_module,
            self
        ))

        self.workers_tasks.extend([stt_task, logic_task, tts_task])

    async def start_conversation(self, conversation_id):
        """
        Cria e inicia a conversa (loop de VAD).
        """
        if conversation_id in self.conversations:
            print(f"[Orchestrator] Conversa {conversation_id} já existe.")
            return
        conv = Conversation(conversation_id, self)
        self.conversations[conversation_id] = conv

        # Inicia a tarefa de loop da conversa
        task = asyncio.create_task(conv.run())
        return task

    def interrupt_conversation(self, conversation_id):
        """
        Interrompe a conversa (incrementa job_id),
        invalidando qualquer pipeline em andamento.
        """
        conv = self.conversations.get(conversation_id)
        if conv and conv._running:
            conv.interrupt()

    def stop_conversation(self, conversation_id):
        """
        Para completamente a conversa.
        """
        conv = self.conversations.get(conversation_id)
        if conv and conv._running:
            conv.stop()

    async def stop_all_conversations(self):
        """
        Para todas as conversas.
        """
        for conv in self.conversations.values():
            conv.stop()

    async def wait_for_workers(self):
        """
        Aguarda o término dos workers e esvaziamento das filas.
        """
        await self.stt_queue.join()
        await self.logic_queue.join()
        await self.tts_queue.join()

        for t in self.workers_tasks:
            t.cancel()
        await asyncio.gather(*self.workers_tasks, return_exceptions=True)
        print("[Orchestrator] Todos os workers finalizados.")

