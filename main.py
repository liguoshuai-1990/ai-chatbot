import os
import queue
import sys
import wave
import re
import io
import time
import logging
import threading
import collections
import numpy as np
import sounddevice as sd
import soundfile as sf
import ollama
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
from piper import PiperVoice

# ================= 配置中心 =================
CONFIG = {
    # 1. 核心模型配置
    "WAKE_WORD": "hey_jarvis",     # 唤醒词模型名称
    "WHISPER_MODEL": "small",      # STT模型大小: tiny, base, small, medium
    "OLLAMA_MODEL": "gemma3",      # LLM模型名称 (需在 ollama run 过)
    
    # 2. TTS 模型路径 (请修改为你本地的实际路径)
    "PIPER_MODEL": "voice.onnx", 
    "PIPER_CONFIG": "voice.onnx.json",

    # 3. 音频基础参数
    "SAMPLE_RATE": 16000,
    "CHUNK_SIZE": 1280,            # 80ms window for openwakeword
    
    # 4. 交互灵敏度参数
    "VAD_THRESHOLD": 0.03,         # VAD触发阈值 (0.0 ~ 1.0)，建议 0.02-0.05
    "PRE_RECORD_SEC": 0.8,         # 预录音时长(秒)，用于找回“被切掉的第一个字”
    "SILENCE_TIMEOUT": 1.2,        # 说话停顿超过几秒认为一句话结束
    "MAX_RECORD_SEC": 15,          # 单次对话最大录音时长
    "AUTO_SLEEP_TIMEOUT": 20,      # 无交互几秒后自动休眠
}

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 屏蔽 huggingface 和 http 请求的繁杂日志，只显示警告和错误
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)  # 很多库用 httpx 发请求
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 确保只保留我们自己的日志
logger = logging.getLogger("Jarvis")

# ================= 核心功能模块 =================

class WakeWordEngine:
    """唤醒词检测"""
    def __init__(self, keyword):
        logger.info(f"[*] 初始化唤醒引擎: {keyword}")
        openwakeword.utils.download_models() # 首次运行会下载
        self.model = Model(wakeword_models=[keyword])
        self.keyword = keyword
        self.reset()

    def detect(self, chunk):
        # chunk: int16 numpy array
        prediction = self.model.predict(chunk)
        return prediction.get(self.keyword, 0) > 0.5

    def reset(self):
        """重置模型状态，防止休眠后误触发"""
        self.model.reset()

class SpeechToText:
    """听觉系统 (Whisper)"""
    def __init__(self, model_size):
        logger.info(f"[*] 加载 STT 模型 ({model_size})...")
        # 如果有显卡，请将 device 改为 "cuda"
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)

    def transcribe(self, audio_np):
        segments, _ = self.model.transcribe(
            audio_np, 
            beam_size=5, 
            language="zh", 
            initial_prompt="以下是中文对话。" # 提示词有助于纠正标点和语种
        )
        return "".join([s.text for s in segments]).strip()

class Brain:
    """大脑 (Ollama)"""
    def __init__(self, model_name):
        logger.info(f"[*] 连接大脑 ({model_name})...")
        self.model_name = model_name
        self.history_limit = 10 # 只保留最近10轮对话
        self.history = [{
            'role': 'system', 
            'content': (
                '你是李卓阳。'
                '你是一个聪明的语音助手。'
                '请用简短、口语化的中文回答，就像朋友聊天一样。'
                '回答限制在 50 字以内。'
                '绝对禁止使用 Markdown、表情符号、星号(*)、井号(#)或列表。'
                '不要输出代码块。'
            )
        }]

    def think(self, text):
        self.history.append({'role': 'user', 'content': text})
        try:
            response = ollama.chat(model=self.model_name, messages=self.history)
            content = response['message']['content']
            
            # 清理思维链 (DeepSeek/R1 等模型会有 <think> 标签)
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            # 移除特殊符号，只保留文字和基本标点，方便 TTS 朗读
            content = re.sub(r'\s+', ' ', content).strip()
            
            self.history.append({'role': 'assistant', 'content': content})
            
            # 滚动清理历史，防止上下文爆炸
            if len(self.history) > self.history_limit:
                self.history = [self.history[0]] + self.history[-(self.history_limit-1):]
            
            return content
        except Exception as e:
            logger.error(f"大脑思考出错: {e}")
            return "抱歉，我的大脑断线了。"

class Speaker:
    """发声系统 (Piper TTS)"""
    def __init__(self, model_path, config_path):
        logger.info("[*] 激活 TTS 系统...")
        if not os.path.exists(model_path):
            logger.error(f"TTS模型文件缺失: {model_path}")
            self.voice = None
            return
            
        try:
            self.voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=False)
        except Exception as e:
            logger.error(f"TTS 加载失败: {e}")
            self.voice = None

    def speak(self, text):
        if not text or not self.voice: return
        try:
            stream_wav = io.BytesIO()
            with wave.open(stream_wav, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)
            
            stream_wav.seek(0)
            data, fs = sf.read(stream_wav)
            sd.play(data, fs)
            sd.wait() # 等待播放完毕
        except Exception as e:
            logger.error(f"TTS 播放错误: {e}")

# ================= 主控制类 =================

class VoiceAssistant:
    def __init__(self):
        self.running = True
        self.is_awake = False
        self.is_processing = False # 是否正在处理（对话/思考/播放中）
        self.last_act_time = time.time()
        
        # 队列与缓冲
        self.audio_queue = queue.Queue()
        self.task_queue = queue.Queue()
        
        # 预录音 Buffer (Deque)
        self.pre_record_len = int(CONFIG["PRE_RECORD_SEC"] * CONFIG["SAMPLE_RATE"] / CONFIG["CHUNK_SIZE"])
        self.audio_buffer = collections.deque(maxlen=self.pre_record_len)

        # 初始化模块
        self.ww = WakeWordEngine(CONFIG["WAKE_WORD"])
        self.stt = SpeechToText(CONFIG["WHISPER_MODEL"])
        self.brain = Brain(CONFIG["OLLAMA_MODEL"])
        self.tts = Speaker(CONFIG["PIPER_MODEL"], CONFIG["PIPER_CONFIG"])

    def audio_callback(self, indata, frames, time_info, status):
        """底层的音频采集回调"""
        if status: print(status, file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def processing_thread(self):
        """后台处理线程：负责 STT -> LLM -> TTS"""
        while self.running:
            try:
                # 获取录音数据
                frames = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue

            self.is_processing = True # 标记为处理中，暂停唤醒检测
            
            try:
                # 1. 语音转文字
                audio_data = b''.join(frames)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                user_text = self.stt.transcribe(audio_np)
                if not user_text:
                    logger.info("未检测到有效内容")
                    continue
                
                logger.info(f"User: {user_text}")

                # 2. 指令匹配
                if any(w in user_text for w in ["再见", "退出", "关机"]):
                    self.tts.speak("再见，随时待命。")
                    self.running = False
                    break

                # 3. 大脑思考
                reply = self.brain.think(user_text)
                logger.info(f"AI: {reply}")

                # 4. 语音播报
                self.tts.speak(reply)
                
                # 播报完更新最后活动时间
                self.last_act_time = time.time()

            except Exception as e:
                logger.error(f"处理流程异常: {e}")
            finally:
                self.is_processing = False # 处理完毕，恢复监听
                # 关键：处理完后清空音频队列，防止听到刚才的TTS声音再次触发
                with self.audio_queue.mutex:
                    self.audio_queue.queue.clear()

    def run(self):
        # 启动处理线程
        t = threading.Thread(target=self.processing_thread, daemon=True)
        t.start()

        # 启动麦克风
        stream = sd.RawInputStream(
            samplerate=CONFIG["SAMPLE_RATE"], 
            blocksize=CONFIG["CHUNK_SIZE"], 
            dtype='int16', 
            channels=1, 
            callback=self.audio_callback
        )

        with stream:
            logger.info(f"\n{'='*40}\n系统就绪，请说: {CONFIG['WAKE_WORD']}\n{'='*40}")
            
            recording_frames = []
            is_recording = False
            silence_start = None

            while self.running:
                try:
                    data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                chunk = np.frombuffer(data, dtype=np.int16)
                vol = np.abs(chunk).mean() / 32768.0 # 归一化音量 0.0-1.0

                # --- 情况A: AI正在说话/思考 ---
                if self.is_processing:
                    # 此时忽略麦克风输入，防止自言自语
                    continue

                # --- 情况B: 待唤醒状态 ---
                if not self.is_awake:
                    # 保持预录音 Buffer 更新
                    self.audio_buffer.append(data)
                    
                    if self.ww.detect(chunk):
                        logger.info(">>> 唤醒成功！我在听...")
                        self.tts.speak("我在") # 简短回应
                        self.is_awake = True
                        self.last_act_time = time.time()
                        self.audio_buffer.clear() # 清空旧缓存，准备录新指令
                    continue

                # --- 情况C: 唤醒后交互 (VAD逻辑) ---
                
                # 1. 自动休眠检测
                if not is_recording and (time.time() - self.last_act_time > CONFIG["AUTO_SLEEP_TIMEOUT"]):
                    logger.info(">>> 长时间无操作，自动休眠")
                    # self.tts.speak("我先休息了") # 可选
                    
                    # === 关键修复：防止立刻唤醒 ===
                    self.is_awake = False
                    self.ww.reset()             # 1. 重置模型概率
                    with self.audio_queue.mutex:
                        self.audio_queue.queue.clear() # 2. 清空积压音频
                    self.audio_buffer.clear()   # 3. 清空预录音
                    time.sleep(0.2)             # 4. 物理缓冲
                    continue

                # 2. 录音逻辑
                if vol > CONFIG["VAD_THRESHOLD"]:
                    if not is_recording:
                        logger.info("开始录音...")
                        is_recording = True
                        # 将预录音（说话前的几百毫秒）加入开头，防止吞字
                        recording_frames.extend(self.audio_buffer)
                        self.audio_buffer.clear()
                    
                    recording_frames.append(data)
                    silence_start = None # 有声音，重置静音计时
                    self.last_act_time = time.time()
                
                elif is_recording:
                    # 录音中，但是当前帧静音
                    recording_frames.append(data) # 继续录入静音片段，保持连贯
                    
                    if silence_start is None: 
                        silence_start = time.time()
                    
                    # 检查静音是否超时 -> 认为一句话结束
                    if time.time() - silence_start > CONFIG["SILENCE_TIMEOUT"]:
                        logger.info(f"录音结束 (时长: {len(recording_frames)*CONFIG['CHUNK_SIZE']/CONFIG['SAMPLE_RATE']:.1f}s)")
                        self.task_queue.put(list(recording_frames)) # 发送给处理线程
                        
                        # 重置录音状态
                        recording_frames = []
                        is_recording = False
                        silence_start = None
                    
                    # 强制最大时长截断
                    if len(recording_frames) * CONFIG["CHUNK_SIZE"] / CONFIG["SAMPLE_RATE"] > CONFIG["MAX_RECORD_SEC"]:
                        logger.info("达到最大录音时长，强制提交")
                        self.task_queue.put(list(recording_frames))
                        recording_frames = []
                        is_recording = False
                        silence_start = None

                else:
                    # 唤醒了但还没说话 (等待用户开口)
                    self.audio_buffer.append(data)

if __name__ == "__main__":
    try:
        bot = VoiceAssistant()
        bot.run()
    except KeyboardInterrupt:
        print("\n[系统] 手动停止")
    except Exception as e:
        print(f"\n[错误] {e}")
