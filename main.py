import sys
import os
import ctypes
import logging
import re
import numpy as np
import sounddevice as sd
import keyboard
import threading
import queue
import time
import random
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QWidget, QVBoxLayout, QGraphicsOpacityEffect
from PyQt6.QtGui import QFont, QIcon, QPainter, QPixmap, QColor, QLinearGradient, QPen, QRadialGradient, QPainterPath, QRegion
from PyQt6.QtCore import pyqtSignal, QObject, QRect, QRectF, Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QPointF, QParallelAnimationGroup

MODEL_NAME = os.getenv("TALKATIVE_MODEL", "base.en")
MODEL_DEVICE = os.getenv("TALKATIVE_DEVICE", "").strip().lower()
CPU_THREADS = max(1, (os.cpu_count() or 4) - 1)
HOTKEY = "ctrl+alt"
HOTKEY_LABEL = "Ctrl+Alt"
MODEL_REVISION = os.getenv("TALKATIVE_MODEL_REVISION", "main")
MODEL_CACHE_DIR = os.getenv(
    "TALKATIVE_MODEL_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
)
ALLOW_MODEL_DOWNLOAD = os.getenv("TALKATIVE_ALLOW_MODEL_DOWNLOAD", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
HOTWORDS = "Codex, TALKATIVE, VPS, BTC, Bitcoin, Ethereum, ETH, SOL, Solana, Dogecoin"
TRANSCRIBE_OPTIONS = {
    "beam_size": 1,
    "best_of": 1,
    "language": "en",
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "without_timestamps": True,
    "hotwords": HOTWORDS,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 300},
}
PASTE_FALLBACK_LENGTH = 40
DICTATION_SYMBOL_REPLACEMENTS = (
    (r"\bnew paragraph\b", "\n\n"),
    (r"\bnew line\b", "\n"),
    (r"\bquestion mark\b", "?"),
    (r"\bquestiong mark\b", "?"),
    (r"\bexclamation mark\b", "!"),
    (r"\bfull stop\b", "."),
    (r"\bperiod\b", "."),
    (r"\bcomma\b", ","),
    (r"\bsemicolon\b", ";"),
    (r"\bcolon\b", ":"),
    (r"\bopen parenthesis\b", "("),
    (r"\bclose parenthesis\b", ")"),
    (r"\bopen paren\b", "("),
    (r"\bclose paren\b", ")"),
    (r"\bopen bracket\b", "["),
    (r"\bclose bracket\b", "]"),
    (r"\bopen brace\b", "{"),
    (r"\bclose brace\b", "}"),
    (r"\bunderscore\b", "_"),
    (r"\bhyphen\b", "-"),
    (r"\bdash\b", "-"),
)
DICTATION_TEXT_REPLACEMENTS = (
    (r"\bcovidx\b", "Codex"),
    (r"\bcodex\b", "Codex"),
    (r"\btalkative\b", "TALKATIVE"),
    (r"\bdogglowns\b", "downloads"),
    (r"\bhugging ?face\b", "Hugging Face"),
    (r"\bchat ?gpt\b", "ChatGPT"),
    (r"\bopen ai\b", "OpenAI"),
    (r"\bfaster whisper\b", "faster-whisper"),
    (r"\bpower ?shell\b", "PowerShell"),
    (r"\bpy ?qt\b", "PyQt"),
    (r"\bread ?me(?: dot|\.) ?m ?d\b", "README.md"),
    (r"\breadme\.md\b", "README.md"),
    (r"\bmain(?: dot|\.) ?py\b", "main.py"),
    (r"\bmain\.py\b", "main.py"),
    (r"\bv\s*p\s*s\b", "VPS"),
    (r"\bb\s*t\s*c\b", "BTC"),
    (r"\bbitcoin\b", "Bitcoin"),
    (r"\bether(?:e|i)um\b", "Ethereum"),
    (r"\be\s*t\s*h\b", "ETH"),
    (r"\bs\s*o\s*l\b", "SOL"),
    (r"\bsolana\b", "Solana"),
    (r"\bdoge ?coin\b", "Dogecoin"),
    (r"\bapi\b", "API"),
    (r"\bcpu\b", "CPU"),
    (r"\bgpu\b", "GPU"),
    (r"\bvbs\b", "VBS"),
)

# Setup logging
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "talkative.log")
logging.basicConfig(
    filename=log_path,
    level=getattr(logging, os.getenv("TALKATIVE_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

def log_error(exctype, value, tb):
    logging.error("Uncaught exception", exc_info=(exctype, value, tb))

sys.excepthook = log_error


def has_working_cuda_runtime():
    if os.name != "nt":
        return False

    required_libraries = ("nvcuda.dll", "cublas64_12.dll")
    for library_name in required_libraries:
        try:
            ctypes.WinDLL(library_name)
        except OSError:
            logging.info("CUDA runtime unavailable: %s", library_name)
            return False

    return True


def build_model_load_candidates():
    if MODEL_DEVICE == "cpu":
        return [{"device": "cpu", "compute_type": "int8", "cpu_threads": CPU_THREADS}]

    if MODEL_DEVICE == "cuda":
        return [
            {"device": "cuda", "compute_type": "int8_float16"},
            {"device": "cuda", "compute_type": "float16"},
            {"device": "cpu", "compute_type": "int8", "cpu_threads": CPU_THREADS},
        ]

    if MODEL_DEVICE == "auto":
        return [
            {"device": "auto", "compute_type": "int8"},
            {"device": "cpu", "compute_type": "int8", "cpu_threads": CPU_THREADS},
        ]

    if has_working_cuda_runtime():
        return [
            {"device": "cuda", "compute_type": "int8_float16"},
            {"device": "cuda", "compute_type": "float16"},
            {"device": "cpu", "compute_type": "int8", "cpu_threads": CPU_THREADS},
        ]

    return [{"device": "cpu", "compute_type": "int8", "cpu_threads": CPU_THREADS}]


def resolve_model_reference():
    if os.path.exists(MODEL_NAME):
        return os.path.abspath(MODEL_NAME)

    if "/" in MODEL_NAME:
        return MODEL_NAME

    return f"Systran/faster-whisper-{MODEL_NAME}"


def resolve_model_source():
    model_reference = resolve_model_reference()

    if os.path.exists(model_reference):
        logging.info("Loading Whisper model from local path: %s", model_reference)
        return model_reference

    try:
        cached_snapshot = snapshot_download(
            repo_id=model_reference,
            revision=MODEL_REVISION,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=True,
        )
        logging.info("Resolved Whisper model from local cache: %s", cached_snapshot)
        return cached_snapshot
    except Exception as cache_error:
        if not ALLOW_MODEL_DOWNLOAD:
            raise RuntimeError(
                f"Whisper model {MODEL_NAME!r} is not available in local cache and downloads are disabled."
            ) from cache_error

        logging.info("Model cache miss for %s. Downloading once to local cache.", model_reference)
        downloaded_snapshot = snapshot_download(
            repo_id=model_reference,
            revision=MODEL_REVISION,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=False,
        )
        logging.info("Downloaded Whisper model to local cache: %s", downloaded_snapshot)
        return downloaded_snapshot

class RecordingIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.visible_top_margin = 15
        self.hidden_offset = 30
        
        self.num_bars = 16
        self.bar_width = 3
        self.bar_spacing = 3
        self.max_height = 18
        self.min_height = 3
        
        self.font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
        self.font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 110)
        self.text = "LISTENING"
        self.text_width = 85
        
        bars_width = self.num_bars * (self.bar_width + self.bar_spacing) - self.bar_spacing
        
        # Layout: [margin][indicator][spacing][text][spacing][bars][margin]
        self.content_width = 20 + 16 + self.text_width + 10 + bars_width + 15
        self.content_height = 40
        
        self.setFixedSize(self.content_width, self.content_height)
        self.update_shape_mask()

        self.target_amplitude = 0.0
        self.amplitudes = [0.0] * self.num_bars

        self.visible_pos = QPoint()
        self.hidden_pos = QPoint()
        self.sync_positions()
        self.move(self.hidden_pos)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self.opacity_effect)

        self.position_animation = QPropertyAnimation(self, b"pos", self)
        self.opacity_animation = QPropertyAnimation(self.opacity_effect, b"opacity", self)
        self.animation_group = QParallelAnimationGroup(self)
        self.animation_group.addAnimation(self.position_animation)
        self.animation_group.addAnimation(self.opacity_animation)
        self.animation_group.finished.connect(self.finish_animation)
        self.hide_after_animation = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)
        self.hide()

    def update_shape_mask(self):
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.0
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_shape_mask()

    def sync_positions(self):
        screen = QApplication.primaryScreen().geometry()
        x = screen.x() + (screen.width() - self.width()) // 2
        y = screen.y() + self.visible_top_margin
        self.visible_pos = QPoint(x, y)
        self.hidden_pos = QPoint(x, y - self.height() - self.hidden_offset)

    def set_amplitude(self, amp):
        self.target_amplitude = min(1.0, amp * 10.0)

    def update_animation(self):
        self.amplitudes.pop(0)
        noise = random.uniform(0.7, 1.3) if self.target_amplitude > 0.01 else 1.0
        new_amp = self.target_amplitude * noise
        self.amplitudes.append(new_amp)
        self.update()

    def show_animation(self):
        self.sync_positions()
        self.hide_after_animation = False
        self.animation_group.stop()

        if not self.isVisible():
            self.move(self.hidden_pos)
            self.opacity_effect.setOpacity(0.0)

        self.show()
        self.raise_()

        self.position_animation.setDuration(400)
        self.position_animation.setEasingCurve(QEasingCurve.Type.OutBack)
        self.position_animation.setStartValue(self.pos())
        self.position_animation.setEndValue(self.visible_pos)

        self.opacity_animation.setDuration(300)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.opacity_animation.setStartValue(self.opacity_effect.opacity())
        self.opacity_animation.setEndValue(1.0)

        self.animation_group.start()

    def hide_animation(self):
        self.target_amplitude = 0.0
        self.sync_positions()

        if not self.isVisible() and self.opacity_effect.opacity() <= 0.01:
            return

        self.hide_after_animation = True
        self.animation_group.stop()

        self.position_animation.setDuration(300)
        self.position_animation.setEasingCurve(QEasingCurve.Type.InBack)
        self.position_animation.setStartValue(self.pos())
        self.position_animation.setEndValue(self.hidden_pos)

        self.opacity_animation.setDuration(250)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.opacity_animation.setStartValue(self.opacity_effect.opacity())
        self.opacity_animation.setEndValue(0.0)

        self.animation_group.start()

    def finish_animation(self):
        if self.hide_after_animation and self.opacity_effect.opacity() <= 0.01:
            self.move(self.hidden_pos)
            self.hide()
        else:
            self.move(self.visible_pos)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.0
        pill_path = QPainterPath()
        pill_path.addRoundedRect(rect, radius, radius)

        # Base frosted glass layer
        painter.setBrush(QColor(16, 16, 18, 245))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(pill_path)

        # Layout computation
        start_x = rect.left() + 20
        center_y = rect.center().y()

        # Pulsing recording dot
        current_time = time.time()
        pulse = (np.sin(current_time * 6) + 1) / 2.0  # 0.0 to 1.0
        dot_radius = 3.5 + pulse * 1.0
        
        # Soft glow for dot
        glow_grad = QRadialGradient(start_x, center_y, dot_radius * 2.5)
        glow_grad.setColorAt(0.0, QColor(255, 255, 255, int(220 * pulse)))
        glow_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setBrush(glow_grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(start_x, center_y), dot_radius * 2.5, dot_radius * 2.5)

        # Solid dot
        painter.setBrush(QColor(255, 255, 255, 255))
        painter.drawEllipse(QPointF(start_x, center_y), dot_radius, dot_radius)

        start_x += 18

        # Text
        painter.setFont(self.font)
        painter.setPen(QColor(255, 255, 255, 255))
        text_rect = QRectF(start_x, rect.top(), self.text_width, rect.height())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text)
        
        start_x += self.text_width + 10

        # Audio Bars
        painter.setBrush(QColor(255, 255, 255, 255))
        painter.setPen(Qt.PenStyle.NoPen)

        for i, amp in enumerate(self.amplitudes):
            bar_x = start_x + i * (self.bar_width + self.bar_spacing)
            bar_h = self.min_height + amp * (self.max_height - self.min_height)
            bar_y = center_y - (bar_h / 2)
            painter.drawRoundedRect(QRectF(bar_x, bar_y, self.bar_width, bar_h), self.bar_width / 2.0, self.bar_width / 2.0)


class Communicator(QObject):
    update_icon = pyqtSignal(str)
    insert_text = pyqtSignal(str)


class TalkativeApp:
    def __init__(self):
        self.app = QApplication(sys.sys_argv if hasattr(sys, 'sys_argv') else sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        self.comm = Communicator()
        self.comm.update_icon.connect(self.set_icon_state)
        self.comm.insert_text.connect(self.type_transcribed_text)

        # UI Elements
        self.tray_icon = QSystemTrayIcon(self.app)
        self.recording_indicator = RecordingIndicator()
        
        self.set_icon_state("idle")
        
        self.menu = QMenu()
        quit_action = self.menu.addAction("Quit")
        quit_action.triggered.connect(self.quit_app)
        self.tray_icon.setContextMenu(self.menu)
        self.tray_icon.show()

        # State
        self.state = "idle" # idle, recording, processing
        self.audio_queue = queue.Queue()
        self.recording = False
        self.samplerate = 16000
        self.audio_data = []
        self.current_amplitude = 0.0
        
        # Auto-timeout timer (60 seconds) to prevent accidental long recordings
        self.timeout_timer = QTimer()
        self.timeout_timer.setSingleShot(True)
        self.timeout_timer.timeout.connect(self.stop_recording)

        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_indicator_amplitude)
        self.anim_timer.start(30)

        # Load Whisper model in a background thread to avoid freezing UI
        self.model = None
        threading.Thread(target=self.load_model, daemon=True).start()

        # Setup global hotkey
        keyboard.add_hotkey(HOTKEY, self.on_hotkey)

    def update_indicator_amplitude(self):
        if self.recording:
            self.recording_indicator.set_amplitude(self.current_amplitude)
            # Decay the stored amplitude quickly so the bars fall if audio stops
            self.current_amplitude *= 0.7
        else:
            self.recording_indicator.set_amplitude(0.0)

    def load_model(self):
        load_started_at = time.perf_counter()
        self.comm.update_icon.emit("processing")
        try:
            model_source = resolve_model_source()
        except Exception:
            logging.error("Failed to resolve Whisper model source for %s.", MODEL_NAME, exc_info=True)
            self.comm.update_icon.emit("idle")
            return

        for model_options in build_model_load_candidates():
            try:
                self.model = WhisperModel(model_source, local_files_only=True, **model_options)
                logging.info(
                    "Model loaded successfully in %.2fs: %s with device=%s compute_type=%s",
                    time.perf_counter() - load_started_at,
                    MODEL_NAME,
                    model_options["device"],
                    model_options["compute_type"],
                )
                break
            except Exception:
                logging.warning(
                    "Model load attempt failed for %s with %s",
                    MODEL_NAME,
                    model_options,
                    exc_info=True,
                )

        if self.model is None:
            logging.error("All model load attempts failed for %s.", MODEL_NAME)

        self.comm.update_icon.emit("idle")

    def create_icon(self, state):
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if state == "idle":
            color = QColor(100, 100, 100) # Gray
        elif state == "recording":
            color = QColor(255, 50, 50) # Red
        elif state == "processing":
            color = QColor(255, 200, 50) # Yellow
        else:
            color = QColor(100, 100, 100)

        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(8, 8, 48, 48)
        painter.end()
        return QIcon(pixmap)

    def set_icon_state(self, state):
        self.state = state
        self.tray_icon.setIcon(self.create_icon(state))
        
        if state == "recording":
            self.tray_icon.setToolTip(f"TALKATIVE - Recording... ({HOTKEY_LABEL} to Stop)")
            self.recording_indicator.show_animation()
        elif state == "processing":
            self.tray_icon.setToolTip("TALKATIVE - Processing...")
            self.recording_indicator.hide_animation()
        else:
            self.tray_icon.setToolTip(f"TALKATIVE - Idle ({HOTKEY_LABEL} to Record)")
            self.recording_indicator.hide_animation()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning(f"Audio status: {status}")
        if self.recording:
            self.audio_queue.put(indata.copy())
            # Update amplitude based on the loudest sample in this chunk
            self.current_amplitude = float(np.max(np.abs(indata)))

    def on_hotkey(self):
        if self.model is None:
            logging.info("Model is not loaded yet.")
            return

        if self.state == "idle":
            self.start_recording()
        elif self.state == "recording":
            self.stop_recording()

    def start_recording(self):
        logging.info("Starting recording...")
        self.recording = True
        self.audio_data = []
        self.comm.update_icon.emit("recording")
        
        # Start timeout timer (60 seconds)
        self.timeout_timer.start(60000)
        
        try:
            self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, dtype='float32', callback=self.audio_callback)
            self.stream.start()
        except Exception as e:
            logging.error(f"Failed to start stream: {e}")
            self.stop_recording()

    def stop_recording(self):
        if not self.recording:
            return
            
        logging.info("Stopping recording...")
        self.recording = False
        self.timeout_timer.stop()
        self.comm.update_icon.emit("processing")
        
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logging.error(f"Error closing stream: {e}")

        # Collect audio
        while not self.audio_queue.empty():
            self.audio_data.append(self.audio_queue.get())

        if len(self.audio_data) > 0:
            audio_np = np.concatenate(self.audio_data).flatten()
            # Process in background
            threading.Thread(target=self.process_audio, args=(audio_np,), daemon=True).start()
        else:
            self.comm.update_icon.emit("idle")

    def normalize_transcript_text(self, text):
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return ""

        text = re.sub(r"[ \t]+", " ", text)
        text = self.apply_dictation_replacements(text, DICTATION_TEXT_REPLACEMENTS)
        text = self.apply_dictation_replacements(text, DICTATION_SYMBOL_REPLACEMENTS)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        text = re.sub(r"\s*([_-])\s*", r"\1", text)
        text = re.sub(r"([,:;!?])([A-Za-z])", r"\1 \2", text)
        text = re.sub(r"\b(i)\b", "I", text)
        text = re.sub(r"\bi('(?:m|d|ll|ve|re|s))\b", lambda match: "I" + match.group(1), text, flags=re.IGNORECASE)
        text = self.capitalize_sentences(text)
        text = self.apply_dictation_replacements(text, DICTATION_TEXT_REPLACEMENTS)
        return text

    def apply_dictation_replacements(self, text, replacements):
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def capitalize_sentences(self, text):
        characters = list(text)
        capitalize_next = True

        for index, char in enumerate(characters):
            if capitalize_next and char.isalpha():
                characters[index] = char.upper()
                capitalize_next = False
            elif char.isalpha():
                capitalize_next = False

            if char in ".!?\n":
                capitalize_next = True

        return "".join(characters).strip()

    def should_use_paste_fallback(self, text):
        return (
            "\n" in text
            or "\t" in text
            or len(text) >= PASTE_FALLBACK_LENGTH
            or any(ord(char) > 127 for char in text)
        )

    def paste_text(self, text):
        clipboard = self.app.clipboard()
        previous_text = clipboard.text()
        clipboard.setText(text)
        keyboard.press_and_release("ctrl+v")
        QTimer.singleShot(250, lambda previous_text=previous_text: clipboard.setText(previous_text))

    def type_transcribed_text(self, text):
        if not text:
            return

        try:
            if self.should_use_paste_fallback(text):
                logging.info("Using clipboard paste fallback for transcribed text.")
                self.paste_text(text)
            else:
                keyboard.write(text, delay=0)
        except Exception as e:
            logging.warning("Direct typing failed, falling back to clipboard paste: %s", e)
            try:
                self.paste_text(text)
            except Exception as paste_error:
                logging.error(f"Text output failed: {paste_error}")

    def process_audio(self, audio_np):
        processing_started_at = time.perf_counter()
        try:
            segments, info = self.model.transcribe(audio_np, **TRANSCRIBE_OPTIONS)
            text = self.normalize_transcript_text(" ".join(segment.text for segment in segments))
            logging.info(
                "Transcription completed in %.2fs for %.2fs of audio.",
                time.perf_counter() - processing_started_at,
                len(audio_np) / self.samplerate,
            )
            logging.info("Transcribed %d characters.", len(text))
            if text:
                self.comm.insert_text.emit(text)
        except Exception as e:
            logging.error(f"Transcription error: {e}")
        finally:
            self.comm.update_icon.emit("idle")

    def quit_app(self):
        logging.info("Quitting app...")
        self.app.quit()


if __name__ == "__main__":
    app = TalkativeApp()
    sys.exit(app.app.exec())
