"""
speech_input.py -- the "Voice Recording" and "Speech to text" boxes of
the pipeline (hearing -> Deaf direction).

Speech recognition uses Vosk, a free PRE-TRAINED speech-to-text model.
It runs fully offline (no internet needed at the school) and is small
enough for a Raspberry Pi 5.

One-time setup:
    pip install vosk sounddevice
    Download vosk-model-small-en-us-0.15.zip from
        https://alphacephei.com/vosk/models
    and unzip it so this folder exists:
        slm_training_v2/models/vosk-model-small-en-us-0.15

Until that is done, is_available() returns False and the pipeline falls
back to typed text, so everything else still runs.
"""

import json
import os

SAMPLE_RATE = 16000          # Vosk models expect 16 kHz mono audio
RECORD_SECONDS = 5.0         # fixed recording length used by record()
MAX_RECORD_SECONDS = 8.0     # record_until_silence(): longest a spoken turn may be
NO_SPEECH_SECONDS = 5.0      # record_until_silence(): give up if nobody starts talking
CHUNK_SECONDS = 0.25         # audio is read and checked in pieces of this length
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "models", "vosk-model-small-en-us-0.15")

_model = None


class SpeechUnavailable(RuntimeError):
    """Raised when the microphone library or the Vosk model is missing."""


def is_available():
    """True when recording and speech recognition are both set up."""
    try:
        import sounddevice  # noqa: F401
        import vosk  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(MODEL_DIR)


def record(seconds=RECORD_SECONDS):
    """Voice Recording: capture `seconds` of microphone audio.
    Returns raw 16-bit mono audio as bytes."""
    try:
        import sounddevice as sd
    except ImportError:
        raise SpeechUnavailable(
            "Microphone support is not installed. Run: pip install sounddevice")
    frames = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                    channels=1, dtype="int16")
    sd.wait()
    return frames.tobytes()


def record_until_silence(max_seconds=MAX_RECORD_SECONDS, no_speech_seconds=NO_SPEECH_SECONDS):
    """Voice Recording that stops as soon as the speaker stops talking,
    instead of always waiting a fixed time. A phone screen is the students'
    current fallback, so the device has to be faster than typing on one.
    Returns (audio bytes, why the recording stopped)."""
    try:
        import sounddevice as sd
    except ImportError:
        raise SpeechUnavailable(
            "Microphone support is not installed. Run: pip install sounddevice")
    frames = int(CHUNK_SECONDS * SAMPLE_RATE)
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=frames,
                           channels=1, dtype="int16") as stream:
        def chunks():
            while True:
                data, _overflowed = stream.read(frames)
                yield bytes(data)
        return capture(chunks(), max_seconds, no_speech_seconds)


def capture(chunks, max_seconds=MAX_RECORD_SECONDS, no_speech_seconds=NO_SPEECH_SECONDS):
    """Collect 16 kHz 16-bit audio chunks until Vosk detects that the
    speaker paused after saying something. Works on any sequence of
    chunks, so it can be tested with a recording instead of a microphone.
    Returns (audio bytes, why the recording stopped)."""
    from vosk import KaldiRecognizer
    recognizer = KaldiRecognizer(_load_model(), SAMPLE_RATE)
    audio = bytearray()
    heard = False
    for chunk in chunks:
        audio += chunk
        seconds = len(audio) / (2 * SAMPLE_RATE)
        if recognizer.AcceptWaveform(chunk):
            # Vosk found the end of an utterance (a pause after speech)
            if json.loads(recognizer.Result()).get("text"):
                return bytes(audio), "stopped when the speaker paused"
        elif json.loads(recognizer.PartialResult()).get("partial"):
            heard = True
        if seconds >= max_seconds:
            return bytes(audio), f"reached the {max_seconds:g}-second limit"
        if not heard and seconds >= no_speech_seconds:
            return bytes(audio), "nobody spoke"
    return bytes(audio), "the audio ended"


def _load_model():
    """Load the Vosk model once and keep it."""
    try:
        from vosk import Model, SetLogLevel
    except ImportError:
        raise SpeechUnavailable(
            "Speech recognition is not installed. Run: pip install vosk")
    global _model
    if _model is None:
        if not os.path.isdir(MODEL_DIR):
            raise SpeechUnavailable(
                f"Vosk model not found at {MODEL_DIR}. Download "
                "vosk-model-small-en-us-0.15 from https://alphacephei.com/vosk/models")
        SetLogLevel(-1)
        _model = Model(MODEL_DIR)
    return _model


def transcribe(audio, vocabulary=None):
    """Speech to text: turn recorded audio into words with the pre-trained
    Vosk model.

    vocabulary -- optional list of words. When given, Vosk only chooses
    among these words (anything else becomes "[unk]"), which is more
    accurate for a small closed vocabulary. Left as None, Vosk recognises
    ordinary English and the pipeline's Reduction step keeps only the
    words in the lookup table, which is the adviser's design.
    """
    try:
        from vosk import KaldiRecognizer
    except ImportError:
        raise SpeechUnavailable(
            "Speech recognition is not installed. Run: pip install vosk")
    _model = _load_model()

    if vocabulary:
        grammar = json.dumps(sorted(set(vocabulary)) + ["[unk]"])
        recognizer = KaldiRecognizer(_model, SAMPLE_RATE, grammar)
    else:
        recognizer = KaldiRecognizer(_model, SAMPLE_RATE)

    # Feed the audio in chunks. Vosk returns a finished phrase whenever it
    # detects a pause, so every phrase is collected, then the remainder.
    texts = []
    chunk = 8000
    for start in range(0, len(audio), chunk):
        if recognizer.AcceptWaveform(audio[start:start + chunk]):
            texts.append(json.loads(recognizer.Result()).get("text", ""))
    texts.append(json.loads(recognizer.FinalResult()).get("text", ""))
    return " ".join(t for t in texts if t).strip()
