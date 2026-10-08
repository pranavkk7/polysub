"""Audio input: decode any video or audio file with FFmpeg, and split long recordings at quiet moments."""
import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 16000


class AudioError(Exception):
    pass


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def load_audio(path):
    """The audio track as 16 kHz mono float32 samples in [-1, 1], whatever the input format."""
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
               "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioError(message[-1] if message else "FFmpeg could not read the file.")
    audio = np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise AudioError("The file has no audio track.")
    return audio


def chunk_bounds(audio, max_seconds=150.0, search_seconds=30.0, sample_rate=SAMPLE_RATE, frame_seconds=0.02):
    """Splits a long recording into pieces of at most `max_seconds`, cutting at the quietest moment in
    the last `search_seconds` of each piece so no word is cut in half. Returns (start, end) sample indexes.

    Needed because Qwen's forced aligner times at most 5 minutes of speech per call."""
    total = len(audio)
    max_len = int(max_seconds * sample_rate)
    if total <= max_len:
        return [(0, total)]

    frame = int(frame_seconds * sample_rate)
    frames = total // frame
    energy = np.square(audio[: frames * frame].reshape(frames, frame)).mean(axis=1)
    smooth = np.convolve(energy, np.ones(15) / 15, mode="same")  # about 0.3 s, so a single quiet frame isn't chosen

    bounds, start = [], 0
    while total - start > max_len:
        low = (start + max_len - int(search_seconds * sample_rate)) // frame
        high = (start + max_len) // frame
        cut = (low + int(np.argmin(smooth[low:high]))) * frame + frame // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, total))
    return bounds
