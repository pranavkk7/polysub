"""PolySub: subtitles from any language to any language.

Speech recognition runs locally on the GPU (Qwen3-ASR with its forced aligner, or Whisper large-v3 for
languages Qwen cannot time), and translation is done by Claude, cue by cue, keeping the timing.
"""
__version__ = "2.0.0"
