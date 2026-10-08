"""Burn subtitles into a copy of the video with FFmpeg (the subtitles become part of the picture)."""
import subprocess
from pathlib import Path


class BurnError(Exception):
    pass


def burn_subtitles(video, subtitles, output):
    video, subtitles, output = Path(video).resolve(), Path(subtitles).resolve(), Path(output).resolve()
    # FFmpeg's subtitles filter treats ':' and '\' in a path specially (Windows drive letters!), so it
    # runs from the subtitle's folder and gets only the file name.
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
        "-vf", f"subtitles={subtitles.name}:force_style='FontSize=20,Outline=2,Shadow=0,MarginV=28'",
        "-c:a", "copy", str(output),
    ]
    result = subprocess.run(command, cwd=subtitles.parent, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise BurnError(message[-1] if message else "FFmpeg could not add the subtitles to the video.")
    return output
