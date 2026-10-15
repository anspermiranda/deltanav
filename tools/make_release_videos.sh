#!/usr/bin/env bash
# Re-encode the two phone videos for the GitHub release: 1280 px H.264, no audio (about 60 MB each).
# usage: tools/make_release_videos.sh before.MOV after.MOV release/
set -euo pipefail
FF=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p "$3"
"$FF" -y -loglevel error -i "$1" -vf scale=1280:-2 -c:v libx264 -crf 26 -preset medium -an -movflags +faststart "$3/before.mp4"
"$FF" -y -loglevel error -i "$2" -vf scale=1280:-2 -c:v libx264 -crf 26 -preset medium -an -movflags +faststart "$3/after.mp4"
ls -la "$3"
