#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE="andrejorsula/drl_air_hockey@sha256:16356368969865589a47d815077c0694b4afe2db042a21359dd857893beece65"
readonly OUTPUT_DIR="${1:-/home/oliver/experiments/airhockey-memory-distillation/upstream-demo-2023-rendered}"
readonly STEPS="${STEPS:-500}"
readonly DISPLAY_NUMBER="${DISPLAY_NUMBER:-97}"
readonly VIRTUAL_DISPLAY=":${DISPLAY_NUMBER}"
readonly DISPLAY_SOCKET="/tmp/.X11-unix/X${DISPLAY_NUMBER}"

for command_name in docker ffmpeg Xvfb; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Required command is missing: %s\n' "${command_name}" >&2
        exit 1
    fi
done

if [[ -e "${DISPLAY_SOCKET}" ]]; then
    printf 'Virtual display socket already exists: %s\n' "${DISPLAY_SOCKET}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
for output_name in self_play.mp4 self_play_raw.mp4; do
    if [[ -e "${OUTPUT_DIR}/${output_name}" ]]; then
        printf 'Refusing to overwrite existing artefact: %s\n' \
            "${OUTPUT_DIR}/${output_name}" >&2
        exit 1
    fi
done

xvfb_pid=""
ffmpeg_pid=""
cleanup() {
    if [[ -n "${ffmpeg_pid}" ]] && kill -0 "${ffmpeg_pid}" 2>/dev/null; then
        kill -INT "${ffmpeg_pid}" 2>/dev/null || true
        wait "${ffmpeg_pid}" 2>/dev/null || true
    fi
    if [[ -n "${xvfb_pid}" ]] && kill -0 "${xvfb_pid}" 2>/dev/null; then
        kill "${xvfb_pid}" 2>/dev/null || true
        wait "${xvfb_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

Xvfb "${VIRTUAL_DISPLAY}" \
    -screen 0 1920x1080x24 \
    -ac \
    -nolisten tcp \
    >"${OUTPUT_DIR}/xvfb.log" 2>&1 &
xvfb_pid=$!

for _ in $(seq 1 50); do
    [[ -S "${DISPLAY_SOCKET}" ]] && break
    sleep 0.1
done
if [[ ! -S "${DISPLAY_SOCKET}" ]]; then
    printf 'Virtual display did not start: %s\n' "${VIRTUAL_DISPLAY}" >&2
    exit 1
fi

ffmpeg \
    -hide_banner \
    -loglevel warning \
    -y \
    -f x11grab \
    -framerate 30 \
    -video_size 1920x1080 \
    -i "${VIRTUAL_DISPLAY}.0" \
    -c:v libx264 \
    -preset veryfast \
    -pix_fmt yuv420p \
    "${OUTPUT_DIR}/self_play_raw.mp4" \
    >"${OUTPUT_DIR}/ffmpeg.log" 2>&1 &
ffmpeg_pid=$!

docker run --rm \
    --ipc host \
    --network host \
    --env CUDA_VISIBLE_DEVICES= \
    --env JAX_PLATFORM_NAME=cpu \
    --env TF_CPP_MIN_LOG_LEVEL=2 \
    --env DISPLAY="${VIRTUAL_DISPLAY}" \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "${IMAGE}" \
    python3 -O /src/drl_air_hockey/scripts/eval_dreamerv3.py \
    --n_episodes 1 \
    --steps_per_game "${STEPS}" \
    --render \
    >"${OUTPUT_DIR}/stdout.log" 2>"${OUTPUT_DIR}/stderr.log"

sleep 1
cleanup
trap - EXIT INT TERM

ffmpeg \
    -hide_banner \
    -i "${OUTPUT_DIR}/self_play_raw.mp4" \
    -vf "blackdetect=d=0.5:pix_th=0.02" \
    -an \
    -f null - \
    >"${OUTPUT_DIR}/blackdetect.log" 2>&1

trim_start="$({
    sed -n \
        's/.*black_start:0 black_end:\([0-9.]*\).*/\1/p' \
        "${OUTPUT_DIR}/blackdetect.log"
} | head -n 1)"

if [[ -n "${trim_start}" ]]; then
    ffmpeg \
        -hide_banner \
        -loglevel warning \
        -y \
        -ss "${trim_start}" \
        -i "${OUTPUT_DIR}/self_play_raw.mp4" \
        -c:v libx264 \
        -preset veryfast \
        -pix_fmt yuv420p \
        "${OUTPUT_DIR}/self_play.mp4" \
        >"${OUTPUT_DIR}/trim.log" 2>&1
else
    cp "${OUTPUT_DIR}/self_play_raw.mp4" "${OUTPUT_DIR}/self_play.mp4"
    printf 'No leading black interval was detected.\n' \
        >"${OUTPUT_DIR}/trim.log"
fi

ffprobe \
    -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height,avg_frame_rate,duration \
    -of default=noprint_wrappers=1 \
    "${OUTPUT_DIR}/self_play.mp4" \
    >"${OUTPUT_DIR}/video_metadata.txt"

sha256sum \
    "${OUTPUT_DIR}/self_play.mp4" \
    "${OUTPUT_DIR}/self_play_raw.mp4" \
    >"${OUTPUT_DIR}/SHA256SUMS.txt"
printf 'Recorded upstream demo: %s\n' "${OUTPUT_DIR}/self_play.mp4"
