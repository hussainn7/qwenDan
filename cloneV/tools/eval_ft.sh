#!/bin/zsh
# Generate + score fine-tuned configs on the 14 held-out Siri sentences.
#   tools/eval_ft.sh RUN EPOCH [gen args...]      e.g. tools/eval_ft.sh r16 4 --sub-temp 0.5 --name ft_r16_e4_sub05
set -e
cd ${0:A:h}/..
PY=~/ComfyUI-Qwen3TTS-Emotional/.venv/bin/python
run=$1 ep=$2; shift 2
name=ft_${run}_e${ep}
for ((i = 1; i <= $#; i++)); do [[ ${@[i]} == --name ]] && name=${@[i+1]}; done
$PY tools/finetune.py gen $run $ep "$@" 2>&1 | grep -E "^ft|merged|Error|error" || true
~/reels/.venv/bin/python tools/transcribe.py sweep/takes/$name/*.wav > /dev/null 2>&1
.venv-score/bin/python tools/score.py $name 2>/dev/null | tail -1
.venv-score/bin/python tools/prosody.py $name 2>/dev/null | tail -1
.venv-score/bin/python tools/natural.py sweep/natural.json sweep/takes/$name/*.wav > /dev/null 2>&1
