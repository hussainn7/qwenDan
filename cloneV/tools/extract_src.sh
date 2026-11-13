#!/bin/zsh
# Extract mono 48 kHz audio from every original iPhone take in ~/Downloads (no " vN" edits) -> data/src/<name>.wav
cd ~/Downloads
for f in *.MP4 *.MOV *.mov(N); do
  [[ "$f" == *" v"* ]] && continue
  [[ "$f" =~ ^([0-9A-F]{8}-|IMG_) ]] || continue
  b=${f%.*}
  out=~/animations/siri-llm/vo/data/src/$b.wav
  [[ -s $out ]] && continue
  ffmpeg -nostdin -v error -y -i "$f" -vn -ac 1 -ar 48000 -c:a pcm_s16le "$out" && echo "ok $b"
done
