#!/bin/bash
# Waits for API credits, then finishes the second run (rep 1) of the main configurations on the three remaining backbones.
cd /Users/yuwang/world_model_2/world_model
until python3 - <<'PY' 2>/dev/null
from openai import OpenAI
c=OpenAI(api_key=open('../openai_api_key.txt').read().strip(),max_retries=0,timeout=20)
c.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":"hi"}],max_tokens=1)
PY
do sleep 60; done
echo "credits available at $(date); resuming"
for m in gpt-4o-mini gpt-4.1-nano gpt-4.1-mini; do
  (for pass in 1 2 3; do python3 -m revision.run_experiments --model $m --methods main --reps 2 --limit 50 --workers 8; done) > revision/logs/repeat2_$m.log 2>&1 &
done
wait
echo "repeat finished at $(date)"
touch revision/logs/repeat_done2
