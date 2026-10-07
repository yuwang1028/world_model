#!/bin/bash
# Waits for API credits, then finishes the gpt-4o-mini ablations (keeps the machine awake meanwhile).
cd /Users/yuwang/world_model_2/world_model
until python3 - <<'PY' 2>/dev/null
from openai import OpenAI
c=OpenAI(api_key=open('../openai_api_key.txt').read().strip(),max_retries=0,timeout=20)
c.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":"hi"}],max_tokens=1)
PY
do sleep 60; done
echo "credits available at $(date); starting ablation"
for pass in 1 2 3; do python3 -m revision.run_experiments --model gpt-4o-mini --methods ablation --reps 1 --limit 50 --workers 16; done
echo "ablation finished at $(date): $(cat revision/records/gpt-4o-mini/*/abl_*_rep0.jsonl | wc -l) of 1592 records"
touch revision/logs/ablation_done
