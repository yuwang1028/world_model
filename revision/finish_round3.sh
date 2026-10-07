#!/bin/bash
# Waits for API credits, then finishes the second run of the two headline mechanism variants on the development split.
cd /Users/yuwang/world_model_2/world_model
until python3 - <<'PY' 2>/dev/null
from openai import OpenAI
c=OpenAI(api_key=open('../openai_api_key.txt').read().strip(),max_retries=0,timeout=20)
c.chat.completions.create(model="gpt-4.1-mini",messages=[{"role":"user","content":"hi"}],max_tokens=1)
PY
do sleep 60; done
echo "credits available at $(date); resuming"
for pass in 1 2 3; do python3 -m revision.run_experiments --model gpt-4.1-mini --methods agree_blind blind_wm --datasets gsm svamp gsm_hard mmlu_elementary stg mmlu_moral mmlu_law --start 50 --limit 30 --reps 2 --workers 10; done
echo "finished at $(date)"; touch revision/logs/round3_done
