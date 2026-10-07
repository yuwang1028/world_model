#!/bin/bash
# Second independent run (rep 1) of the three reviewer-style configurations with all four backbones.
cd /Users/yuwang/world_model_2/world_model
for m in gpt-3.5-turbo gpt-4o-mini gpt-4.1-nano gpt-4.1-mini; do
  (for pass in 1 2 3; do python3 -m revision.run_experiments --model $m --methods main --reps 2 --limit 50 --workers 8; done) > revision/logs/repeat_$m.log 2>&1 &
done
wait
echo "repeat finished at $(date)"
touch revision/logs/repeat_done
