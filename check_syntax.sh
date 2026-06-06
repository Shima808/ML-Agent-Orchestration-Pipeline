#!/bin/bash
cd ~/ml-orchestration
for f in core/client.py core/cost_tracker.py agents/builder.py agents/evaluator.py agents/critic.py agents/orchestrator.py main.py; do
    venv/bin/python3 -m py_compile "$f" && echo "OK: $f"
done
