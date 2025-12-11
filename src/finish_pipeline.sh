#!/bin/bash

# Log file
LOG_FILE="/app/src/pipeline_completion.log"

echo "Starting Pipeline Completion Monitor at $(date)" > $LOG_FILE

# 1. Wait for enrichment
echo "Waiting for 'builder.py --steps umls' to finish..." >> $LOG_FILE
# Loop while process exists
while pgrep -f "builder.py --steps umls" > /dev/null; do
    echo "Enrichment still running... $(date)" >> $LOG_FILE
    sleep 30
done

echo "Enrichment finished (or not found). Proceeding to Text Embeddings at $(date)" >> $LOG_FILE

# 2. Run Text Embeddings (this generates the missing .npy files)
cd /app/src
python3 knowledge_graph/builder.py --steps text >> $LOG_FILE 2>&1

echo "Text Embeddings Complete at $(date)" >> $LOG_FILE
echo "ALL DONE." >> $LOG_FILE
