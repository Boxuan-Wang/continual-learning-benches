#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   1) Set DATA_ROOT / OUTPUT_DIR below, then run:
#      ./start_llm_cl_finetune.sh [extra run_llama_ewc.py args...]
#   2) Optional one-off override via env vars:
#      DATA_ROOT=/path/to/data OUTPUT_DIR=/path/to/out ./start_llm_cl_finetune.sh [extra args...]
#
# Examples:
#   ./start_llm_cl_finetune.sh
#   USE_LORA=0 TRAINABLE_PARAM_PATTERNS="lm_head" ./start_llm_cl_finetune.sh
#   MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" ./start_llm_cl_finetune.sh --epochs-per-context 2

DATA_ROOT="${DATA_ROOT:-/home/admin/workspace/aop_lab/collabmask/data/cl_preprocessed_4}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/admin/workspace/aop_lab/collabmask/results/mistral_7b/ewc}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Error: data_root does not exist: $DATA_ROOT"
  exit 1
fi

if [[ ! -f "run_llama_ewc.py" ]]; then
  echo "Error: run_llama_ewc.py not found. Run this script from repo root."
  exit 1
fi

# ---------- Defaults (override with env vars) ----------
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-AI-ModelScope/Mistral-7B-v0.1}"

USE_LORA="${USE_LORA:-1}"   # 1=true, 0=false
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

TRAINABLE_PARAM_PATTERNS="${TRAINABLE_PARAM_PATTERNS:-lm_head}"

EPOCHS_PER_CONTEXT="${EPOCHS_PER_CONTEXT:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE:-1}"
FISHER_MAX_BATCHES="${FISHER_MAX_BATCHES:-200}"

LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
REG_STRENGTH="${REG_STRENGTH:-1.0}"
GAMMA="${GAMMA:-1.0}"

MAX_LENGTH="${MAX_LENGTH:-1024}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
MALFORMED_POLICY="${MALFORMED_POLICY:-error}"  # error|skip

mkdir -p "$OUTPUT_DIR"

LORA_FLAG="--use-lora"
if [[ "$USE_LORA" == "0" ]]; then
  LORA_FLAG="--no-use-lora"
fi

echo "Starting LLM-CL finetuning..."
echo "  model:        $MODEL_NAME"
echo "  data_root:    $DATA_ROOT"
echo "  output_dir:   $OUTPUT_DIR"
echo "  use_lora:     $USE_LORA"
echo "  learning_rate:$LEARNING_RATE"

"$PYTHON_BIN" run_llama_ewc.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  "$LORA_FLAG" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --lora-target-modules "$LORA_TARGET_MODULES" \
  --trainable-param-patterns "$TRAINABLE_PARAM_PATTERNS" \
  --epochs-per-context "$EPOCHS_PER_CONTEXT" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --fisher-batch-size "$FISHER_BATCH_SIZE" \
  --fisher-max-batches "$FISHER_MAX_BATCHES" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --reg-strength "$REG_STRENGTH" \
  --gamma "$GAMMA" \
  --max-length "$MAX_LENGTH" \
  --num-workers "$NUM_WORKERS" \
  --log-every "$LOG_EVERY" \
  --malformed-policy "$MALFORMED_POLICY" \
  "$@"
