#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# an example: run LoRA-E123 and MosLoRA-E123 on Librilight10

methods="LoRA MosLoRA"

keys="E1 E2 E3"

for method in "${methods}"
do

  for key in "${keys}"
  do
    ./run_training_inference_for_whisper.sh Librilight10 en ${methods} whisper_small "${key}" 10 13 4
  done

done
