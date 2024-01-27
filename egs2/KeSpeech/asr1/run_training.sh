#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=Beijing
train_set=train_Beijing
valid_set=dev_Beijing
test_sets=test_Beijing # 这个test是包含所有方言和阶段的！

dumpdir=/home/rosie/DeepLearning/speech-toolkits/corpora/KeSpeech

#asr_config=conf/w2v2.yaml

asr_config=conf/train_asr_branchformer.yaml
inference_config=conf/decode_asr_branchformer.yaml
lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

./asr.sh \
    --nj 32 \
    --inference_nj 32 \
    --ngpu 2 \
    --stage 6 \
    --stop_stage 13 \
    --lang ${lang} \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --dumpdir "${dumpdir}" \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --lm_config "${lm_config}"                         \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${valid_set} ${test_sets}"            \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"


