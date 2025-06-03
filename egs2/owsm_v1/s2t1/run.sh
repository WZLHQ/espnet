#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

language="IND" # "US UK IND CHN JPN PT RU KR Librispeech100 CA ES"
dumpdir=/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_datasets/AESRC2020/dump_bpe

for lang in $language
do

    train_set="${lang}_train"
    train_dev="${lang}_valid"

    if [[ ${lang} == *"Libri"* ]]; then
      test_set="${lang}_valid_clean ${lang}_valid_other ${lang}_test_clean ${lang}_test_other" # wer_valid=wer_valid_clean + wer_valid_other; wer_test=wer_test_clean + wer_test_other
    else
      test_set="${lang}_valid ${lang}_test"
    fi

    nbpe=20000
    s2t_config=conf/tuning/train_s2t_transformer_lr1e-3_warmup5k.yaml
    inference_config=conf/decode_s2t.yaml

    ./s2t.sh \
        --stage 11 \
        --stop_stage 11 \
        --use_lm false \
        --ngpu 4 \
        --nj 128 \
        --gpu_inference true \
        --inference_nj 4 \
        --num_splits_s2t 5 \
        --feats_type raw \
        --audio_format flac.ark \
        --token_type bpe \
        --nbpe ${nbpe} \
        --dumpdir "${dumpdir}" \
        --s2t_config "${s2t_config}" \
        --inference_config "${inference_config}" \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --bpe_train_text "dump/raw/${train_set}/text" \
        --bpe_nlsyms data/nlsyms.txt \
        --lm_train_text "dump/raw/${train_set}/text" "$@"

done
