#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
set -u
set -o pipefail


corpus=CDSD # select from [CDSD, AESRC, Librispeech, ...]
token_type=whisper_multilingual # select from [whisper_multilingual, bpe, char]

if [[ "${corpus}" == *"CDSD"* ]]; then
    #--------------for CDSD-----------------#
    # $data_sets are selected from ["CDSD-partA", "CDSD-partB", "CDSD-partA CDSD-partB"]
    # corpus_path is "/autodl-fs/data/corora"

    data_sets="CDSD-partA"
    corpus_path=/autodl-fs/data/corora
    local_data_opts="--corpus ${corpus} --corpus_path ${corpus_path} --data_sets ${data_sets}"

    whisper_language=zh

elif [[ "${corpus}" == *"AESRC"* ]]; then
    #--------------for AESRC-----------------#
    # $langs are selected from "US UK IND CHN JPN PT RU KR CA ES"
    # corpus_path is "/autodl-fs/data/corora"

    data_sets="US UK IND CHN JPN PT RU KR CA ES"
    corpus_path=/autodl-fs/data/corora
    local_data_opts="--corpus ${corpus} --corpus_path ${corpus_path} --data_sets ${data_sets}"

    whisper_language=en

fi

for subset in ${data_sets}; do
    
    if [[ "${token_type}" == *"bpe"* ]]; then
        
        # example: 
        # if [[ "${subset}" == *"combined"* ]]; then
        #     nbpe=5000
        # else
        #     nbpe=150
        # fi

        echo "nbpe depends on the corpus type"
        exit 1
    else
        nbpe=1
    fi

    # dataset
    train_set="${subset}_train"
    valid_set="${subset}_valid"
    test_sets="${subset}_valid ${subset}_test"


    ./asr.sh \
        --stage 5 \
        --stop_stage 5 \
        --skip_data_prep false \
        --skip_train false \
        --skip_eval false \
        --lang "${subset}" \
        --ngpu 1 \
        --nj 32 \
        --inference_nj 32 \
        --nbpe "${nbpe}" \
        --max_wav_duration 30 \
        --audio_format "flac.ark" \
        --feats_type raw \
        --token_type  $token_type \
        --local_data_opts "${local_data_opts}" \
        --whisper_language ${whisper_language} \
        --use_lm false \
        --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_sets}" \
        --lm_train_text "data/${train_set}/text" \
        --bpe_train_text "data/${train_set}/text" "$@"

done
