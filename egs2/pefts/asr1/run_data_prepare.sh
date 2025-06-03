#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
set -u
set -o pipefail

#----------------------------Attention---------------------------#
# 1. the local_for_** are copied from other recipes and we assume that path.sh of it is empty
#    if path.sh is not empty, you are supposed to add its path in ./path.sh
# 2. for CDSD-partA, since there are 44 speakers, therefore we randomly select 4 speakers for test set, and the others are split 95-5 for train and dev sets.
#    for CDSD-partB, since there are only 8 speakers (each contributes 10h, totally 80h), therefore we roughly split the whole data by 76h-2h-2h.
# 3. if token_type=char, then re-run both stage 4 and stage 5.

#------------------Now only suport CDSD, AESRC, Librispeech100, Librilight10-------------------#
# TODO: Add Aishell1, Kespeech, Fleurs.
corpus=CDSD

# select from [whisper_multilingual, bpe, char]
token_type=whisper_multilingual

if [[ "${corpus}" == "CDSD" ]]; then
    #--------------for CDSD-----------------#
    # $data_sets are selected from ["CDSD-partA", "CDSD-partB", "CDSD-partA CDSD-partB", "CDSD-partA-spk01"]
    #--------------for CDSD-normal-----------------#
    # $data_sets are selected from ["CDSD-normal-partA-spk01", "CDSD-normal-partB-spk**"]
    local_data_dir=local/local_for_CDSD
    data_sets="CDSD-normal-partB-spk06"
    corpus_path=/media/rosie/Samsung_T5/Data_set/speech_corpus/CDSD
    local_data_opts="--corpus_path ${corpus_path} --data_sets ${data_sets}"
    whisper_language=zh

elif [[ "${corpus}" == "AESRC" ]]; then
    #--------------for AESRC-----------------#
    # $data_sets are selected from "US UK IND CHN JPN PT RU KR CA ES"
    local_data_dir=local/local_for_AESRC
    data_sets="US UK IND CHN JPN PT RU KR CA ES"
    corpus_path=/your/corora/path
    local_data_opts="--corpus ${corpus} --corpus_path ${corpus_path} --data_sets ${data_sets}"
    whisper_language=en

elif [[ "${corpus}" == "Librispeech100" ]]; then
    #--------------for Librispeech100-----------------#
    local_data_dir=local/local_for_librispeech100
    data_sets="Librispeech100"
    local_data_opts="--data_sets ${data_sets}"
    whisper_language=en

elif [[ "${corpus}" == "Librilight10" ]]; then
    #--------------for Librilight10-----------------#
    local_data_dir=local/local_for_librilight10
    data_sets="Librilight10"
    local_data_opts=
    whisper_language=en

fi

for subset in ${data_sets}; do
    
    if [[ "${token_type}" == *"bpe"* ]]; then
        
        nbpe=300 # for Librilight10
        # nbpe=5000 # for Librispeech100 
    else
        nbpe=1
    fi

    # dataset
    train_set="${subset}_train"
    if [[ "${corpus}" == *"Libri"* ]]; then
        valid_set="Librispeech_valid"
        test_sets="Librispeech_valid_clean Librispeech_valid_other Librispeech_test_clean Librispeech_test_other"
    else
        valid_set="${subset}_valid"
        test_sets="${subset}_valid ${subset}_test"
    fi

    ./asr.sh \
        --stage 1 \
        --stop_stage 5 \
        --skip_data_prep false \
        --skip_train false \
        --skip_eval false \
        --lang "${subset}" \
        --ngpu 1 \
        --nj 32 \
        --inference_nj 32 \
        --local_data_dir ${local_data_dir} \
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
        --bpe_train_text "data/${train_set}/text" \
        # "$@"

done
