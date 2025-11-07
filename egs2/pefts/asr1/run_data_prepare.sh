#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
set -u
set -o pipefail

: << 'EOF'

# running args for each corpus

1. For Librilight10, Librispeech100, Librispeech360:
    1. corpus = Librilight10, Librispeech100, or Librispeech360,
    2. Specify token_type
    3. Specify your own path for LIBRILIGHT_LIMITED and LIBRISPEECH in db.sh

2. For AESRC:
    1. corpus = AESRC
    2. Specify token_type
    3. Specify your own path for $corpus_path, where you store your raw data
    4. data_sets is selected from "US UK IND CHN JPN PT RU KR CA ES". We only support process the accent one by one.

3. For CDSD:
    1. corpus = CDSD
    2. Specify token_type
    3. Specify your own path for $corpus_path, where you store your raw data
    4. There are two mode for CDSD, one is to process the corpus as a whole; then data_sets="CDSD-partB" or "CDSD-partA"
       The other is to process the PartA in a speaker-wise manner; 
       then data_sets is selected from "CDSD-partA-spk04 CDSD-partA-spk05 CDSD-partA-spk07 CDSD-partA-spk12 CDSD-partA-spk13 CDSD-partA-spk14 CDSD-partA-spk23 CDSD-partA-spk38 CDSD-partA-spk42"
       We only support process the spk one by one.

4. For Libri360-spk:
    similar to speaker-wise PartA of CDSD

EOF


# support: Librilight10, Librispeech100, Librispeech360, Libri360-spk, AESRC, CDSD
corpus=AESRC

# select from [whisper_multilingual, whisper_en, bpe, char]
# for Whisper without "en", token_type=whisper_multilingual
# for whisper with "en", token_type=whisper_en
# for hubert, it depends on the corpus. Token_type can be bpe or char.
token_type=whisper_en

if [[ "${corpus}" == *"CDSD"* ]]; then
    local_data_dir=local/local_for_AESRC_and_CDSD
    # if $data_sets contains "spk", it performs the speaker-wise data prepare.
    # Note the selected speaker ids from partA are [04,05,07,12,13,14,23,38,42]
    # e.g., CDSD-partA-spk04 CDSD-partA-spk05 CDSD-partA-spk07 CDSD-partA-spk12 CDSD-partA-spk13 CDSD-partA-spk14 CDSD-partA-spk23 CDSD-partA-spk38 CDSD-partA-spk42
    # if not, it performs the whole partA/partB data prepare. e.g., data_sets="CDSD-partB" or "CDSD-partA"
    data_sets="CDSD-partA-spk42"
    corpus_path=/root/shared-data/datasets/CDSD-Interspeech # folder where you store your raw data
    local_data_opts="--corpus ${corpus} --corpus_path ${corpus_path} --data_sets ${data_sets}"
    whisper_language=zh

elif [[ "${corpus}" == *"AESRC"* ]]; then
    local_data_dir=local/local_for_AESRC_and_CDSD
    # US UK IND CHN JPN PT RU KR CA ES
    data_sets="US"
    corpus_path=/root/shared-data/datasets/AESRC # folder where you store your raw data
    local_data_opts="--corpus ${corpus} --corpus_path ${corpus_path} --data_sets ${data_sets}"
    whisper_language=en

elif [[ "${corpus}" == *"Librispeech100"* ]]; then
    local_data_dir=local/local_for_librispeech100
    data_sets="Librispeech100"
    local_data_opts="--data_sets ${data_sets}"
    whisper_language=en

elif [[ "${corpus}" == *"Librispeech360"* ]]; then
    local_data_dir=local/local_for_librispeech360
    data_sets="Librispeech360"
    local_data_opts="--data_sets ${data_sets}"
    whisper_language=en

elif [[ "${corpus}" == "Libri360-spk" ]]; then
    local_data_dir=local/local_for_librispeech360_speaker_adaptation
    # "Libri360-spk210 Libri360-spk3389 Libri360-spk2368 Libri360-spk3615 Libri360-spk479 Libri360-spk525 Libri360-spk6553 Libri360-spk492 Libri360-spk2388 Libri360-spk6458"
    data_sets="Libri360-spk210"
    corpus_path=/root/autodl-tmp/corpus_downloading_dir/Librispeech/LibriSpeech/train-clean-360 # folder where you store your raw data
    local_data_opts="--corpus_path ${corpus_path} --data_sets ${data_sets}"
    whisper_language=en

elif [[ "${corpus}" == *"Librilight10"* ]]; then
    local_data_dir=local/local_for_librilight10
    data_sets="Librilight10"
    local_data_opts=
    whisper_language=en
fi

for subset in ${data_sets}; do
    
    if [[ "${token_type}" == *"bpe"* ]]; then
        if [[ "${corpus}" == "Librilight10" ]]; then
            nbpe=300
        elif [[ "${corpus}" == "Librispeech100" ]]; then
            nbpe=5000
        elif [[ "${corpus}" == "AESRC" ]]; then
            nbpe=300 # not quite sure
        else
            echo "you are supposed to set the nbpe value for $corpus"
            exit 1
        fi
    else
        nbpe=1
    fi

    # dataset
    train_set="${subset}_train"
    if [[ "${corpus}" == *"Librispeech"* ]] || [[ "${corpus}" == "Librilight10" ]]; then
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
        --nj 8 \
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
