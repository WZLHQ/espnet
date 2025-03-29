#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


#----------------------------template-------------------------------#
#                                          ---subcorpus----whisper_language--method-----model------keynum--start_stage--stop_stage
# ./run_training_inference_for_whisper.sh    "CDSD-partA"        zh          LoRA  whisper_small    TEST       10           13

#----------------------------run logs-------------------------------#
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh LoRA whisper_small A1-B1 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh LoRA whisper_small A2-B1 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh LoRA whisper_small A4 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small A0-B1 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small A0-B2 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small A0-B3 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small A3 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small A4 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small A5 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small AdapterH1 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small AdapterH2 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small AdapterH3 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small AdapterH4 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small AdapterH5 11 13
# ./run_training_inference_for_whisper.sh CDSD CDSD-partA zh loraAdapterH whisper_small AdapterH6 11 13; 
# ./run_training_inference_for_whisper.sh Librilight10 Librilight10 en LoRA whisper_small A1 12 13; 
# ./run_training_inference_for_whisper.sh Librilight10 Librilight10 en LoRA whisper_small A2 12 13; 
# ./run_training_inference_for_whisper.sh Librilight10 Librilight10 en LoRA whisper_small A3 12 13; 
# ./run_training_inference_for_whisper.sh Librilight10 Librilight10 en LoRA whisper_small A4 12 13; 
# ./run_training_inference_for_whisper.sh Librilight10 Librilight10 en LoRA whisper_small A5 12 13; 
# ./run_training_inference_for_whisper.sh Librilight10 Librilight10 en LoRA whisper_small A6 12 13; 
# ./run_training_inference_for_whisper.sh Librilight10 Librilight10 en LoRA whisper_small "A4 A5 A6" 12 13 2
# ./run_training_inference_for_whisper.sh Librilight10 Librilight10 en LoRA whisper_small "B1 B2 B3 B4 B5 B6" 11 13 2
# ./run_training_inference_for_whisper.sh Librilight10 Librilight10 en LoRA whisper_small "C1 C2 C3 C4 C5 C6" 11 13 4

#---------------------training---------------------#
# ./run_training_inference_for_whisper.sh CDSD-partA zh LoRA whisper_small A1 13 13 4

# specify gpu id
export CUDA_VISIBLE_DEVICES=0

# select from [CDSD-partA, Librispeech100, Librilight10] or any combination of them
subcorpus=$1

# ATTENTION: for english corpus, whisper_language=en; for Chinese corpus, whisper_language=zh
whisper_language=$2

# select a method from [LoRA, MosLoRA, MeLoRA, loraAdapterH, ...]
# NOTE that loraAdapterH denotes LoRA combines houslby adapter
# TODO add FT, Houlsby_Adapter, MeLoRA, VeRA, LanFusion, CAM, and so on.
method=$3

# select from [whisper_small, whisper_medium, whisper_large, ...]
model=$4

# assign a special key for each experiment
key=$5

# [10, 11, 12, 13]
start_stage=$6
stop_stage=$7

# depends on backbone model size
# for whisper_small, inference_nj=8
inference_nj=$8

# output dir
explink=/root/autodl-fs/espnet_outputs
expdir=${explink}/${subcorpus}_"${method}"_outputs
# 检查文件夹是否存在
if [ ! -d "$expdir" ]; then
  # 如果文件夹不存在，则创建文件夹
  mkdir "$expdir"
  echo "文件夹 $expdir 已创建."
fi
# 检查软连接是否存在
if [ ! -d "espnet_outputs" ]; then
  # 如果文件夹不存在，则创建文件夹
  ln -s $explink espnet_outputs
  echo "软连接$explink 已创建."
fi

# LM/ASR/decoding configuration
if [[ "$model" == *"w2v2"* ]] || [[ "$model" == *"hubert"* ]]; then
    inference_config="conf/decoding/decode_asr_SSL_ctc_beam3.yaml"
    inference_asr_model=valid.loss.ave.pth
    if [[ "${subcorpus}" == *"AESRC"* ]]; then
      token_type=bpe
    else
      token_type=char
    fi
elif [[ "$model" == *"whisper"* ]]; then
    if [[ "${subcorpus}" == *"AESRC"* ]]; then
      # we change maxlenratio from 0.3 to 0.25, since 0.3 causes cuda-out-of-memory
      inference_config="conf/decoding/decode_asr_whisper_noctc_beam3_maxlenratio0.25.yaml"
    elif [[ "${subcorpus}" == *"Libri"* ]]; then
      # we change maxlenratio from 0.3 to 0.25, since 0.3 causes cuda-out-of-memory
      inference_config="conf/decoding/decode_asr_whisper_noctc_beam3_maxlenratio0.25.yaml"
    else
      inference_config="conf/decoding/decode_asr_whisper_noctc_beam3.yaml" # maxlenratio is 0.3
    fi
    inference_asr_model=valid.acc.ave.pth
    token_type=whisper_multilingual
else
    echo "Model not recognized. Please check the model name."
    exit 1
fi

# decoding
decode_batch_size=1
# Other configures if exits
use_lm=false
use_wordlm=false
use_ngram=false
lm_config=conf/LM/train_lm_transformer.yaml
inference_lm=valid.loss.ave.pth

for sub in "${subcorpus}"
do

  # dataset
  if [[ "${sub}" == *"AESRC"* ]]; then
    train_set="${sub}_train"
    train_dev="${sub}_valid"
    test_set="${sub}_valid ${sub}_test"
    # 150 for each accent; 5000 for all combined accents
    nbpe=150

  elif [[ "${sub}" == *"Libri"* ]]; then
    train_set="${sub}_train"
    train_dev="Librispeech_valid"
    # test_set="Librispeech_valid_clean Librispeech_valid_other Librispeech_test_clean Librispeech_test_other"
    test_set="Librispeech_test_clean Librispeech_test_other"
    nbpe=5000 # TODO need to verify

  elif [[ "${sub}" == *"CDSD"* ]]; then
    train_set="${sub}_train"
    train_dev="${sub}_valid"
    test_set="${sub}_valid ${sub}_test"
    # PBE is not for CDSD, therefor it does not make any sense. we set nbpe to 30 to prevent from complain
    nbpe=30

  else
    # TODO need to verify
    train_set=train_${sub}
    train_dev=dev_${sub}
    test_set="${train_dev} test_$sub"
    nbpe=30
  fi

  for k in ${key}
  do

    # corresponding config file, need manually create
    asr_config=conf/tuning/${method}/train_asr_${model}_${sub}-${k}.yaml

    ./asr.sh \
        --nj 32 \
        --inference_nj ${inference_nj} \
        --gpu_inference true \
        --ngpu 1 \
        --stage $start_stage \
        --stop_stage $stop_stage \
        --lang ${sub} \
        --batch_size ${decode_batch_size} \
        --whisper_language ${whisper_language} \
        --audio_format "flac.ark" \
        --feats_type raw \
        --nbpe ${nbpe} \
        --token_type  $token_type \
        --feats_normalize "" \
        --expdir "${expdir}" \
        --use_lm ${use_lm}                                 \
        --use_ngram ${use_ngram}                           \
        --use_word_lm ${use_wordlm}                        \
        --asr_config "${asr_config}"                       \
        --inference_config "${inference_config}"           \
        --lm_config "${lm_config}"                         \
        --cleaner whisper_basic                            \
        --inference_asr_model "${inference_asr_model}"     \
        --inference_lm ${inference_lm}                     \
        --train_set "${train_set}"                         \
        --valid_set "${train_dev}"                         \
        --test_sets "${test_set}"            \
        --asr_speech_fold_length 512 \
        --asr_text_fold_length 150 \
        --lm_fold_length 150 \
        --lm_train_text "data/${train_set}/text" \
        # "$@"
  done

done
