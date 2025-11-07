#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


# select:
# Librilight10, Librispeech100, Librispeech360
# accents:[US UK IND CHN JPN PT RU KR CA ES] from AESRC
# speakers from Libri360-spk
subcorpus=$1

# select a method from:
# [FT, VeRA, DictLoRA, DictLoRA4LanFusion, DictLoRA4VeLoRA, DictLoRA4FasterVeLoRA, DictLoRA4ECAM, DictLoRA4PCAM, DictLoRA4MOLE, DictLoRA4SAMD, DictLoRA4CAT]
# DictLoRA4MOLE denotes MoeLoRA in paper
# DictLoRA4SAMD denotes MoeLoRA* in paper
method=$2

# select: 
# [whisper, hubert]
model=$3
# select: 
# [base, large] for hubert
# [base_en, small_en, medium_en] for Whisper
model_size=$4

# assign a special key for each experiment
key=$5

# [10, 11, 12, 13]
start_stage=$6
stop_stage=$7

# depends on backbone model size
# for whisper_small, inference_nj=8
# for whisper_medium, inference_nj=3
inference_nj=$8

# specify gpu id
export CUDA_VISIBLE_DEVICES=$9

# args that overwrite the args in $asr_config.
asr_args=${10}

# output dir that contains all experiments
explink=${11}

# specify test sets
# this would be helpful when decoding test/dev sets of other domains.
# 
specify_test_set=${12}

# check if the folder "espnet_outputs" exists.
if [ ! -d "espnet_outputs" ]; then
  # if no, then create
  ln -s $explink espnet_outputs
  echo "softlink $explink has been created."
fi

# decoding and other configures if exits
decode_batch_size=1 # actually, this value is limited to 1.
use_lm=false
use_wordlm=false
use_ngram=false
lm_config=conf/LM/train_lm_transformer.yaml # this won't work unless use_lm==True
inference_lm=valid.loss.ave.pth # this won't work unless use_lm==True

for sub in ${subcorpus}
do

  # LM/ASR/decoding configuration for each $subcorpus
  if [[ "$model" == *"hubert"* ]]; then
      # SSL models do not need cleaner ???
      cleaner=none
      inference_asr_model=valid.loss.ave.pth
      inference_config="conf/decoding/decode_asr_SSL_ctc_beam3.yaml"
      if [[ "$sub" == "Librilight10" ]]; then
        token_type=bpe # or Char
        nbpe=300
      elif [[ "$sub" == "Librispeech100" ]]; then
        token_type=bpe # or Char
        nbpe=5000
      elif [[ "$sub" == *"CDSD"* ]]; then
        token_type=char
        nbpe=1 # make no sense, just prevent from complain
      else
       echo "please specify token_type for ${sub}"
       exit 1
      fi

  elif [[ "$model" == *"whisper"* ]]; then
      # whisper models do need whisper_basic as cleaner
      cleaner=whisper_basic
      inference_config="conf/decoding/decode_asr_whisper_noctc_beam3.yaml"
      # you can specify the inference_asr_model
      inference_asr_model=valid.acc.ave.pth
      if [[ "${model_size}" == *"en"* ]]; then
        token_type=whisper_en
        whisper_language=en
      else
        token_type=whisper_multilingual
        if [[ "$sub" == "Librilight10" ]] || [[ "$sub" == *"Librispeech"* ]] ; then
          whisper_language=en
        elif [[ "$sub" == *"CDSD"* ]]; then
          whisper_language=zh
        else
          echo "please specify whisper_language for ${sub}"
          exit 1
        fi
      fi
      nbpe=1 # make no sense, just prevent from complain

  else
      echo "$model not recognized. Please check the model name."
      exit 1
  fi

  # create output dir for current experiment
  expdir=${explink}/${sub}_"${model}"_"${method}"_outputs
  if [ ! -d "$expdir" ]; then
    mkdir "$expdir"
    echo "folder $expdir has been created."
  fi

  # dataset
  train_set="${sub}_train"
  
  if [[ "$sub" == "Librilight10" ]] || [[ "$sub" == *"Librispeech"* ]] ; then
    train_dev="Librispeech_valid"
  else
    train_dev="${sub}_valid"
  fi

  if [[ "${specify_test_set}" == "" ]]; then
    if [[ "$sub" == "Librilight10" ]] || [[ "$sub" == *"Librispeech"* ]] ; then
      # test_set="Librispeech_valid_clean Librispeech_valid_other Librispeech_test_clean Librispeech_test_other"
      test_set="Librispeech_valid_clean Librispeech_test_clean"
    else
      # test_set="${sub}_valid ${sub}_test"
      test_set="${sub}_test"
    fi
  else
    test_set=$specify_test_set
  fi

  for k in ${key}
  do

    # Note that the $asr_args will overwrite the args of the template
    base_asr_config=conf/tuning/${method}/${model}_${model_size}_template.yaml
    # create asr tag for current experiment
    asr_tag=${method}_${model}-${model_size}_${sub}-${k}

    ./asr.sh \
        --nj 4 \
        --inference_nj ${inference_nj} \
        --gpu_inference true \
        --ngpu 1 \
        --stage $start_stage \
        --stop_stage $stop_stage \
        --lang ${sub} \
        --batch_size ${decode_batch_size} \
        --audio_format "flac.ark" \
        --feats_type raw \
        --nbpe ${nbpe} \
        --token_type  $token_type \
        --feats_normalize "" \
        --expdir "${expdir}" \
        --asr_tag "${asr_tag}" \
        --asr_args "${asr_args}" \
        --use_lm ${use_lm}                                 \
        --use_ngram ${use_ngram}                           \
        --use_word_lm ${use_wordlm}                        \
        --asr_config "${base_asr_config}"                  \
        --inference_config "${inference_config}"           \
        --lm_config "${lm_config}"                         \
        --cleaner "${cleaner}"                             \
        --whisper_language ${whisper_language}             \
        --inference_asr_model "${inference_asr_model}"     \
        --inference_lm ${inference_lm}                     \
        --train_set "${train_set}"                         \
        --valid_set "${train_dev}"                         \
        --test_sets "${test_set}"            \
        --asr_speech_fold_length 512 \
        --asr_text_fold_length 150 \
        --lm_fold_length 150 \
        --lm_train_text "data/${train_set}/text" \
        --ignore_init_mismatch true \
        # "$@"
  done
done

# the following code renames the name of inference_tag
# this would be helpful in some cases
# --inference_tag "MAS-LoRA" \
