#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


#----------------------------run logs-------------------------------#


#----------------------------TODO-------------------------------#


#----------------------------training---------------------------#
# ./run_training_inference.sh "CDSD-partA" LoraAdapterH hubert base "A1" 11 13 4 0 "--batch_bins 20000000 --adapter_conf bottleneck=36" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_new


# select from [CDSD-partA, CDSD-partB, Librilight10, Librispeech100] or any combination of them
subcorpus=$1

# select a method from [FT, LoRA, MosLoRA, MeLoRA, LoraAdapterH, ...]
# NOTE that loraAdapterH denotes LoRA combines houslby adapter; FT denotes full-model fine-tuning
# TODO add FT, Houlsby_Adapter, MeLoRA, VeRA, LanFusion, CAM, and so on.
method=$2

# select from 
# [whisper_small, whisper_medium, whisper_large, ...]
# [hubert_base, hubert_large, hubert_xlarge]
model=$3
model_size=$4

# assign a special key for each experiment
key=$5

# [10, 11, 12, 13]
start_stage=$6
stop_stage=$7

# depends on backbone model size
# for whisper_small, inference_nj=8
inference_nj=$8

# specify gpu id
export CUDA_VISIBLE_DEVICES=$9

# args that overwrite the args in asr config.
# for FT, you problem only need to specify the batch_bins and lr, thus "--batch_bins ** --optim_conf lr=***"
# for lora, you might need to specify bs, lr, rank, alpha, thus "--batch_bins ** --optim_conf lr=*** --adapter_conf rank=1 --adapter_conf dropout_rat"
# for LoraAdapterH adapter-only mode, "--batch_bins ** --optim_conf lr=*** --adapter_conf use_lora=false --adapter_conf bottleneck=**"
asr_args=${10}

# output dir that contains all experiments
explink=${11}

# 检查软连接是否存在
if [ ! -d "espnet_outputs" ]; then
  # 如果文件夹不存在，则创建文件夹
  ln -s $explink espnet_outputs
  echo "软连接$explink 已创建."
fi

# decoding and other configures if exits
decode_batch_size=1
use_lm=false
use_wordlm=false
use_ngram=false
lm_config=conf/LM/train_lm_transformer.yaml
inference_lm=valid.loss.ave.pth

for sub in ${subcorpus}
do

  # LM/ASR/decoding configuration
  if [[ "$model" == *"w2v2"* ]] || [[ "$model" == *"hubert"* ]] || [[ "$model" == *"wavlm"* ]]; then
      inference_asr_model=valid.loss.ave.pth
      inference_config="conf/decoding/decode_asr_SSL_ctc_beam3.yaml"
      if [[ "$sub" == "Librilight10" ]]; then
        token_type=bpe
        nbpe=300
      elif [[ "$sub" == "Librispeech100" ]]; then
        token_type=bpe
        nbpe=5000
      elif [[ "$sub" == *"CDSD"* ]]; then
        token_type=char
        nbpe=1 # make no sense, just prevent from complain
      else
       echo "please specify the corpus"
       exit 1
      fi
      # SSL models do not need cleaner
      cleaner=none
  elif [[ "$model" == *"whisper"* ]]; then
      if [[ "${sub}" == *"Libri"* ]]; then
        # we change maxlenratio from 0.3 to 0.25, since 0.3 causes cuda-out-of-memory
        inference_config="conf/decoding/decode_asr_whisper_noctc_beam3_maxlenratio0.25.yaml"
      else
        inference_config="conf/decoding/decode_asr_whisper_noctc_beam3.yaml" # maxlenratio is 0.3
      fi
      inference_asr_model=valid.acc.ave.pth
      token_type=whisper_multilingual
      nbpe=1 # make no sense, just prevent from complain
      # whisper models do need whisper_basic as cleaner
      cleaner=whisper_basic
  else
      echo "Model not recognized. Please check the model name."
      exit 1
  fi

  # output dir for current experiment
  expdir=${explink}/${sub}_"${model}"_"${method}"_outputs
  # 检查文件夹是否存在
  if [ ! -d "$expdir" ]; then
    # 如果文件夹不存在，则创建文件夹
    mkdir "$expdir"
    echo "文件夹 $expdir 已创建."
  fi

  # dataset
  if [[ "${sub}" == *"Libri"* ]]; then
    train_set="${sub}_train"
    train_dev="Librispeech_valid"
    # test_set="Librispeech_valid_clean Librispeech_valid_other Librispeech_test_clean Librispeech_test_other"
    test_set="Librispeech_test_clean Librispeech_test_other"

  else
    train_set="${sub}_train"
    train_dev="${sub}_valid"
    test_set="${sub}_valid ${sub}_test"

  fi

  for k in ${key}
  do

    base_asr_config=conf/tuning_new/${method}/${model}_${model_size}_template.yaml
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
