#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


: << 'EOF'

# running template for each method

domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
key=E1
backbone=medium_en
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" FT whisper $backbone $key 10 13 3 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA whisper $backbone $key 10 13 3 0 "--adapter_conf key_name=$domain" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4CAT whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""

    # DictLoRA4MOLE and DictLoRA4SAMD denotes MoeLoRA and MoeLoRA*
    ./run_training_inference.sh "$domain" DictLoRA4MOLE whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4SAMD whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
done

EOF



# select:
# [CDSD-partA, CDSD-partB] from CDSD
# [Librilight10, Librispeech100]
# [US UK IND CHN JPN PT RU KR CA ES] from AESRC2020
subcorpus=$1

# select a method from [FT, LoRA, VeRA, DictLoRA, DictLoRA4LanFusion, DictLoRA4VeLoRA, DictLoRA4FasterVeLoRA, DictLoRA4ECAM, DictLoRA4PCAM, DictLoRA4MOLE, DictLoRA4SAMD, DictLoRA4CAT]
# DictLoRA4MOLE denotes MoeLoRA in paper
# DictLoRA4SAMD denotes MoeLoRA* in paper
method=$2

# select: 
# [whisper, hubert]
model=$3
# select: 
# [base, large] for hubert
# [small, medium, small_en, medium_en] for Whisper
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
# for FT, you only need to specify the batch_bins and lr, thus "--batch_bins ** --optim_conf lr=***"
# for lora, you might need to specify bs, lr, rank, alpha, thus "--batch_bins ** --optim_conf lr=*** --adapter_conf rank=1 --adapter_conf dropout_rate"
asr_args=${10}

# output dir that contains all experiments
# "/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR" for Incremental ASR
explink=${11}

# specify test sets
# IND_valid IND_test US_valid US_test UK_valid UK_test CHN_valid CHN_test JPN_valid JPN_test PT_valid PT_test RU_valid RU_test KR_valid KR_test CA_valid CA_test ES_valid ES_test
# Librispeech_valid_clean Librispeech_test_clean
specify_test_set=${12}

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
  if [[ "$model" == *"hubert"* ]]; then
      # SSL models do not need cleaner ??
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

      # you can specify the model
      inference_asr_model=valid.acc.ave.pth

      if [[ "${model_size}" == *"en"* ]]; then
        token_type=whisper_en
        whisper_language=en
      else
        token_type=whisper_multilingual
        if [[ "$sub" == "Librilight10" ]] || [[ "$sub" == "Librispeech100" ]] ; then
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
  train_set="${sub}_train"
  
  if [[ "${sub}" == *"Libri"* ]]; then
    train_dev="Librispeech_valid"
  else
    train_dev="${sub}_valid"
  fi

  if [[ "${specify_test_set}" == "" ]]; then
    if [[ "${sub}" == *"Libri"* ]]; then
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

    base_asr_config=conf/tuning/${method}/${model}_${model_size}_template.yaml
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

# --inference_tag "test-time_lora_merge" \
