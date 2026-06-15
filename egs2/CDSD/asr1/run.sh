#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


#----------------------------CDSD run logs-------------------------------#
# ./run.sh CDSD-partB FT e_branchformer fasle A1 10 13 4 0 "" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A2 11 13 4 1 "--optim_conf lr=0.001 --scheduler_conf warmup_steps=15000" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A3 11 13 4 0 "--optim_conf lr=0.002 --scheduler_conf warmup_steps=35000" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A4 11 13 4 1 "--optim_conf lr=0.0005 --scheduler_conf warmup_steps=35000" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A5 11 13 4 1 "--optim_conf lr=0.003 --scheduler_conf warmup_steps=35000" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle RPE 11 13 4 0 "" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle Mscale1 11 13 4 0 "" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A6 11 13 4 1 "" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A7 11 13 4 1 "" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A7a 11 13 4 0 "--batch_size 256 --scheduler_conf warmup_steps=10000" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A6a 11 13 4 0 "" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A6b 11 13 4 1 "--optim_conf lr=0.005" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A7b 11 13 4 0 "--encoder_conf merge_type=averaging" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A7e 11 13 4 1 "--encoder_conf merge_type=glu3" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A7d 11 13 4 0 "--encoder_conf merge_type=conv1" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A7c 11 13 4 1 "--encoder_conf merge_type=original1" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A8 11 13 4 1 "--encoder_conf conv_after_att=true" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A7a1 11 13 4 1 "--encoder_conf merge_type=original" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A7a2 11 13 4 0 "--encoder_conf merge_type=original --encoder_conf cgmlp_linear_units=512" "" espnet_outputs ""
# ./run.sh CDSD-partB FT e_branchformer fasle A7a3 11 13 4 0 "--encoder_conf cgmlp_linear_units=512" "" espnet_outputs ""


# ./run.sh CDSD-partB FT conformer fasle A1 10 13 4 0 "" "" espnet_outputs ""
# ./run.sh CDSD-partB FT conformer fasle A2 11 13 4 1 "--optim_conf lr=0.002" "" espnet_outputs ""
# ./run.sh CDSD-partB FT conformer fasle A3 11 13 4 1 "--optim_conf lr=0.001" "" espnet_outputs ""


# ./run.sh CDSD-partB FT transformer fasle A1 11 13 4 0 "" "" espnet_outputs ""
# ./run.sh CDSD-partB FT transformer fasle B1 11 13 4 1 "--encoder_conf is_ATT_MLP_parallel=true" "" espnet_outputs ""
# ./run.sh CDSD-partB FT transformer fasle C1 11 13 4 0 "" "" espnet_outputs ""
# ./run.sh CDSD-partB FT transformer fasle C2 11 13 4 1 "--encoder_conf is_ATT_MLP_parallel=true" "" espnet_outputs ""


#----------------------------Aishell1 run logs-------------------------------#
# ./run.sh Aishell1 FT branchformer fasle A1 11 13 4 1 "" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle B1 11 13 8 0 "--encoder_conf using_glu=true" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle B2 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle B3 11 13 4 1 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_att=true" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A2 11 13 4 0 "--encoder_conf merge_method=averaging" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle B4 11 13 4 1 "--encoder_conf using_silu=true --encoder_conf merge_method=averaging" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A4 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A3 11 13 4 1 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_mlp=true" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A5 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_att=true" "" espnet_outputs ""

# ./run.sh Aishell1 FT branchformer fasle A6 11 13 4 1 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A1_residual" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A7 11 13 4 1 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A2_residual" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A4a 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A_residual" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A4b 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A_residual" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A4c 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A_residual --encoder_conf kernel_size=31" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A4d 11 13 4 1 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A2_residual --encoder_conf kernel_size=3" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A4f 11 13 4 1 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A2_residual --encoder_conf kernel_size=7" "" espnet_outputs ""
# ./run.sh Aishell1 FT branchformer fasle A4g 11 13 4 1 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A2_residual --encoder_conf kernel_size=15" "" espnet_outputs ""

# ./run.sh Aishell1 FT conformer fasle A1 11 13 4 0 "" "" espnet_outputs ""; ./run.sh Aishell1 FT transformer fasle A1 11 13 4 0 "" "" espnet_outputs ""
# ./run.sh Aishell1 FT e_branchformer fasle A1 11 13 4 1 "" "" espnet_outputs ""; ./run.sh Aishell1 FT branchformer fasle A4g 11 13 4 1 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A2_residual --encoder_conf kernel_size=15" "" espnet_outputs ""
# ./run.sh Aishell1 FT e_branchformer fasle A2 11 13 4 1 "" "" espnet_outputs ""

# ./run.sh Aishell1 FT transformer fasle A2 11 13 4 1 "--encoder_conf conv_after_mlp=true " "" espnet_outputs ""
# ./run.sh Aishell1 FT transformer fasle A3 11 13 4 0 "--encoder_conf conv_after_att=true " "" espnet_outputs ""

# 4090D: ./run.sh Aishell1 FT branchformer fasle A1 11 13 4 0 "--optim_conf lr=0.0005" "" espnet_outputs ""; ./run.sh Aishell1 FT branchformer fasle A3 11 13 4 0 "--optim_conf lr=0.001" "" espnet_outputs ""; ./run.sh Aishell1 FT branchformer fasle A2 11 13 4 0 "--optim_conf lr=0.002" "" espnet_outputs ""
# 4090D: ./run.sh Aishell1 FT branchformer fasle A4 11 13 4 0 "--optim_conf lr=0.002 --scheduler_conf warmup_steps=3200" "" espnet_outputs ""
# 4090D poor: ./run.sh Aishell1 FT branchformer fasle A5 11 13 4 0 "--optim_conf lr=0.001 --scheduler_conf warmup_steps=3200" "" espnet_outputs "" 
# 4090D poor: ./run.sh Aishell1 FT branchformer fasle A6 11 13 4 0 "--optim_conf lr=0.0005 --scheduler_conf warmup_steps=3200" "" espnet_outputs ""
# 4090D: not bad./run.sh Aishell1 FT branchformer fasle A7 11 13 4 0 "--optim_conf lr=0.0002 --scheduler_conf warmup_steps=3200" "" espnet_outputs ""
# 4090D poor./run.sh Aishell1 FT branchformer fasle A8 11 13 4 0 "--optim_conf lr=0.001 --scheduler_conf warmup_steps=100" "" espnet_outputs ""
# 4090D no scheduler ./run.sh Aishell1 FT branchformer fasle A9 11 13 4 0 "--optim_conf lr=0.001" "" espnet_outputs ""
# 4090D good ./run.sh Aishell1 FT branchformer fasle A10 11 13 4 0 "" "" espnet_outputs ""
# 4090D poor: ./run.sh Aishell1 FT branchformer fasle A11 11 13 4 0 "--optim_conf lr=0.002" "" espnet_outputs ""
# 4090D poor./run.sh Aishell1 FT branchformer fasle A12 11 13 4 0 "--optim_conf lr=0.0015" "" espnet_outputs ""
# 4090D good ./run.sh Aishell1 FT branchformer fasle A13 11 13 4 0 "--scheduler_conf warmup_steps=50000" "" espnet_outputs ""
# 4090D poor ./run.sh Aishell1 FT branchformer fasle A14 11 13 4 0 "--scheduler_conf warmup_steps=15000" "" espnet_outputs ""
# 4090D ./run.sh Aishell1 FT branchformer fasle A15 11 13 4 0 "--encoder_conf merge_method=averaging" "" espnet_outputs ""
# 4090D ./run.sh Aishell1 FT branchformer fasle A16 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A_residual --encoder_conf kernel_size=7" "" espnet_outputs ""
# 4090D ./run.sh Aishell1 FT branchformer fasle A17 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A_residual --encoder_conf kernel_size=15" "" espnet_outputs ""
# 4090D ./run.sh Aishell1 FT branchformer fasle A18 12 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A_residual --encoder_conf kernel_size=7 --encoder_conf is_att2mlp=true" "" espnet_outputs ""
# 4090D ./run.sh Aishell1 FT branchformer fasle A19 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A_residual --encoder_conf kernel_size=7 --encoder_conf is_att2mlp=true" "" espnet_outputs ""
# 4090D ./run.sh Aishell1 FT branchformer fasle A20 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A_residual --encoder_conf kernel_size=7 --encoder_conf is_att2mlp=true" "" espnet_outputs ""
# 4090D ./run.sh Aishell1 FT branchformer fasle A21 11 13 4 0 "--encoder_conf using_glu=true --encoder_conf merge_method=averaging --encoder_conf conv_after_merge=true --encoder_conf conv_type=A_residual --encoder_conf kernel_size=7 --encoder_conf is_att2mlp=true --encoder_conf cgmlp_linear_units=5120 --encoder_conf num_blocks=12" "" espnet_outputs ""

# ./run.sh Aishell1 FT conformer fasle A1 11 13 4 0 "" "" espnet_outputs ""
# ./run.sh Aishell1 FT e_branchformer fasle A1 11 13 4 0 "" "" espnet_outputs ""

# [CDSD-partA, CDSD-partB] from CDSD
# Aishell1
subcorpus=$1

# select a method from [FT, LoRA, adapter]
method=$2

# select: 
# [branchformer, e_branchformer, conformer, transformer]
# [whisper, whisper_small_en, hubert, hubert_small, ...]
model=$3
is_ctc_only=$4 # true or false
is_streaming=false # defualt, we do not consider streaming decoding

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

# args that overwrite the args in asr/inference_config
asr_args=${10}
inference_args=${11}

# output dir that contains all experiments
explink=${12}

# specify test sets
specify_test_set=${13}

# 检查软连接是否存在
if [ ! -d "espnet_outputs" ]; then
  # 如果文件夹不存在，则创建文件夹
  ln -s $explink espnet_outputs
  echo "软连接$explink 已创建."
fi

# LM args, by default, we do not use LM 
use_lm=false
use_wordlm=false
use_ngram=false
lm_config=conf/LM/train_lm_transformer.yaml
inference_lm=valid.loss.ave.pth

# decoding args
cleaner=none # only whisper needs cleaner
decode_batch_size=1 # untill now, espnet only suport decode batch size 1
inference_config="conf/decoding/decode_asr.yaml"
nbpe=1 # depends the backbone model

for sub in ${subcorpus}
do

  if [[ "$model" == *"branchformer"* || "$model" == "hubert" || "$model" == "conformer"  || "$model" == "transformer" ]]; then

      if [[ "${is_ctc_only}" == true ]]; then
        inference_asr_model=valid.loss.ave.pth
        inference_args="--ctc_weight 1.0"
      else
        inference_asr_model=valid.acc.ave.pth
      fi

      if [[ "$sub" == "Librilight10" ]]; then
        token_type=bpe # or Char
        nbpe=300
      elif [[ "$sub" == "Librispeech100" ]]; then
        token_type=bpe # or Char
        nbpe=5000
      elif [[ "$sub" == *"CDSD"* || "$sub" == *"Aishell"* ]]; then
        token_type=char
      else
       echo "please specify token_type for ${sub}"
       exit 1
      fi

  elif [[ "$model" == *"whisper"* ]]; then

      # whisper models do need whisper_basic as cleaner
      cleaner=whisper_basic

      # you can specify the model
      inference_asr_model=valid.acc.ave.pth

      if [[ "${model}" == *"en"* ]]; then
        token_type=whisper_en
        whisper_language=en
      else
        token_type=whisper_multilingual
        echo "please specify whisper_language for ${sub}"
        exit 1
      fi

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
  train_dev="${sub}_valid"
  if [[ "${specify_test_set}" == "" ]]; then
    if [[ "${sub}" == *"Libri"* ]]; then
      # test_set="${sub}_valid_clean ${sub}_valid_other ${sub}_test_clean ${sub}_test_other"
      test_set="${sub}_valid_clean ${sub}_test_clean"
    else
      test_set="${sub}_valid ${sub}_test"
    fi
  else
    test_set=$specify_test_set
  fi

  for k in ${key}
  do

    base_asr_config=conf/tuning/${method}/${model}_template.yaml
    asr_tag=${method}_${model}_${sub}-${k}

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
        --inference_args "${inference_args}" \
        --use_lm ${use_lm}                                 \
        --use_ngram ${use_ngram}                           \
        --use_word_lm ${use_wordlm}                        \
        --asr_config "${base_asr_config}"                  \
        --inference_config "${inference_config}"           \
        --lm_config "${lm_config}"                         \
        --cleaner "${cleaner}"                             \
        --whisper_language "${whisper_language:-}"         \
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
