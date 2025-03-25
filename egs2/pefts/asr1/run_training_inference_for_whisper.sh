#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


corpus=CDSD # select from KeSpeech | Aishell-1 | aesrc | CDSD | ...

if [[ "${corpus}" == "KeSpeech" ]]; then
  whisper_language=zh
  language="Beijing Northeastern Jiao-Liao Lan-Yin Jiang-Huai Ji-Lu Southwestern Zhongyuan" # "Jiao-Liao Lan-Yin Jiang-Huai Ji-Lu Southwestern Zhongyuan Mandarin100"

elif [[ "${corpus}" == "Aishell-1" ]]; then
  whisper_language=zh
  language="Aishell-1"

elif [[ "${corpus}" == "aesrc" ]]; then
  whisper_language=en
  language="US UK IND CHN JPN PT RU KR Librispeech100" # "US UK IND CHN JPN PT RU KR Librispeech100"

elif [[ "${corpus}" == "CDSD" ]]; then
  whisper_language=zh
  language="CDSD-partA"
fi

# select a method from:
# ------------------ PEFT methods -------------------#
# FT | Houlsby_Adapter | LoRA | SELoRA | MELoRA | MOSLoRA | VERA | NoRA |VeDA | 

# --------------- cross-accent methods --------------#
# SimAdapter | LanFusion | P-CAM | E-CAM |

method=LoRA
model=whisper_small # whisper_medium | whisper_small | whisper_large
key=${corpus}"-A1"

# specify gpu id
export CUDA_VISIBLE_DEVICES=0

# N hour --> 3600*N seconds; 
sleep 0

# output dir
explink=/autodl-fs/data/espnet_outputs
if [[ "${corpus}" == "aesrc" ]]; then
    echo "TODO"
    exit 1
elif [[ "${corpus}" == "KeSpeech" ]]; then
    echo "TODO"
    exit 1
elif [[ "${corpus}" == "CDSD" ]]; then
  expdir=${explink}/${corpus}_"${method}"_outputs
fi

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
    if [[ "${corpus}" == "aesrc" ]]; then
      token_type=bpe
    else # Kespeech
      token_type=char
    fi
elif [[ "$model" == *"whisper"* ]]; then
    if [[ "${corpus}" == "aesrc" ]]; then
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
inference_nj=4

# Other configures if exits
use_lm=false
use_wordlm=false
use_ngram=false
lm_config=conf/LM/train_lm_transformer.yaml
inference_lm=valid.loss.ave.pth

for lang in $language
do

  # dataset
  if [[ "${corpus}" == *"aesrc"* ]]; then
    train_set="${lang}_train"
    train_dev="${lang}_valid"

    if [[ ${lang} == *"Libri"* ]]; then
      test_set="${lang}_valid_clean ${lang}_valid_other ${lang}_test_clean ${lang}_test_other" # wer_valid=wer_valid_clean + wer_valid_other; wer_test=wer_test_clean + wer_test_other
    else
      test_set="${lang}_valid ${lang}_test"
    fi

    if [[ "${lang}" == *"combined"* ]]; then
        nbpe=5000
    elif [[ "${lang}" == *"Libri"* ]]; then
        nbpe=5000
    else
        nbpe=150
    fi

  elif [[ "${corpus}" == *"CDSD"* ]]; then
    train_set="${lang}_train"
    train_dev="${lang}_valid"
    test_set="${lang}_valid ${lang}_test"
    nbpe=30

  else # kespeech
    train_set=train_${lang}
    train_dev=dev_${lang}
    test_set="${train_dev} test_$lang"
    nbpe=30
  fi

  for k in $key
  do

    # corresponding config file, need manually create
    asr_config=conf/tuning/${method}/train_asr_${model}_${k}.yaml

    # training stage
    stage=11
    stop_stage=13

    # Data processing and training
    # stages 3~5 perfom data processing-related stuff
    # stages 6~9 perform language model-related stuff
    # stages 10~13 perform ASR-related stuff
    ./asr.sh \
        --nj 32 \
        --inference_nj ${inference_nj} \
        --gpu_inference true \
        --ngpu 1 \
        --stage $stage \
        --stop_stage $stop_stage \
        --lang ${lang} \
        --batch_size ${decode_batch_size} \
        --local_data_opts ${lang} \
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
        --lm_train_text "data/${train_set}/text" "$@"
  done

done
