#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


#----------------------------run logs-------------------------------#


#----------------------------TODO-------------------------------#


#----------------------------training---------------------------#
# ./run_training_inference.sh "Librispeech100" FT whisper small_en IASR-0 12 13 6 0 "" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "IND_valid IND_test US_valid US_test UK_valid UK_test CHN_valid CHN_test JPN_valid JPN_test PT_valid PT_test RU_valid RU_test KR_valid KR_test CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "Librispeech100" FT whisper small_en IASR-1 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --patience 3" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en IASR-FT-07-0 10 13 4 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en IASR-FT-07-1 11 13 4 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en IASR-FT-07-2 11 13 4 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1.2" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-0 10 13 4 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-1 11 13 4 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-2 11 13 4 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1.2" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en IASR-FT-07-3 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en IASR-FT-07-4 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en IASR-FT-07-5 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1.2" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-3 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-4 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf lora_L2_norm_weight=2" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-5 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf lora_L2_norm_weight=5" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-6 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf lora_L2_norm_weight=1 --adapter_conf rank=64 --adapter_conf alpha=64" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-7 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf lora_L2_norm_weight=2 --adapter_conf rank=64 --adapter_conf alpha=64" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-8 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf lora_L2_norm_weight=5 --adapter_conf rank=64 --adapter_conf alpha=64" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-9 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf lora_L2_norm_weight=1 --adapter_conf rank=128 --adapter_conf alpha=128" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-10 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf lora_L2_norm_weight=2 --adapter_conf rank=128 --adapter_conf alpha=128" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-11 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf lora_L2_norm_weight=5 --adapter_conf rank=128 --adapter_conf alpha=128" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-9-1 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1.1 --model_conf lora_L2_norm_weight=1 --adapter_conf rank=128 --adapter_conf alpha=128" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-9-2 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1.2 --model_conf lora_L2_norm_weight=1 --adapter_conf rank=128 --adapter_conf alpha=128" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en IASR-LoRA-07-9-3 12 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf lora_L2_norm_weight=0 --adapter_conf rank=128 --adapter_conf alpha=128 --optim_conf weight_decay=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en IASR-FT-07-0-1 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf FT_WCA=true" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en IASR-FT-07-0-2 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --scheduler_conf warmup_steps=1500 --model_conf temperature=1 --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-Emb-FT-0 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,token_embedding" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-0 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-0/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0 --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-1 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-2 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0 --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-3 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-4 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=2" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-5 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=3" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-6 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=4" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-1-1 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-1-2 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0.5" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-1-3 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-Emb-FT-1 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,token_embedding" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en LoRA-A1 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --adapter_conf rank=64 --adapter_conf alpha=64" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en LoRA-A2 12 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --adapter_conf rank=64 --adapter_conf alpha=64 --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en LoRA-A3 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --adapter_conf rank=64 --adapter_conf alpha=64 --optim_conf lr=0.00001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en LoRA-A1-1 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --adapter_conf rank=64 --adapter_conf alpha=64 --optim_conf weight_decay=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-7 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0 --optim_conf lr=0.000005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-8 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0 --optim_conf lr=0.000001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en LoRA-A1-2 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --adapter_conf rank=64 --adapter_conf alpha=64 --optim_conf weight_decay=1 --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-Emb-FT-2 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,token_embedding --optim_conf lr=0.000001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en LoRA-A4 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --adapter_conf rank=64 --adapter_conf alpha=64 --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-WCA-A1 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=1 --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-AWD-A1 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005 --optim_conf weight_decay=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-AWD-A2 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005 --optim_conf weight_decay=0.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-WCA-A2 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0.1 --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-WCA-A3 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0.01 --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to8-combined" FT whisper small_en DR-IASR-1 10 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en LoRA-A2-1 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --adapter_conf rank=64 --adapter_conf alpha=64 --optim_conf lr=0.0001 --model_conf temperature=0.9" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" LoRA whisper small_en LoRA-A2-4 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --adapter_conf rank=64 --adapter_conf alpha=64 --optim_conf lr=0.0001 --model_conf temperature=0.6" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-A1 10 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-A2 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --encoder_conf adapter_dim=128 --decoder_conf adapter_dim=128" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-A3 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --encoder_conf adapter_dim=256 --decoder_conf adapter_dim=256" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-WCA-B1 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-WCA-B2 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" FT whisper small_en ATT-MLP-FT-WCA-B3 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --model_conf FT_WCA=true --model_conf ft_L2_norm_weight=0.01" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-A4 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --encoder_conf adapter_dim=256 --decoder_conf adapter_dim=256 --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-B1 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --encoder_conf use_adapterH=false --encoder_conf use_CgAdapter=true --decoder_conf adapter_dim=256 --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-B2 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --encoder_conf use_adapterH=false --encoder_conf use_CgAdapter=true --decoder_conf adapter_dim=256 --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-B3 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --encoder_conf use_adapterH=false --encoder_conf use_CgAdapter=true --decoder_conf adapter_dim=256 --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-B4 11 13 5 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth --encoder_conf adapter_dim=64 --decoder_conf adapter_dim=64 --encoder_conf use_CgAdapter=true" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-A5 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test Librispeech100_valid_clean Librispeech100_test_clean CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en AdapterH-C1 13 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-small_en_Librispeech100-IASR-1/valid.acc.ave_3best.pth" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "0to7-combined_valid 0to7-combined_test"

# ./run_training_inference.sh "aishell-1" FT whisper small A1 12 13 5 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "CDSD-partA_valid CDSD-partA_test CDSD-partB_valid CDSD-partB_test"

# ./run_training_inference.sh "CDSD-partA" FT whisper small A1 12 13 6 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/aishell-1_whisper_FT_outputs/asr_FT_whisper-small_aishell-1-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 7 --optim_conf lr=0.00005 --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "dev_aishell-1 test_aishell-1"
# ./run_training_inference.sh "CDSD-partA" FT whisper small A2 12 13 6 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/aishell-1_whisper_FT_outputs/asr_FT_whisper-small_aishell-1-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 7 --optim_conf lr=0.00001 --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "dev_aishell-1 test_aishell-1"
# ./run_training_inference.sh "CDSD-partA" FT whisper small A3 12 13 6 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/aishell-1_whisper_FT_outputs/asr_FT_whisper-small_aishell-1-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 7 --optim_conf lr=0.000005 --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "dev_aishell-1 test_aishell-1"
# ./run_training_inference.sh "CDSD-partA" FT whisper small A4 12 13 6 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/aishell-1_whisper_FT_outputs/asr_FT_whisper-small_aishell-1-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 7 --optim_conf lr=0.000001 --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "dev_aishell-1 test_aishell-1"

# ./run_training_inference.sh "CDSD-partA" FT whisper small A1-1 12 13 6 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/aishell-1_whisper_FT_outputs/asr_FT_whisper-small_aishell-1-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 3 --optim_conf lr=0.00005 --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "dev_aishell-1 test_aishell-1"

# ./run_training_inference.sh "CDSD-partB" FT whisper small A1 12 13 4 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/aishell-1_whisper_FT_outputs/asr_FT_whisper-small_aishell-1-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 7 --optim_conf lr=0.00005 --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "dev_aishell-1 test_aishell-1"
# ./run_training_inference.sh "CDSD-partB" FT whisper small A2 12 13 4 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/aishell-1_whisper_FT_outputs/asr_FT_whisper-small_aishell-1-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 7 --optim_conf lr=0.00001 --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "dev_aishell-1 test_aishell-1"
# ./run_training_inference.sh "CDSD-partB" FT whisper small A3 12 13 4 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/aishell-1_whisper_FT_outputs/asr_FT_whisper-small_aishell-1-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 7 --optim_conf lr=0.000005 --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "dev_aishell-1 test_aishell-1"
# ./run_training_inference.sh "CDSD-partB" FT whisper small A4 12 13 4 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/aishell-1_whisper_FT_outputs/asr_FT_whisper-small_aishell-1-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 7 --optim_conf lr=0.000001 --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_IASR "dev_aishell-1 test_aishell-1"


# select:
# [CDSD-partA, CDSD-partB] from CDSD
# [Librilight10, Librispeech100]
# [US UK IND CHN JPN PT RU KR CA ES] from AESRC2020
# aishell-1
subcorpus=$1

# select a method from [FT, LoRA, AdapterH]
# TODO add LoRA variants, LanFusion, CAM, and so on.
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
# Librispeech100_valid_clean Librispeech100_test_clean
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
        whisper_language=zh
        # echo "please specify whisper_language for ${sub}"
        # exit 1
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
  if [[ "${sub}" == "aishell-1" ]]; then
    train_set="train_${sub}"
    train_dev="dev_${sub}"
    if [[ "${specify_test_set}" == "" ]]; then
      test_set="dev_${sub} test_${sub}"
    else
      test_set=$specify_test_set
    fi

  else
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
        # "$@"
  done
done
