#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


#----------------------------run logs-------------------------------#
# ./run_training_inference.sh "CHN" FT whisper small_en A1 10 13 6 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_pre_trained_models/tuned_whisper/whisper_small_en_ATT-MLP-tuned_on_Librispeech100.pth --optim_conf lr=0.00001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "CHN" LoRA whisper small_en A1 10 13 6 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_pre_trained_models/tuned_whisper/whisper_small_en_ATT-MLP-tuned_on_Librispeech100.pth" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "CHN" ProxyTuning whisper small_en A1 12 13 6 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_pre_trained_models/tuned_whisper/whisper_small_en_ATT-MLP-tuned_on_Librispeech100.pth" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "CHN" ProxyTuning whisper small_en A2 11 13 6 1 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_pre_trained_models/tuned_whisper/whisper_small_en_ATT-MLP-tuned_on_Librispeech100.pth --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "CHN" ProxyTuning whisper small_en A3 11 13 6 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_pre_trained_models/tuned_whisper/whisper_small_en_ATT-MLP-tuned_on_Librispeech100.pth --scheduler_conf warmup_steps=500" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "CHN" ProxyTuning whisper small_en B1 11 13 6 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_pre_trained_models/tuned_whisper/whisper_small_en_ATT-MLP-tuned_on_Librispeech100.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "Librispeech100" FT whisper tiny_en A1 12 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning "0to7-combined_valid 0to7-combined_test"

# ./run_training_inference.sh "CHN" ProxyTuning whisper small_en C1 11 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "CHN" ProxyTuning whisper small_en C2 11 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.000005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "CHN" FT whisper tiny_en A1 10 13 6 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-tiny_en_Librispeech100-A1/valid.acc.ave_3best.pth --optim_conf lr=0.00001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "CHN" FT whisper tiny_en A2 11 13 6 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-tiny_en_Librispeech100-A1/valid.acc.ave_3best.pth --optim_conf lr=0.00002" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "CHN" FT whisper tiny_en A3 11 13 6 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-tiny_en_Librispeech100-A1/valid.acc.ave_3best.pth --optim_conf lr=0.000005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "CHN" ProxyTuning whisper small_en D1 12 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "CHN" ProxyTuning whisper small_en E1 12 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "CHN" ProxyTuning whisper small_en E2 11 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "CHN" FT whisper largeV2 A1 11 13 6 0 "" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "CHN" ProxyTuning whisper largeV2 A1 12 13 1 0 "" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "CHN" LoRA whisper largeV2 A1 12 13 1 1 "" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en A1 11 13 5 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en A2 11 13 5 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en A1-1 11 13 5 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en A1-2 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" FT whisper tiny_en B1 10 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-tiny_en_Librispeech100-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" FT whisper tiny_en B2 10 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet/egs2/pefts/asr1/espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-tiny_en_Librispeech100-A1/valid.acc.ave_3best.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en B1 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en B2 11 13 5 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "Librispeech100" FT whisper base_en A1 11 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning "Librispeech100_valid_clean Librispeech100_test_clean 0to7-combined_valid 0to7-combined_test"
# ./run_training_inference.sh "Librispeech100" FT whisper base_en A2 11 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning "Librispeech100_valid_clean Librispeech100_test_clean 0to7-combined_valid 0to7-combined_test"
# ./run_training_inference.sh "Librispeech100" FT whisper base_en A3 11 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning "Librispeech100_valid_clean Librispeech100_test_clean 0to7-combined_valid 0to7-combined_test"

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en Y1 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_pre_trained_models/tuned_whisper/whisper_small_en_ATT-MLP-tuned_on_Librispeech100.pth --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en Y2 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en Y3 11 13 5 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en Y4 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" FT whisper base_en B1 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_pre_trained_models/tuned_whisper/whisper_base_en_ATT-MLP-tuned_on_Librispeech100.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" FT whisper base_en B2 11 13 5 0 "--init_param /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_pre_trained_models/tuned_whisper/whisper_base_en_ATT-MLP-tuned_on_Librispeech100.pth --model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en C1 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en C2 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en use_AESRC_tuned_tiny_model 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en use_AESRC_tuned_tiny_model_1 11 13 5 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en use_AESRC_tuned_tiny_model_2 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001 --optim_conf weight_decay=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en C3 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001 --decoder_conf use_adapterH=true --decoder_conf adapter_dim=256" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en A1 10 13 5 0 "--model_conf trainable_target_name=adapter" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en A2 11 13 5 0 "--model_conf trainable_target_name=adapter" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en B1 11 13 5 1 "--model_conf trainable_target_name=adapter" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" AdapterH whisper small_en B2 11 13 5 1 "--model_conf trainable_target_name=adapter" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en C4 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en E1 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en E2 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en E3 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en E4 11 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en D2 12 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en D1 12 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.00005" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en D3 12 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001 --optim_conf weight_decay=0.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en D4 12 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001 --optim_conf weight_decay=1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""

# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en F1 12 13 5 1 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --model_conf proxy_logits_weight=1 --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001 --optim_conf weight_decay=0.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en D5 12 13 5 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --model_conf proxy_logits_weight=0.3 --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001 --optim_conf weight_decay=0.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""
# ./run_training_inference.sh "0to7-combined" ProxyTuning whisper small_en D6 11 13 5 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight,decoder_fusion_module --model_conf proxy_logits_weight=0.7 --decoder_conf return_layer_results=true --proxy_decoder_conf use_decoder_fusion_module=true --optim_conf lr=0.0001 --optim_conf weight_decay=0.1" /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_outputs_ProxyTuning ""


# select:
# [CDSD-partA, CDSD-partB] from CDSD
# [Librilight10, Librispeech100]
# [US UK IND CHN JPN PT RU KR CA ES] from AESRC2020
# "0to7-combined" denotes combining the first 7 accents
# "0to8-combined" denotes combining "0to7-combined" with Librispeech100
# aishell-1
subcorpus=$1

# select a method from [FT, LoRA, AdapterH, ProxyTuning]
# TODO add LoRA variants, LanFusion, CAM, and so on.
method=$2

if [[ "${method}" == 'ProxyTuning' ]]; then
  use_proxy_tuning=true
else
  use_proxy_tuning=false
fi

# select: 
# [whisper, hubert]
model=$3
# select: 
# [base, large] for hubert
# [small, medium, small_en, medium_en, largeV2] for Whisper
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
        --use_proxy_tuning ${use_proxy_tuning}             \
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
