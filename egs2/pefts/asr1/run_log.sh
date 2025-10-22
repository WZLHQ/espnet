#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
set -u
set -o pipefail

#----------------------------run logs-------------------------------#
# ./run_training_inference.sh "Librispeech100" FT whisper base_en A1 11 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" espnet_outputs "Librispeech_valid_clean Librispeech_test_clean IND_valid IND_test US_valid US_test UK_valid UK_test CHN_valid CHN_test JPN_valid JPN_test PT_valid PT_test RU_valid RU_test KR_valid KR_test CA_valid CA_test ES_valid ES_test"
# ./run_training_inference.sh "Librispeech100" FT whisper base_en A2 11 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --optim_conf lr=0.00005" espnet_outputs "Librispeech_valid_clean Librispeech_test_clean IND_valid IND_test US_valid US_test UK_valid UK_test CHN_valid CHN_test JPN_valid JPN_test PT_valid PT_test RU_valid RU_test KR_valid KR_test CA_valid CA_test ES_valid ES_test"

# ./run_training_inference.sh "US" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=US" espnet_outputs ""; ./run_training_inference.sh "UK" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=UK" espnet_outputs ""; ./run_training_inference.sh "IND" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=IND" espnet_outputs ""; ./run_training_inference.sh "CHN" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=CHN" espnet_outputs ""; ./run_training_inference.sh "JPN" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=JPN" espnet_outputs "";
# ./run_training_inference.sh "PT" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=PT" espnet_outputs ""; ./run_training_inference.sh "RU" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=RU" espnet_outputs ""; ./run_training_inference.sh "KR" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=KR" espnet_outputs ""; ./run_training_inference.sh "CA" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=CA" espnet_outputs ""; ./run_training_inference.sh "ES" DictLoRA whisper base_en A0 11 13 6 0 "--adapter_conf key_name=ES" espnet_outputs "";

# ./run_training_inference.sh "US" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=US --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs ""; ./run_training_inference.sh "UK" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=UK --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs ""; ./run_training_inference.sh "IND" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=IND --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs ""; ./run_training_inference.sh "CHN" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=CHN --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs ""; ./run_training_inference.sh "JPN" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=JPN --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs "";
# ./run_training_inference.sh "PT" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=PT --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs ""; ./run_training_inference.sh "RU" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=RU --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs ""; ./run_training_inference.sh "KR" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=KR --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs ""; ./run_training_inference.sh "CA" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=CA --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs ""; ./run_training_inference.sh "ES" DictLoRA whisper base_en A2 11 13 6 0 "--adapter_conf key_name=ES --adapter_conf rank=64 --adapter_conf alpha=64" espnet_outputs "";

# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4LanFusion whisper base_en A1 12 13 6 0 "" espnet_outputs ""

# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en B1 12 13 6 0 "--optim_conf lr=0.01" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en B2 12 13 6 0 "--optim_conf lr=0.1" espnet_outputs ""

# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en C1 11 13 6 0 "--optim_conf lr=0.01" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en C2 11 13 6 0 "--optim_conf lr=0.1" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en C0 11 13 6 0 "--optim_conf lr=0.001" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en C0a 11 13 6 0 "--optim_conf lr=0.005" espnet_outputs ""

# ./run_training_inference.sh "UK PT RU CA" DictLoRA4SELoRA whisper base_en B-A1 10 13 6 0 "--optim_conf lr=0.1" espnet_outputs ""

# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en B-Test-1 11 13 6 0 "" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en B3 11 13 6 0 "--optim_conf lr=0.1" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en B4 11 13 6 0 "--optim_conf lr=0.1" espnet_outputs ""

# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en A1 11 13 6 0 "--optim_conf lr=0.1" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en A2 11 13 6 0 "--optim_conf lr=0.01" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en A3 11 13 6 0 "--optim_conf lr=0.001" espnet_outputs ""

# ./run_training_inference.sh "RU KR CA ES" DictLoRA4SELoRA whisper base_en A3 11 13 6 0 "--optim_conf lr=0.001" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en B2-TEST 11 13 6 0 "--optim_conf lr=0.1" espnet_outputs ""
# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4SELoRA whisper base_en B2-TEST-1 11 13 6 0 "--optim_conf lr=0.1" espnet_outputs ""

# ./run_training_inference.sh "IND CHN PT RU KR CA ES" DictLoRA4ReLoRA whisper base_en A1 10 13 6 0 "" espnet_outputs ""

# ./run_training_inference.sh "US" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=US --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "UK_valid IND_valid CHN_valid JPN_valid PT_valid RU_valid KR_valid CA_valid ES_valid"
# ./run_training_inference.sh "UK" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=UK --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "US_valid IND_valid CHN_valid JPN_valid PT_valid RU_valid KR_valid CA_valid ES_valid"
# ./run_training_inference.sh "IND" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=IND --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "US_valid UK_valid CHN_valid JPN_valid PT_valid RU_valid KR_valid CA_valid ES_valid"
# ./run_training_inference.sh "CHN" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=CHN --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "US_valid UK_valid IND_valid JPN_valid PT_valid RU_valid KR_valid CA_valid ES_valid"
# ./run_training_inference.sh "JPN" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=JPN --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "US_valid UK_valid IND_valid CHN_valid PT_valid RU_valid KR_valid CA_valid ES_valid"
# ./run_training_inference.sh "PT" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=PT --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "US_valid UK_valid IND_valid CHN_valid JPN_valid RU_valid KR_valid CA_valid ES_valid"
# ./run_training_inference.sh "RU" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=RU --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "US_valid UK_valid IND_valid CHN_valid JPN_valid PT_valid KR_valid CA_valid ES_valid"
# ./run_training_inference.sh "KR" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=KR --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "US_valid UK_valid IND_valid CHN_valid JPN_valid PT_valid RU_valid CA_valid ES_valid"
# ./run_training_inference.sh "CA" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=CA --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "US_valid UK_valid IND_valid CHN_valid JPN_valid PT_valid RU_valid KR_valid ES_valid"
# ./run_training_inference.sh "ES" DictLoRA whisper base_en A1 12 13 6 0 "--adapter_conf key_name=ES --adapter_conf rank=32 --adapter_conf alpha=32" espnet_outputs "US_valid UK_valid IND_valid CHN_valid JPN_valid PT_valid RU_valid KR_valid CA_valid"

# ./run_training_inference.sh "US" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=US" espnet_outputs ""; ./run_training_inference.sh "UK" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=UK" espnet_outputs "";
# ./run_training_inference.sh "IND" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=IND" espnet_outputs ""; ./run_training_inference.sh "CHN" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=CHN" espnet_outputs "";
# ./run_training_inference.sh "JPN" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=JPN" espnet_outputs ""; ./run_training_inference.sh "PT" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=PT" espnet_outputs "";
# ./run_training_inference.sh "RU" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=RU" espnet_outputs ""; ./run_training_inference.sh "KR" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=KR" espnet_outputs "";
# ./run_training_inference.sh "CA" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=CA" espnet_outputs ""; ./run_training_inference.sh "ES" DictLoRA4ReLoRA whisper base_en B1 10 13 6 0 "--adapter_conf domain=ES" espnet_outputs "";

# ./run_training_inference.sh "US" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=US --optim_conf lr=0.05" espnet_outputs ""; ./run_training_inference.sh "UK" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=UK --optim_conf lr=0.05" espnet_outputs "";
# ./run_training_inference.sh "IND" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=IND --optim_conf lr=0.05" espnet_outputs ""; ./run_training_inference.sh "CHN" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=CHN --optim_conf lr=0.05" espnet_outputs "";
# ./run_training_inference.sh "JPN" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=JPN --optim_conf lr=0.05" espnet_outputs ""; ./run_training_inference.sh "PT" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=PT --optim_conf lr=0.05" espnet_outputs "";
# ./run_training_inference.sh "RU" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=RU --optim_conf lr=0.05" espnet_outputs ""; ./run_training_inference.sh "KR" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=KR --optim_conf lr=0.05" espnet_outputs "";
# ./run_training_inference.sh "CA" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=CA --optim_conf lr=0.05" espnet_outputs ""; ./run_training_inference.sh "ES" DictLoRA4ReLoRA whisper base_en B2 10 13 6 0 "--adapter_conf domain=ES --optim_conf lr=0.05" espnet_outputs "";

# ./run_training_inference.sh "US" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=US --optim_conf lr=0.01" espnet_outputs ""; ./run_training_inference.sh "UK" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=UK --optim_conf lr=0.01" espnet_outputs "";
# ./run_training_inference.sh "IND" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=IND --optim_conf lr=0.01" espnet_outputs ""; ./run_training_inference.sh "CHN" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=CHN --optim_conf lr=0.01" espnet_outputs "";
# ./run_training_inference.sh "JPN" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=JPN --optim_conf lr=0.01" espnet_outputs ""; ./run_training_inference.sh "PT" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=PT --optim_conf lr=0.01" espnet_outputs "";
# ./run_training_inference.sh "RU" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=RU --optim_conf lr=0.01" espnet_outputs ""; ./run_training_inference.sh "KR" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=KR --optim_conf lr=0.01" espnet_outputs "";
# ./run_training_inference.sh "CA" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=CA --optim_conf lr=0.01" espnet_outputs ""; ./run_training_inference.sh "ES" DictLoRA4ReLoRA whisper base_en B3 10 13 6 0 "--adapter_conf domain=ES --optim_conf lr=0.01" espnet_outputs "";

# ./run_training_inference.sh "US" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=US --optim_conf lr=0.008" espnet_outputs ""; ./run_training_inference.sh "UK" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=UK --optim_conf lr=0.008" espnet_outputs "";
# ./run_training_inference.sh "IND" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=IND --optim_conf lr=0.008" espnet_outputs ""; ./run_training_inference.sh "CHN" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=CHN --optim_conf lr=0.008" espnet_outputs "";
# ./run_training_inference.sh "JPN" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=JPN --optim_conf lr=0.008" espnet_outputs ""; ./run_training_inference.sh "PT" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=PT --optim_conf lr=0.008" espnet_outputs "";
# ./run_training_inference.sh "RU" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=RU --optim_conf lr=0.008" espnet_outputs ""; ./run_training_inference.sh "KR" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=KR --optim_conf lr=0.008" espnet_outputs "";
# ./run_training_inference.sh "CA" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=CA --optim_conf lr=0.008" espnet_outputs ""; ./run_training_inference.sh "ES" DictLoRA4ReLoRA whisper base_en B0 10 13 6 0 "--adapter_conf domain=ES --optim_conf lr=0.008" espnet_outputs "";

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper base_en B0-1 10 13 6 0 "--adapter_conf domain=$domain --optim_conf lr=0.008" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper base_en B1-1 11 13 6 0 "--adapter_conf domain=$domain" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper base_en R32_5best_B1 11 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_5best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper base_en R32_7best_B1 11 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper base_en R32_5best_B2 11 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_5best --optim_conf lr=0.008" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper base_en R32_7best_B2 11 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best --optim_conf lr=0.008" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper base_en R32_1best_B1 11 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_1best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper base_en R32_1best_A1 12 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_1best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper base_en R32_3best_A1 12 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_3best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper base_en R32_7best_A1 12 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper base_en R32_5best_A1 12 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_5best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper base_en R32_1best_A1 10 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_1best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper base_en R32_3best_A1 10 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_3best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper base_en R32_7best_A1 10 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper base_en R32_5best_A1 10 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_5best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper base_en R32_3best_A1 10 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_3best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper base_en R32_5best_A1 11 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_5best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper base_en R32_7best_A1 11 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" FT whisper base_en A1 10 13 6 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper base_en R32_7best_B2-1 11 13 6 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best --optim_conf lr=0.008" espnet_outputs ""
# done

# domains=("Librispeech100")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" FT whisper medium_en A1 12 13 1 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --max_epoch 4" espnet_outputs "Librispeech_valid_clean Librispeech_test_clean IND_test US_test UK_test CHN_test JPN_test PT_test RU_test KR_test CA_test ES_test"
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA whisper medium_en A1 11 13 2 0 "--init_param espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-medium_en_Librispeech100-A1/valid.acc.ave_3best.pth --adapter_conf key_name=$domain" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" FT whisper medium_en A1 11 13 1 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --init_param espnet_outputs/Librispeech100_whisper_FT_outputs/asr_FT_whisper-medium_en_Librispeech100-A1/valid.acc.ave_3best.pth" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper medium_en R32_7best_A1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper medium_en R32_7best_A1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper medium_en R32_7best_A1 12 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper medium_en R32_7best_A1 11 13 2 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper medium_en R32_7best_B1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper medium_en R32_7best_A2 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper medium_en R32_7best_A2 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper medium_en R32_7best_A2 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper medium_en R32_7best_A2 11 13 2 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper medium_en R32_7best_A3 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best --optim_conf lr=0.05" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper medium_en R32_7best_B2 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper medium_en R32_7best_C1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4ReLoRA whisper medium_en R32_7best_A2-1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4MoELoRA whisper medium_en R32_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4MoELoRA whisper medium_en R32_7best_B1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best --adapter_conf is_experts_trainable=false" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4MoELoRA whisper base_en R32_7best_A1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4MoELoRA whisper base_en R32_7best_B1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best --adapter_conf is_experts_trainable=false" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4MoELoRA whisper medium_en R32_7best_A2 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best --optim_conf lr=0.00005" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4MoELoRA whisper medium_en R32_7best_B2 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best --adapter_conf is_experts_trainable=false --optim_conf lr=0.005" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4FasterReLoRA whisper base_en R32_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done


# the following code is to compute the effciency on US accent.
# batch size=8 and max_epoch=2 for all methods
# model=("base_en")
# for m in "${model[@]}"; do
#     ./run_training_inference.sh "US" FT whisper $m US_efficiency 10 13 3 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
#     ./run_training_inference.sh "US" DictLoRA whisper $m US_efficiency 10 11 3 0 "--adapter_conf key_name=US --adapter_conf rank=32 --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
#     ./run_training_inference.sh "US" DictLoRA4MOLE whisper $m US_efficiency 10 11 3 0 "--adapter_conf domain=US --adapter_conf R_Nbest=R32_7best --optim_conf lr=0.005 --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
#     ./run_training_inference.sh "US" DictLoRA4CAT whisper $m US_efficiency 10 11 3 0 "--adapter_conf domain=US --adapter_conf R_Nbest=R32_7best --optim_conf lr=0.005 --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
#     ./run_training_inference.sh "US" DictLoRA4LanFusion whisper $m US_efficiency 10 11 3 0 "--adapter_conf domain=US --adapter_conf R_Nbest=R32_7best --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
#     ./run_training_inference.sh "US" DictLoRA4PCAM whisper $m US_efficiency 10 11 3 0 "--adapter_conf domain=US --adapter_conf R_Nbest=R32_7best --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
#     ./run_training_inference.sh "US" DictLoRA4ECAM whisper $m US_efficiency 10 11 3 0 "--adapter_conf domain=US --adapter_conf R_Nbest=R32_7best --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
#     ./run_training_inference.sh "US" DictLoRA4SAMD whisper $m US_efficiency 10 11 3 0 "--adapter_conf domain=US --adapter_conf R_Nbest=R32_7best --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
#     ./run_training_inference.sh "US" DictLoRA4VeLoRA whisper $m US_efficiency 10 11 3 0 "--adapter_conf domain=US --adapter_conf R_Nbest=R32_7best --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
#     ./run_training_inference.sh "US" DictLoRA4FasterVeLoRA whisper $m US_efficiency 10 11 3 0 "--adapter_conf domain=US --adapter_conf R_Nbest=R32_7best --batch_size=8 --valid_batch_size=8 --max_epoch=2" espnet_outputs ""
# done

# domains=("CHN" "PT" "CA")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" VeRA whisper base_en A1_lr_1e-2_rank_32_new 11 13 3 0 "--optim_conf lr=0.01" espnet_outputs ""
# done

# domains=("CHN" "PT" "CA")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper base_en motivation_experiments_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_1best" espnet_outputs ""
# done



# create running template for each method
# domains=("US")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" FT whisper medium_en comput_RTF 11 13 3 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA whisper medium_en comput_RTF 11 13 3 0 "--adapter_conf key_name=$domain" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper medium_en comput_RTF 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper medium_en comput_RTF 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4CAT whisper medium_en comput_RTF 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper medium_en comput_RTF 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper medium_en comput_RTF 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
#     ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper medium_en comput_RTF 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper base_en R32_7best_B2-3 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for domain in "${domains[@]}"; do
#     ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper medium_en R32_7best_B1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=R32_7best" espnet_outputs ""
# done

# keys=("R32_1best" "R32_3best")
# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for k in "${keys[@]}"; do
#     for domain in "${domains[@]}"; do
#         ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper base_en ${k}_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=$k --optim_conf lr=0.008 --keep_nbest_models=2 --max_epoch=3" espnet_outputs ""
#     done
# done

# keys=("R32_7best")
# domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
# for k in "${keys[@]}"; do
#     for domain in "${domains[@]}"; do
#         ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper base_en ${k}_A2 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf R_Nbest=$k --keep_nbest_models=2" espnet_outputs ""
#     done
# done

domains=("UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" DictLoRA4CAT whisper base_en R8_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=8 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A0/valid.acc.ave_3best.pth" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper base_en R8_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=8 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A0/valid.acc.ave_3best.pth" espnet_outputs ""
    # ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper base_en R8_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=8 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A0/valid.acc.ave_3best.pth" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper base_en R8_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=8 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A0/valid.acc.ave_3best.pth" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper base_en R8_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=8 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A0/valid.acc.ave_3best.pth" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper base_en R8_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=8 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A0/valid.acc.ave_3best.pth" espnet_outputs ""
done

domains=("UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" DictLoRA4CAT whisper base_en R64_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=64 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A2/valid.acc.ave_3best.pth" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper base_en R64_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=64 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A2/valid.acc.ave_3best.pth" espnet_outputs ""
    # ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper base_en R64_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=64 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A2/valid.acc.ave_3best.pth" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper base_en R64_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=64 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A2/valid.acc.ave_3best.pth" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper base_en R64_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=64 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A2/valid.acc.ave_3best.pth" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper base_en R64_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=64 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A2/valid.acc.ave_3best.pth" espnet_outputs ""
done

sleep 21600
domains=("UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper base_en R8_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=8 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A0/valid.acc.ave_3best.pth" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper base_en R64_7best_A1 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf rank=64 --adapter_conf expert_path=espnet_outputs/{}_whisper_DictLoRA_outputs/asr_DictLoRA_whisper-{}_en_{}-A2/valid.acc.ave_3best.pth" espnet_outputs ""
done


domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper base_en R32_7best_C1-1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf initial_type=vera" espnet_outputs ""
    # ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper base_en R32_7best_C2 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf initial_type=kaiming" espnet_outputs ""
done

domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
for domain in "${domains[@]}"; do
    # ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper base_en R32_7best_C1 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf initial_type=vera" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper base_en R32_7best_C2 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --adapter_conf initial_type=kaiming" espnet_outputs ""
done

domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper base_en R32_7best_B3 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --optim_conf lr=0.0005" espnet_outputs ""
done

domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper base_en R32_7best_B4 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
done

domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper base_en R32_7best_A3 11 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
done

domains=("US")
key=compute_RTF_1
backbone=medium_en
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" FT whisper $backbone $key 10 11 3 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA whisper $backbone $key 10 11 3 0 "--adapter_conf key_name=$domain --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper $backbone $key 10 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper $backbone $key 10 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4CAT whisper $backbone $key 10 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper $backbone $key 10 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper $backbone $key 10 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper $backbone $key 10 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
done

domains=("US")
key=compute_RTF_1
backbone=base_en
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" DictLoRA4MOLE whisper $backbone $key 10 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4SAMD whisper $backbone $key 10 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
done

domains=("US")
key=compute_memory_time
backbone=medium_en
for domain in "${domains[@]}"; do
    # ./run_training_inference.sh "$domain" DictLoRA4MOLE whisper $backbone $key 11 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --optim_conf lr=0.005 --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4SAMD whisper $backbone $key 11 11 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best --optim_conf lr=0.00005 --batch_size=8 --valid_batch_size=6 --max_epoch=4" espnet_outputs ""
done
