# ESPnet-pefts: ESPnet2 with parameter-efficient fine-tuning abilities for ASR

## 0 What's NEW ?
- I'm looking for a **postdoctoral position**, please contact me hqwz2618@163.com.

## 1 Key Features

- support various PEFT methods on both self-supervised pre-trained models and supervised models across corpora.

    > for PEFT methods, we now support full or partial model fine-tuning, [LoRA](https://arxiv.org/pdf/2106.09685v1/1000), [LanFusion](https://www.sciencedirect.com/science/article/pii/S0167639324000098), [CAM](https://www.sciencedirect.com/science/article/pii/S1566253524002847), VeLoRA/FasterVeLoRA.

    > for self-supervised pre-trained models, we now support hubert (A example of FT).

    > for supervised models, we now support whisper.

    > for corpora, we now support [AESRC2020](http://dx.doi.org/10.1109/ICASSP39728.2021.9413386), [CDSD](https://arxiv.org/pdf/2310.15930), [Librilight10](https://arxiv.org/pdf/1912.07875), [Librispeech](https://ieeexplore.ieee.org/document/7178964).

- todo list

    > update you github of fairseq, loralib, whisper
    
    > refined run_data_prepare.sh
    
## 2 Installation

#### 2.1 Requirements
- Python 3.7+
- gcc 4.9+ for PyTorch1.10.2+
- We use conda enviroment by default, otherwise you can refer to https://espnet.github.io/espnet/installation.html
- We do not install kaldi
- If there are any installation issues, please take a look at ./Z_IssuesList.md for possible solutions

#### 2.2 Installation ESPnet-pefts
1. Git clone ESPnet-pefts
``` bash
cd <any-place>
git clone https://github.com/WZLHQ/espnet.git
```
2. Setup conda environment
``` bash
cd <espnet-root>/tools
./setup_miniforge.sh miniconda espnet 3.8
```
3. Overwrite conda channels and then Make
```yaml
# 3.1 overwrite the root/.condarc with following contents:
channels:
  - https://software.repos.intel.com/python/conda/
  - conda-forge
  - defaults
show_channel_urls: true
```
``` bash
# 3.2 Make
# The Makefile tries to install ESPnet and all dependencies, including PyTorch.
# By default, the versions of pytorch and pytorch-cuda are 2.3.0 and 12.1, respectively.
cd <espnet-root>/tools
make
```
4. Install fairseq
```bash
# if you don't need finetune Meta SSL models, maybe you can skipe this step.
cd <espnet-root>/tools
./activate_python.sh
# In our implementation, pip==24.0 is needed for fairseq
pip install pip==24.0
./installers/install_fairseq.sh
pip install --upgrade pip
```
5. Install loralib
```bash
cd <espnet-root>/tools
./activate_python.sh
./installers/install_lora.sh
```
6. Install whisper
```bash
cd <espnet-root>/tools
./activate_python.sh
./installers/install_whisper.sh
# NOTE: after installation, you are supposed to checkout to branch "whisper-espnet-pefts" which is create from the tag v20230308
```
7. Checkout to LoraExpertTuning branch
```
All the code is only supported at LoraExpertTuning branch; in the future, we may merge it to the master branch.
```
## 3 Usage on VS Code
1. Open <espnet-root>/egs2/pefts/asr1 in VS Code
> we create a new recipe named "pefts" in which all the experiments are conducted

2. Checkout to LoraExpertTuning branch
> All the code is only supported at LoraExpertTuning branch; in the future, we may merge it to the master brach.

3. Set some soft links as follows for easy coding:
```bash
cd <espnet-root>/egs2/pefts/asr1
ln -s your/data/dir data
ln -s your/dump/dir dump
ln -s ../../../espnet pefts-espnet
ln -s ../../../espnet2 pefts-espnet2
ln -s ../../../tools/fairseq pefts-fairseq # if exists
ln -s ../../../tools/LoRA pefts-loralib
ln -s ../../../tools/whisper pefts-whisper
```
4. Data preparation
```bash
# There might be several path args that should be replaced with your own path.
# Please follow the instructions in ./run_data_prepare.sh
./run_data_prepare.sh
```

5. Training and inference

```
Two args that may cause confusion

    init_param: list
        It locates at ./conf/tuning/any_method/**.yaml.
        We use it to load backbone weights or multiple LoRA expert weights.
        For expert fusion methods, like SEF, DEF, and (Faster)VeLoRA, the first elmente denotes the backbone path.
        In our experiments on AESRC, we fine-tune the original whisper on Librispeech100 and use the fine-tuned whisper as the backbone.
        If you use the original whisper as backbone, you must have to replace it with the path of original whisper, e.g., "/root/.cache/whisper/base.pt"

    expert_path: str
        It locates at ./conf/tuning/expert_fusion_method/**.yaml.
        It denotes the expert path of your LoRA experts for expert fusion methods, like SEF, DEF, and (Faster)VeLoRA.
        You are supposed to specify your own expert path before training.

```

```sh
# Running example for each method on AESRC
domains=("US" "UK" "IND" "CHN" "JPN" "PT" "RU" "KR" "CA" "ES")
key=E1 # a unique name for current experiment
backbone=medium_en # selected from [base_en, medium_en]
for domain in "${domains[@]}"; do
    ./run_training_inference.sh "$domain" FT whisper $backbone $key 10 13 3 0 "--model_conf trainable_target_name=query.weight,key.weight,value.weight,out.weight,mlp.0.weight,mlp.2.weight" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA whisper $backbone $key 10 13 3 0 "--adapter_conf key_name=$domain" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4VeLoRA whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4FasterVeLoRA whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4CAT whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4LanFusion whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4PCAM whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4ECAM whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""

    # DictLoRA4MOLE and DictLoRA4SAMD denotes MoeLoRA and MoeLoRA*, respectively
    ./run_training_inference.sh "$domain" DictLoRA4MOLE whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
    ./run_training_inference.sh "$domain" DictLoRA4SAMD whisper $backbone $key 10 13 3 0 "--adapter_conf domain=$domain --adapter_conf Nbest=7best" espnet_outputs ""
done
```
