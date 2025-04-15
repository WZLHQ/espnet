# ESPnet-pefts: ESPnet with parameter-efficient fine-tuning abilities for ASR

## 1 Key Features

- support various PEFT methods on both self-supervised pre-trained models and supervised models across various corpora.

    > for PEFT methods, we now support full or partial model fine-tuning, LoRA, and houslby Adapter. 

    > for self-supervised pre-trained models, we now support hubert.

    > for supervised models, we now support whisper.

    > for corpora, we now support [**CDSD**](https://arxiv.org/pdf/2310.15930) and [**Librilight10**](https://arxiv.org/pdf/1912.07875). If you need CDSD, you should apply the data first at http://melab.psych.ac.cn/CDSD.html.

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
# This script tries to create a new miniconda if the output directory doesn't exist.
# We use python=3.8 for our experiments.
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
cd <espnet-root>/tools
make
# The Makefile tries to install ESPnet and all dependencies, including PyTorch.
# By default, the versions of pytorch and pytorch-cuda are 2.3.0 and 12.1, respectively.
# In our experiments, we adopt this default settings.
```
4. Install fairseq
```bash
cd <espnet-root>/tools
./activate_python.sh
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
6. Replace whisper
- First, we should install whisper by:
```bash
cd <espnet-root>/tools
./activate_python.sh && ./installers/install_whisper.sh
```
- (TODO) Then, we replace some files with **.

## 3 Usage on VS Code
In the master branch, we create a new recipe named "pefts" in which all the experiments are conducted. So the usage is as follows:
1. Open <espnet-root>/egs2/pefts/asr1 in VS Code
2. Set soft links as follows:
```bash
cd <espnet-root>/egs2/pefts/asr1
ln -s your/data/dir data
ln -s your/dump/dir dump
ln -s ../../../espnet pefts-espnet
ln -s ../../../espnet2 pefts-espnet2
ln -s ../../../tools/fairseq pefts-fairseq
ln -s ../../../tools/LoRA pefts-loralib
ln -s ../../../tools/miniconda/envs/espnet/lib/python3.8/site-packages/whisper pefts-whisper
# if exits
ln -s ../../../tools/miniconda/envs/espnet/lib/python3.8/site-packages/peft pefts-peft
ln -s ../../../tools/miniconda/envs/espnet/lib/python3.8/site-packages/transformers pefts-transformers
# TODO: why the soft links should be set?
```
3. Corpus preparation
```bash
# There might be several path args that should be replaced with your own path.
# Therefor, it would be better that you have already been familiar with 1) what the first stage of ESPnet do; 2) what the ${local_data_dir}/data.sh do.
./run_data_prepare.sh
```
4. Training and inference
```bash
# run_training_inference.sh allows you specify the corpus, PEFT method, model and its size, key (a unique tag for current experiments) and so on.
# It might be a lit bit hard to understand, so you are suggested to figure out the whole pipline through an example below. 
# After you truely understand how it works, you can do the any combination among corpus, model, and method.
./run_training_inference.sh "Librilight10" LoRA hubert base "A0" 10 13 4 0 "--adapter_conf rank=8 --adapter_conf alpha=8" /*/espnet_outputs
```
