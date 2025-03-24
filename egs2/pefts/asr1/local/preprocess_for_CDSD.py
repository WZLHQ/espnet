# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

import sys

def load_wav_to_text_mapping(file_path):
    """
    将txt文件加载为一个字典，方便通过wav id快速获取对应的text。
    
    Args:
        file_path (str): txt文件路径，每行格式为 "wav_id text"
    
    Returns:
        dict: 键为wav id，值为对应的text。
    """
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            wav_id, text = line.strip().split(" ", 1)  # 只分割一次，防止text中包含空格
            mapping[wav_id] = text
    return mapping

def get_text_by_wav_id(wav_id, mapping):
    """
    根据wav id获取对应的text。
    
    Args:
        wav_id (str): 需要查询的wav id。
        mapping (dict): wav id到text的映射。
    
    Returns:
        str: 对应的text，如果没有找到返回None。
    """
    return mapping.get(wav_id, None)

fin = open(sys.argv[1], "r")
fout_text = open(sys.argv[2], "w")
fout_utt2spk = open(sys.argv[3], "w")

for line in fin.readlines():
    uttid, path = line.strip("\n").split("\t")
    '''
    uttid: Audio-19-S019T024E000N00022
    path: /media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/espnet_datasets/CDSD/CDSD-Interspeech/after_catting/1h/Audio/19/S019T024E000N00022.wav
    '''
    speaker_id=path.split("/")[-2]
    wav_id=path.split("/")[-1].rstrip(".wav")

    text_path,_=path.split("Audio/")
    text_path=text_path+"Text/"+speaker_id+"_label.txt"

    mapping=load_wav_to_text_mapping(text_path)
    text_ori=get_text_by_wav_id(wav_id,mapping)
    assert text_ori is not None, "speaker_id={}; wav_id={}".format(speaker_id,wav_id)

    feild = path.split("/")
    accid = feild[-4]
    spkid = accid + "-" + feild[-2]
    fout_utt2spk.write(uttid + "\t" + spkid + "\n")
    fout_text.write(text_ori + "\n")
