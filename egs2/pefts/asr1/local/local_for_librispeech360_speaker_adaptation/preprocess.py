# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

import sys

def load_wav_to_text_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            wav_id, text = line.strip().split(" ", 1)
            mapping[wav_id] = text
    return mapping

def get_text_by_wav_id(wav_id, mapping):
    return mapping.get(wav_id, None)

fin = open(sys.argv[1], "r")
fout_text = open(sys.argv[2], "w")
fout_utt2spk = open(sys.argv[3], "w")

for line in fin.readlines():
    # e.g., 
    # uttid=1053-289242-0000
    # path=/root/autodl-tmp/corpus_downloading_dir/Librispeech/LibriSpeech/train-clean-360/1053/289242/1053-289242-0000.flac
    # spkid=1053
    uttid, path = line.strip("\n").split("\t")
    speaker_id=uttid.split("-")[0]

    # text_path=/root/autodl-tmp/corpus_downloading_dir/Librispeech/LibriSpeech/train-clean-360/1053/289242/1053-289242.trans.txt
    text_path = path.rsplit("-", 1)[0]+".trans.txt"

    mapping=load_wav_to_text_mapping(text_path)
    text_ori=get_text_by_wav_id(uttid,mapping)
    assert text_ori is not None, "speaker_id={}; wav_id={}".format(speaker_id,uttid)

    fout_utt2spk.write(uttid + "\t" + speaker_id + "\n")
    fout_text.write(text_ori + "\n")
