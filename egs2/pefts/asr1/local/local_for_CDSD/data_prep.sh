#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group @ NWPU (Author: Xian Shi)
# Apache 2.0

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

corpus_path=$1     
data=$2         # data transformed into kaldi format
data_set=$3


if [[ "${data_set}" == *"normal"* ]]; then
    key=CDSD-normal
else
    key=CDSD
fi

 # raw_data is with metadata, txt and wav
if [[ "${data_set}" == *"partA"* ]]; then
    if [[ "${data_set}" == *"spk"* ]]; then
        spkid="${data_set: -2}"
        raw_data=${corpus_path}/${key}/1h/Audio/${spkid}
    else
        raw_data=${corpus_path}/${key}/1h
    fi
else # partB
    if [[ "${data_set}" == *"spk"* ]]; then
        spkid="${data_set: -2}"
        raw_data=${corpus_path}/${key}/10h/Audio/${spkid}
    else
        raw_data=${corpus_path}/${key}/10h
    fi
fi

# generate kaldi format data for all
if [ -d ${raw_data} ];then

    echo "Generating kaldi format data."
    mkdir -p $data/data_all
    find $raw_data -type f -name "*.wav" > $data/data_all/wavpath
    awk -F'/' '{print $(NF-2)"-"$(NF-1)"-"$NF}' $data/data_all/wavpath | sed 's:\.wav::g' > $data/data_all/uttlist
    paste $data/data_all/uttlist $data/data_all/wavpath > $data/data_all/wav.scp

    # faster than for in shell
    python local/local_for_CDSD/preprocess.py $data/data_all/wav.scp $data/data_all/trans $data/data_all/utt2spk

    ./utils/utt2spk_to_spk2utt.pl $data/data_all/utt2spk > $data/data_all/spk2utt

fi

# clean transcription
if [ -d $data/data_all ];then
    echo "Cleaning transcription."
    tr '[a-z]' '[A-Z]' < $data/data_all/trans > $data/data_all/trans_upper
    # turn "." in specific abbreviations into "<m>" tag
    sed -i -e 's: MR\.: MR<m>:g' -e 's: MRS\.: MRS<m>:g' -e 's: MS\.: MS<m>:g' \
        -e 's:^MR\.:MR<m>:g' -e 's:^MRS\.:MRS<m>:g' -e 's:^MS\.:MS<m>:g' $data/data_all/trans_upper
	# fix bug
    sed -i 's:^ST\.:STREET:g' $data/data_all/trans_upper
    sed -i 's: ST\.: STREET:g' $data/data_all/trans_upper
    # punctuation marks
    sed -i "s%,\|\.\|?\|!\|;\|-\|:\|,'\|\.'\|?'\|!'\| '% %g" $data/data_all/trans_upper
    sed -i 's:<m>:.:g' $data/data_all/trans_upper
    # blank
    sed -i 's:[ ][ ]*: :g' $data/data_all/trans_upper
    paste $data/data_all/uttlist $data/data_all/trans_upper > $data/data_all/text

    # critally, must replace tab with space between uttid and text
    sed -e "s/\t/ /g" -i $data/data_all/text
fi

echo "local/local_for_AESRC_and_CDSD/data_prep.sh succeeded"
exit 0;
