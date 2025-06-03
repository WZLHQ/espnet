#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

data=$1     # data transformed into kaldi format
recog_set=$2

if [ -d ${data} ];then

    echo "------------------split methods-----------------"

    if [[ "${recog_set}" == *"partA"* ]]; then
        if [[ "${recog_set}" == *"spk"* ]]; then
            key='0~10p'
        else
            key='0~20p'
        fi
    else # partB
        if [[ "${recog_set}" == *"spk"* ]]; then
            key='0~10p'
        else
            key='0~40p'
        fi
    fi

    # and we promise there is no overlap among train, dev and test(or called cv)
    sed -n "${key}" $data/data_all/wav.scp > $data/cv.scp
    ./utils/filter_scp.pl --exclude $data/cv.scp $data/data_all/wav.scp > $data/train_and_dev.scp
    sed -n "${key}" $data/train_and_dev.scp > $data/dev.scp
    ./utils/filter_scp.pl --exclude $data/dev.scp $data/train_and_dev.scp > $data/train.scp
    ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/${recog_set}_train
    ./utils/subset_data_dir.sh --utt-list $data/dev.scp $data/data_all $data/${recog_set}_valid
    ./utils/subset_data_dir.sh --utt-list $data/cv.scp $data/data_all $data/${recog_set}_test

fi

echo "local/local_for_CDSD/subset_data.sh succeeded"
