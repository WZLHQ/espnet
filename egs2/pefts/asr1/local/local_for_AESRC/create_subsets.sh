#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

data=$1     # data transformed into kaldi format
recog_set=$2

# divide development set for cross validation
if [ -d ${data} ];then
    echo "------------------split methods-----------------"
    if [[ "${recog_set}" == "CDSD-partB" ]]; then
        # for CDSD-partB (totally 80 hours, roughly 92000 audio), we split the train-dev-test by 76h-2h-2h (roughly)
        # and we promise there is no overlap among train, dev and test(or called cv)
        sed -n '0~40p' $data/data_all/wav.scp > $data/cv.scp
        ./utils/filter_scp.pl --exclude $data/cv.scp $data/data_all/wav.scp > $data/train_and_dev.scp
        sed -n '0~40p' $data/train_and_dev.scp > $data/dev.scp
        ./utils/filter_scp.pl --exclude $data/dev.scp $data/train_and_dev.scp > $data/train.scp
        ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/${recog_set}_train
        ./utils/subset_data_dir.sh --utt-list $data/dev.scp $data/data_all $data/${recog_set}_valid
        ./utils/subset_data_dir.sh --utt-list $data/cv.scp $data/data_all $data/${recog_set}_test

    else
        # for CDSD-partA and AESRC, we specify test set from the given speaker list (at local/local_for_AESRC_and_CDSD/files/cvlist/**)
        # and split the train-dev sets by 95-5
        ./utils/subset_data_dir.sh --spk-list local/local_for_AESRC_and_CDSD/files/cvlist/${recog_set}_cv_spk $data/data_all $data/cv/$recog_set
        cat $data/cv/$recog_set/wav.scp >> $data/cv.scp

        ./utils/filter_scp.pl --exclude $data/cv.scp $data/data_all/wav.scp > $data/train_and_dev.scp
        #95-5 split for dev set
        sed -n '0~20p' $data/train_and_dev.scp > $data/dev.scp
        ./utils/filter_scp.pl --exclude $data/dev.scp $data/train_and_dev.scp > $data/train.scp
        ./utils/subset_data_dir.sh --utt-list $data/train.scp $data/data_all $data/${recog_set}_train
        ./utils/subset_data_dir.sh --utt-list $data/dev.scp $data/data_all $data/${recog_set}_valid
        ./utils/subset_data_dir.sh --utt-list $data/cv.scp $data/data_all $data/${recog_set}_test
    fi
fi

echo "local/local_for_AESRC_and_CDSD/subset_data.sh succeeded"


# if [[ "${recog_set}" == *"combined"* ]]; then
#     accents="US UK IND CHN JPN PT RU KR" # this determined by $data_sets defined in data.sh
#     for i in ${accents};do
#         ./utils/subset_data_dir.sh --spk-list local/local_for_AESRC_and_CDSD/files/cvlist/${i}_cv_spk $data/data_all $data/cv/$i
#         cat $data/cv/$i/wav.scp >> $data/cv.scp
#     done
# else
#     ./utils/subset_data_dir.sh --spk-list local/local_for_AESRC_and_CDSD/files/cvlist/${recog_set}_cv_spk $data/data_all $data/cv/$recog_set
#     cat $data/cv/$recog_set/wav.scp >> $data/cv.scp
# fi
