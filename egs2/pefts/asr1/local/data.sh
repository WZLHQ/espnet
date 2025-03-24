#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

corpus=$1
corpus_path=$2 # the root path of your raw datasets
# for CDSD, select from ["CDSD-partA", "CDSD-partA CDSD-partB"]
# for AESRC, select from "US UK IND CHN JPN PT RU KR CA ES"
data_sets=$3
stage=1
stop_stage=100000
data=data

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [[ "${corpus}" == *"CDSD"* ]] || [[ "${corpus}" == *"AESRC"* ]]; then

    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        # we assume that you have already download and unzip the *.zip file
        # please apply the CDSD data at https://arxiv.org/pdf/2310.15930
        # please apply the AESRC data at ***
        echo "data already been unzipped"
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

        echo "Data preparation"

        for data_set in ${data_sets};do

            if [[ "${corpus}" == *"CDSD"* ]]; then
                
                if [[ "${data_set}" == *"partA"* ]]; then # for partA
                    local/data_prep.sh ${corpus_path}/CDSD-Interspeech/after_catting/1h ${data} ${corpus}
                elif [[ "${data_set}" == *"partB"* ]]; then # for partB
                    local/data_prep.sh ${corpus_path}/CDSD-Interspeech/after_catting/10h ${data} ${corpus}
                fi

            elif [[ "${corpus}" == *"AESRC"* ]]; then
                local/data_prep.sh ${corpus_path}/Datatang-English/data/${data_set} ${data} ${corpus}
            fi

            ./utils/fix_data_dir.sh ${data}/data_all
            local/create_subsets.sh ${data} "${data_set}"
            rm -rf ${data}/data_all
            rm -rf ${data}/cv.scp ${data}/train_and_dev.scp ${data}/dev.scp ${data}/train.scp ${data}/cv

        done
    fi

elif [[ "${corpus}" == *"**"* ]]; then

    echo ""

fi


log "Successfully finished. [elapsed=${SECONDS}s]"
