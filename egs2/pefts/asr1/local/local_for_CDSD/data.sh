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

corpus_path=$1
data_sets=$2
stage=1
stop_stage=100000
data=data

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # we assume that you have already download and unzip the *.zip file
    # please apply the CDSD data at https://arxiv.org/pdf/2310.15930
    echo "data already been unzipped"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    echo "Data preparation"

    for data_set in ${data_sets}; do

        local/local_for_CDSD/data_prep.sh ${corpus_path} ${data} ${data_set}

        ./utils/fix_data_dir.sh ${data}/data_all
        local/local_for_CDSD/create_subsets.sh ${data} "${data_set}"
        rm -rf ${data}/data_all
        rm -rf ${data}/cv.scp ${data}/train_and_dev.scp ${data}/dev.scp ${data}/train.scp ${data}/cv

    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
