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

echo "Assume that you have prepared the raw data"

for data_set in ${data_sets}; do

    spkid="${data_set#*spk}"
    raw_data=${corpus_path}/${spkid}
    local/local_for_librispeech360_speaker_adaptation/data_prep.sh ${raw_data} ${data}
    ./utils/fix_data_dir.sh ${data}/data_all
    local/local_for_librispeech360_speaker_adaptation/create_subsets.sh ${data} "${data_set}"
    rm -rf ${data}/data_all
    rm -rf ${data}/cv.scp ${data}/train_and_dev.scp ${data}/dev.scp ${data}/train.scp ${data}/cv

done

log "Successfully finished. [elapsed=${SECONDS}s]"
