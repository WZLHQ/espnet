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


stage=1
stop_stage=100000
data_url=www.openslr.org/resources/12
data_sets=$1

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Data Download to ${LIBRISPEECH}"
	for part in train-clean-360; do
            local/local_for_librispeech360/download_and_untar.sh ${LIBRISPEECH} ${data_url} ${part}
	done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    for part in train-clean-360; do
        # use underscore-separated names in data directories.
        new_part=${part//train-clean-360/train}
        local/local_for_librispeech360/data_prep.sh ${LIBRISPEECH}/LibriSpeech/${part} data/${data_sets}_${new_part//-/_}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
