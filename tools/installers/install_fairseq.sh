#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi
torch_version=$(python3 -c "import torch; print(torch.__version__)")
python_36_plus=$(python3 <<EOF
from packaging.version import parse as V
import sys

if V("{}.{}.{}".format(*sys.version_info[:3])) >= V("3.6"):
    print("true")
else:
    print("false")
EOF
)

pt_plus(){
    python3 <<EOF
import sys
from packaging.version import parse as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

echo "[INFO] torch_version=${torch_version}"


if "$(pt_plus 1.8.0)" && "${python_36_plus}"; then

    rm -rf fairseq
    # NOTE(QH): using custom fairseq for espnet-pefts
    # git clone https://github.com/espnet/fairseq.git # original
    git clone https://github.com/WZLHQ/fairseq.git
    python3 -m pip install --editable ./fairseq
    python3 -m pip install filelock

else
    echo "[WARNING] fairseq requires pytorch>=1.8.0, python>=3.6"

fi
