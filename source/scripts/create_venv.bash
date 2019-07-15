#!/bin/bash

SCRIPT_PATH="$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )"

VENV_PATH="${SCRIPT_PATH}/../venv/"

python3 -m ensurepip --version > /dev/null 2>&1

if [ "$?" -ne 0 ]; then
  (>&2 echo "Please install python3-venv or python3-ensurepip")
  exit 1
fi

python3 -m venv "${VENV_PATH}"

source "${VENV_PATH}/bin/activate"

pip install "pip==19.1.1" "wheel==0.33.4" "setuptools==41.0.1"
pip install -r "${SCRIPT_PATH}/requirements.txt"

