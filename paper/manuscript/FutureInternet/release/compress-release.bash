#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )"

cd "${SCRIPT_DIR}"

rm -rf BASN/
mkdir -p BASN/

cp -r ../Definitions/ ../images/ ../tex/ ../main.bib ../main.tex ../Makefile BASN/

zip -r BASN.zip BASN/

rm -rf BASN/

