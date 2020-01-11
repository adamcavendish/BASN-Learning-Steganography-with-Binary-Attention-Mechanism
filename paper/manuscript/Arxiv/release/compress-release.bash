#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname ${BASH_SOURCE[0]} )" && pwd )"

cd "${SCRIPT_DIR}"

rm -rf BASN/
mkdir -p BASN/

cp -r  ../images/ ../tex/ ../main.tex ../build/pdflatex/main.bbl BASN/

for file in $(find . -iname *-eps-converted-to.pdf); do
  echo "Move ${file} -> ${file%-eps-converted-to.pdf}.pdf"
  mv "${file}" "${file%-eps-converted-to.pdf}.pdf"
done
find . -iname *.py  -exec bash -c 'echo removing "{}"; rm {}' \;
find . -iname *.eps -exec bash -c 'echo removing "{}"; rm {}' \;
find . -iname *.png -exec pngquant -f --ext .png --quality 80-100 -s 1 {} \;

cd BASN/
tar czf ../BASN.tar.gz *
cd ../

rm -rf BASN/

