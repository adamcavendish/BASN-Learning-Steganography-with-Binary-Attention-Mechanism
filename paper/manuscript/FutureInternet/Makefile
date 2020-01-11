CWD := $(dir $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

all: pdflatex

pdflatex:
	mkdir -p build/pdflatex/
	latexmk -pdf -shell-escape -output-directory=build/pdflatex -synctex=1 -file-line-error main.tex
	ln -sf build/pdflatex/main.pdf BASN-Learning-Steganography-with-Binary-Attention-Mechanism.pdf

xelatex:
	mkdir -p build/xelatex/
	latexmk -xelatex -shell-escape -output-directory=build/xelatex/ -synctex=1 -file-line-error main.tex
	ln -sf build/xelatex/main.pdf BASN-Learning-Steganography-with-Binary-Attention-Mechanism.pdf

lualatex:
	mkdir -p build/lualatex/
	latexmk -lualatex -shell-escape -output-directory=build/lualatex/ -synctex=1 -file-line-error main.tex
	ln -sf build/lualatex/main.pdf BASN-Learning-Steganography-with-Binary-Attention-Mechanism.pdf

clean:
	rm -f build/*/*.log
	rm -f build/*/*.aux
	rm -f build/*/*.dvi
	rm -f build/*/*.lof
	rm -f build/*/*.lot
	rm -f build/*/*.toc
	rm -f build/*/*.bbl
	rm -f build/*/*.blg
	rm -f build/*/*.bcf
	rm -f build/*/*.xml

clean-pdf:
	rm -f build/*/*.pdf
	
clean-all: clean clean-pdf
	rm -rf build/

