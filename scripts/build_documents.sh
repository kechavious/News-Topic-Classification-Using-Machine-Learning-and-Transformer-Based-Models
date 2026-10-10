#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pandoc writeup/paper.md --citeproc --bibliography=writeup/references.bib --pdf-engine=pdflatex -o writeup/paper.pdf
pandoc presentation/slides.md -t beamer --pdf-engine=pdflatex -o presentation/slides.pdf
