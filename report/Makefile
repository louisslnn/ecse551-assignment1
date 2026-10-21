all: report.pdf

report.pdf: report.tex
	latexmk -pdf report.tex

.PHONY: clean
clean:
	latexmk -C
