ms1: clean milestone1.tex
	pdflatex milestone1.tex

propose: clean propose.tex
	pdflatex propose.tex

clean:
	-rm *.log *.aux *.out
	-rm *.pdf
