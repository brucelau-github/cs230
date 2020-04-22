pdf: clean propose.tex
	pdflatex propose.tex
	-rm *.log *.aux *.out
clean:
	-rm propose.pdf
