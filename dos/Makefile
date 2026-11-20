DOC   := thermo_dos
BUILD := build
TEX   := pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)

.PHONY: all view clean

all: $(BUILD)/$(DOC).pdf

$(BUILD):
	mkdir -p $(BUILD)

# pass 1 writes the .aux for bibtex; passes 2 and 3 settle citations and references
$(BUILD)/$(DOC).pdf: $(DOC).tex $(DOC).bib | $(BUILD)
	$(TEX) $(DOC).tex
	BIBINPUTS=.: bibtex $(BUILD)/$(DOC)
	$(TEX) $(DOC).tex
	$(TEX) $(DOC).tex

view: all
	xdg-open $(BUILD)/$(DOC).pdf >/dev/null 2>&1 &

clean:
	rm -rf $(BUILD)
