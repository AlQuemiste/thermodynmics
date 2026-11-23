DOC   := thermo_dos
SUPPL := thermo_dos_suppl
BUILD := build
TEX   := pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)

.PHONY: all view clean

all: $(BUILD)/$(DOC).pdf

$(BUILD):
	mkdir -p $(BUILD)

# The main text and the supplement refer to each other's labels (package xr-hyper),
# so they are typeset together: pass 1 writes both .aux files for bibtex and for
# the cross-references; passes 2 and 3 settle citations and references.
$(BUILD)/$(DOC).pdf $(BUILD)/$(SUPPL).pdf: $(DOC).tex $(SUPPL).tex $(DOC).bib | $(BUILD)
	$(TEX) $(DOC).tex
	$(TEX) $(SUPPL).tex
	BIBINPUTS=.: bibtex $(BUILD)/$(DOC)
	BIBINPUTS=.: bibtex $(BUILD)/$(SUPPL)
	$(TEX) $(DOC).tex
	$(TEX) $(SUPPL).tex
	$(TEX) $(DOC).tex
	$(TEX) $(SUPPL).tex

view: all
	xdg-open $(BUILD)/$(DOC).pdf >/dev/null 2>&1 &
	xdg-open $(BUILD)/$(SUPPL).pdf >/dev/null 2>&1 &

clean:
	rm -rf $(BUILD)
