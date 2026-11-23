DOC   := thermo_dos
SUPPL := thermo_dos_suppl
BUILD := build
TEX   := pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)

.PHONY: all main suppl view clean

all: main suppl
main: $(BUILD)/$(DOC).pdf
suppl: $(BUILD)/$(SUPPL).pdf

$(BUILD):
	mkdir -p $(BUILD)

# pass 1 writes the .aux for bibtex; passes 2 and 3 settle citations and references
$(BUILD)/$(DOC).pdf: $(DOC).tex $(DOC).bib | $(BUILD)
	$(TEX) $(DOC).tex
	BIBINPUTS=.: bibtex $(BUILD)/$(DOC)
	$(TEX) $(DOC).tex
	$(TEX) $(DOC).tex

# the supplement refers to labels of the main text (package xr-hyper, prefix M-),
# read from the main text's .aux file
$(BUILD)/$(SUPPL).pdf: $(SUPPL).tex $(DOC).bib $(BUILD)/$(DOC).pdf | $(BUILD)
	$(TEX) $(SUPPL).tex
	BIBINPUTS=.: bibtex $(BUILD)/$(SUPPL)
	$(TEX) $(SUPPL).tex
	$(TEX) $(SUPPL).tex

view: all
	xdg-open $(BUILD)/$(DOC).pdf >/dev/null 2>&1 &
	xdg-open $(BUILD)/$(SUPPL).pdf >/dev/null 2>&1 &

clean:
	rm -rf $(BUILD)
