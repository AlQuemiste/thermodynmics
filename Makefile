DOC     ?= thermodynamics
BIBFILE := $(DOC).bib
TEX     := pdflatex -interaction=nonstopmode -halt-on-error
BIB     := bibtex
VIEWER  ?= xdg-open
BUILD   := build


.PHONY: all doc view clean

all: doc

doc: $(BUILD)/$(DOC).pdf

$(BUILD):
	mkdir -p $(BUILD)

# main doc
$(BUILD)/$(DOC).pdf: $(DOC).tex | $(BUILD)
	$(TEX) -output-directory=$(BUILD) $(DOC).tex
	$(TEX) -output-directory=$(BUILD) $(DOC).tex

view: $(BUILD)/$(DOC).pdf
	$(VIEWER) $(BUILD)/$(DOC).pdf >/dev/null 2>&1 &

clean:
	rm -rf $(BUILD)
