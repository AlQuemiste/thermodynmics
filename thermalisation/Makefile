# build thermalize.pdf into ./build
DOC     := thermalize
BUILD   := build
PYTHON  := $(HOME)/.venv/myPy/bin/python
VIEWER  ?= xdg-open
FIGS    := figs/chain.pdf figs/fput.pdf figs/bath.pdf figs/debye.pdf figs/lenk_mz.pdf figs/lenk_quench.pdf
LATEX   := pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)

.PHONY: all pdf figs check25 view clean distclean

all: pdf

pdf: $(BUILD)/$(DOC).pdf

# three passes settle the table of contents and cross-references
$(BUILD)/$(DOC).pdf: $(DOC).tex $(FIGS) | $(BUILD)
	$(LATEX) $(DOC).tex > /dev/null
	$(LATEX) $(DOC).tex > /dev/null
	$(LATEX) $(DOC).tex | grep -E "LaTeX Warning|Error|Output written" || true

$(BUILD):
	mkdir -p $(BUILD)

# figures are regenerated only on request (fput ~3 min, bath ~1 min,
# lenk ~4 min: exact diagonalization up to N = 20, D = 10626)
figs:
	$(PYTHON) code/fig_fput.py
	$(PYTHON) code/fig_bath.py
	$(PYTHON) code/fig_debye.py
	$(PYTHON) code/fig_lenk.py

# quench observables at N = 25 quoted in the text (D = 23751: ~15 min, ~15 GB RAM)
check25:
	$(PYTHON) code/lenk_check.py 25

view: pdf
	$(VIEWER) $(BUILD)/$(DOC).pdf >/dev/null 2>&1 &

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -f figs/*.npz
