(TeX-add-style-hook
 "preamble"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("scrartcl" "12pt" "oneside" "paper=A4" "DIV=15" "BCOR=0mm" "abstract=true" "headings=small")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("inputenc" "utf8") ("fontenc" "T1") ("scrlayer-scrpage" "headsepline" "footsepline" "automark") ("microtype" "auto") ("biblatex" "backend=biber" "style=iso-numeric" "citestyle=numeric-comp" "maxbibnames=2" "firstinits=true") ("hyperref" "pdftitle={PaperSummary}" "pdfsubject={}" "pdfauthor={Arvind Nayak}" "pdfkeywords={Academic Report,Summary}" "hidelinks")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "scrartcl"
    "scrartcl12"
    "babel"
    "inputenc"
    "lmodern"
    "fontenc"
    "scrlayer-scrpage"
    "amsmath"
    "amssymb"
    "microtype"
    "graphicx"
    "subfig"
    "multirow"
    "multicol"
    "booktabs"
    "threeparttable"
    "longtable"
    "rotating"
    "ltablex"
    "pdfpages"
    "listings"
    "url"
    "footnote"
    "todonotes"
    "blindtext"
    "biblatex"
    "hyperref")
   (LaTeX-add-environments
    "theorem"
    "corollary"
    "conjecture"
    "tabular")
   (LaTeX-add-bibliographies
    "references"))
 :latex)

