# v3.9.4 Doxygen formula recovery assets

These 29 PNGs reproduce the original archived v3.9.4 formula numbering and
formula text. They are not copied or renumbered from development documentation.
The original retained `cpp_api/api/_formulas.tex` is 1,737 bytes, SHA256
`31adb22408e895f689f8f84405c8bb6d5e533418c444e69db4b857a28a8c5c94`.

`manifest.json` records every original formula ID/text and text SHA256, PNG
SHA256/length/dimensions, and successful PNG decode validation. The input comes
from the complete original release archive (SHA256
`ee2002f7928037cb6c058b3775f37a5b77c14798ce67645ca75e5b721a186c77`).

Generation used Ubuntu 22.04, Doxygen 1.9.1, Ghostscript 9.55 and
texlive-latex-extra. The extracted original formula strings and eqnarray
context were retained; amsmath was enabled. The isolated Doxyfile used
`INPUT = mainpage.dox`, `OUTPUT_DIRECTORY = output`, `GENERATE_HTML = YES`,
`USE_MATHJAX = NO`, `EXTRA_PACKAGES = amsmath`, `HAVE_DOT = NO`, and was executed
as `doxygen Doxyfile` from its isolated formula replay directory. This was a
formula-only replay, not a new build of historical C++ sources. The original
formula strings and historical HTML are not changed by deployment.

The deploy owner applies these assets only to v3.9.4 with exactly the original
TeX SHA256 above, and only to actually referenced missing PNG paths. Existing
images are never overwritten. Unknown formula IDs, a different/missing source
TeX or asset checksum failures stop publication instead of substituting dev
formula numbers. Total PNG payload: 103,351 bytes.
