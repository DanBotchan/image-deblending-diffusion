# Project page: Image Deblending with Diffusion Models (WACV 2027)

A static site: `index.html` plus `static/img/`. No build step.

- Preview locally: `python -m http.server` in this folder, then open http://localhost:8000.
- Regenerate images from the paper figures (needs Pillow and PyMuPDF):
  `python tools/prep_assets.py ../PaperOverleaf/figures .`
- Links: buttons whose `href` is `#` render as dashed "soon" buttons. Replace them with the real
  Paper, Supplementary, Code and arXiv URLs in the hero `<nav class="links">`.
- Deploy on GitHub Pages: push this folder as a repo (or to a `gh-pages` branch) and enable
  Pages from the root. `.nojekyll` makes GitHub serve the files as-is.
