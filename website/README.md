# Project page: Image Deblending with Diffusion Models (WACV 2027)

A static site: `index.html` plus `static/img/`. No build step.

- Preview locally: `python -m http.server 8000 --directory website` from the repository root, then open http://localhost:8000.
- Regenerate images from the paper figures (needs Pillow and PyMuPDF):
  `python website/tools/prep_assets.py ../PaperOverleaf/figures website`
- Links: buttons whose `href` is `#` render as dashed "soon" buttons. Replace them with the real
  Paper, Supplementary and arXiv URLs in the hero `<nav class="links">`.
- Deploy on GitHub Pages: push this folder as a repo (or to a `gh-pages` branch) and enable
  Pages from the root. `.nojekyll` makes GitHub serve the files as-is.
