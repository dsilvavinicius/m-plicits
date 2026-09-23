# Release checklist (on acceptance)

Everything below is prepared; the steps are the ones that make things
public or need the final author information. Work top to bottom.

## 1. Authors and links

- [x] `docs/index.html` authors line and BibTeX, `README.md` citation, `LICENSE` holders, the arXiv `AUTHORS` block and the camera-ready wrapper all carry the confirmed author list and affiliations (2026-09-16).
- [x] Decisions recorded 2026-09-16: data on a GitHub release (v1.0), arXiv under the arXiv.org perpetual non-exclusive license, Tab. 2 kept as printed (the reproducible numbers are in REPRODUCING.md §7).
- [ ] Camera-ready wrapper `neurips_2026.tex`: switch to `\usepackage[main, final]{neurips_2026}` and make the abstract's last sentence point at the repository (already done in the arXiv copy). Then replace `docs/assets/m-plicits.pdf` with the camera-ready PDF (`python overleaf/neurips2026/compress_pdf.py <pdf> docs/assets/m-plicits.pdf --quality 80 --max-dpi 200`).

## 2. Data archives

Build (from the private tree; prints the SHA-256 of each archive):

```bash
python tools/make_input_zip.py --src ../i3d_work --out ../m-plicits-data.zip \
    --renderer-data renderer/cuda/data --attributes-data ../neural_implicit_attributes/data \
    --attributes-shapenets ../neural_implicit_attributes/shapeNets \
    --noise-meshes ../i3d_work/rebuttal_2026/paper_noise_recs/noise_data_and_reconstructions/rec/noise_meshes
```

- [ ] Paste both SHA-256 values into `tools/download_data.py` (`ARCHIVES`).
- [ ] Publish: `gh release create v1.0 ../m-plicits-data.zip ../m-plicits-paper-meshes.zip --title "Data and released models" --notes "Inputs, noisy variants, released checkpoints, renderer/texture assets; optional paper reconstruction meshes."` (each asset is under GitHub's 2 GB limit). A Google Drive mirror is optional: put the file ids in the release notes and pass them to `download_data.py --gdrive`.
- [ ] Optional, citable: archive the release on Zenodo for a DOI and add it to the README.

## 3. arXiv

- [x] `python overleaf/neurips2026/make_arxiv.py`; `overleaf/arxiv/clean/main_arxiv.pdf` checked (page 1 author block fixed 2026-09-21).
- [x] v1 uploaded 2026-09-23 as arXiv submission 8105364: `[preprint]` build, abstract says "will be released at" the repository, Comments empty, cs.CV primary with cs.GR and cs.LG cross-lists, arXiv non-exclusive license. The id arrives by e-mail at announcement (14:00 ET cutoff, announced 20:00 ET).
- [x] Paper button and Resources card in `docs/index.html`, and the README link, point at `docs/assets/m-plicits.pdf` (the preprint, images recompressed to 200 dpi, 9.5 MB) since 2026-09-23.
- [ ] After the arXiv id exists: restore the arXiv button in `docs/index.html` (commented placeholder next to the Paper button) and add the arXiv link to the README.
- [ ] v2 on acceptance: switch `repo_sentence` in `make_arxiv.py` back to "are available at", name the venue in Comments (see `overleaf/arxiv/SUBMISSION_NOTES.md`), replace via the arXiv "Replace" action.

## 4. Repository and project page

- [ ] Commit the above; `git push`.
- [ ] Make the repository public: `gh repo edit dsilvavinicius/m-plicits --visibility public --accept-visibility-change-consequences`.
- [ ] Enable Pages from `main` / `docs`: `gh api -X POST repos/dsilvavinicius/m-plicits/pages -f "source[branch]=main" -f "source[path]=/docs"` (or Settings → Pages). Pages sites are public even for private repositories, which is why this is an acceptance-day step. The page is served at https://dsilvavinicius.github.io/m-plicits/ and already carries absolute social-card URLs and a `.nojekyll` marker.
- [ ] Check the page's video plays and the slider works on the live URL.

## 5. Sanity

- [ ] `python tools/download_data.py` in a fresh clone, then `python reproduce.py all`, and compare with `REPRODUCING.md`'s verification log.
