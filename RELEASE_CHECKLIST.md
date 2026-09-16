# Release checklist (on acceptance)

Everything below is prepared; the steps are the ones that make things
public or need the final author information. Work top to bottom.

## 1. Authors and links

- [x] `docs/index.html` authors line and BibTeX, `README.md` citation, `LICENSE` holders, the arXiv `AUTHORS` block and the camera-ready wrapper all carry the confirmed author list and affiliations (2026-09-16).
- [x] Decisions recorded 2026-09-16: data on a GitHub release (v1.0), arXiv under the arXiv.org perpetual non-exclusive license, Tab. 2 kept as printed (the reproducible numbers are in REPRODUCING.md §7).
- [ ] Camera-ready wrapper `neurips_2026.tex`: switch to `\usepackage[main, final]{neurips_2026}` and make the abstract's last sentence point at the repository (already done in the arXiv copy).

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

- [ ] `python overleaf/neurips2026/make_arxiv.py` (after step 1); check `overleaf/arxiv/clean/main_arxiv.pdf`.
- [ ] Upload `overleaf/arxiv/m-plicits-arxiv.zip`; metadata and abstract in `overleaf/arxiv/SUBMISSION_NOTES.md`.
- [ ] After the arXiv id exists: fill the Paper and arXiv buttons in `docs/index.html` (two `TODO on acceptance` comments) and the README's paper link.

## 4. Repository and project page

- [ ] Commit the above; `git push`.
- [ ] Make the repository public: `gh repo edit dsilvavinicius/m-plicits --visibility public --accept-visibility-change-consequences`.
- [ ] Enable Pages from `main` / `docs`: `gh api -X POST repos/dsilvavinicius/m-plicits/pages -f "source[branch]=main" -f "source[path]=/docs"` (or Settings → Pages). Pages sites are public even for private repositories, which is why this is an acceptance-day step. The page is served at https://dsilvavinicius.github.io/m-plicits/ and already carries absolute social-card URLs and a `.nojekyll` marker.
- [ ] Check the page's video plays and the slider works on the live URL.

## 5. Sanity

- [ ] `python tools/download_data.py` in a fresh clone, then `python reproduce.py all`, and compare with `REPRODUCING.md`'s verification log.
