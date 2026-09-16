# Release checklist (on acceptance)

Everything below is prepared; the steps are the ones that make things
public or need the final author information. Work top to bottom.

## 1. Authors and links

- [ ] `docs/index.html`: replace "Authors revealed on acceptance" (the `.authors` line) and the BibTeX block; keep the venue line.
- [ ] `README.md`: Citation block.
- [ ] `LICENSE`: copyright holder line.
- [ ] `overleaf/neurips2026/make_arxiv.py`: `AUTHORS` block, then rerun `python make_arxiv.py`.
- [ ] Camera-ready wrapper `neurips_2026.tex`: `\usepackage[main, final]{neurips_2026}`, the same author block, and the abstract's last sentence pointing at the repository (already done in the arXiv copy).

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
