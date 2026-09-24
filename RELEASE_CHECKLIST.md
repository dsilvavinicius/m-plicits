# Release checklist (on acceptance)

Everything below is prepared; the steps are the ones that make things
public or need the final author information. Work top to bottom.

State on 2026-09-24 (accepted as a poster): repository public, Pages live,
data on GitHub release v1.0, camera-ready wrapper and PDF built (the page's
Paper link is the camera-ready copy), arXiv v1 submitted (id pending
announcement), arXiv v2 package pre-built with the "are available" wording.
Left: the arXiv links and v2 once v1 is announced (step 3), the camera-ready
upload to OpenReview by the deadline in the acceptance e-mail, and the
public-download run (step 5).

## 1. Authors and links

- [x] `docs/index.html` authors line and BibTeX, `README.md` citation, `LICENSE` holders, the arXiv `AUTHORS` block and the camera-ready wrapper all carry the confirmed author list and affiliations (2026-09-16).
- [x] Decisions recorded 2026-09-16: data on a GitHub release (v1.0), arXiv under the arXiv.org perpetual non-exclusive license, Tab. 2 kept as printed (the reproducible numbers are in REPRODUCING.md §7).
- [x] Camera-ready wrapper `neurips_2026.tex` uses `\usepackage[main, final]{neurips_2026}` and the abstract's last sentence points at the repository (2026-09-23); `docs/assets/m-plicits.pdf` is the compressed camera-ready build (38 pages with the checklist, NeurIPS 2026 footer). Compile recipe and notes: `overleaf/neurips2026/CAMERA_READY_CHANGES.md`.
- [ ] On acceptance, if the camera-ready changes (reviewer requests, acknowledgments): recompile (`BIBINPUTS="..;" latexmk -pdf -bibtex -g neurips_2026.tex` in `overleaf/neurips2026`), then `python overleaf/neurips2026/compress_pdf.py overleaf/neurips2026/neurips_2026.pdf docs/assets/m-plicits.pdf --quality 80 --max-dpi 200`, commit, push. Upload the camera-ready to OpenReview as NeurIPS instructs.

## 2. Data archives

Build (from the private tree; prints the SHA-256 of each archive):

```bash
python tools/make_input_zip.py --src ../i3d_work --out ../m-plicits-data.zip \
    --renderer-data renderer/cuda/data --attributes-data ../neural_implicit_attributes/data \
    --attributes-shapenets ../neural_implicit_attributes/shapeNets \
    --noise-meshes ../i3d_work/rebuttal_2026/paper_noise_recs/noise_data_and_reconstructions/rec/noise_meshes
```

- [x] Both SHA-256 values are in `tools/download_data.py` (`ARCHIVES`), 2026-09-16.
- [x] Published 2026-09-16 as release v1.0 (`gh release create v1.0 ../m-plicits-data.zip ../m-plicits-paper-meshes.zip --title "Data and released models" ...`); the release becomes visible with the repository. A Google Drive mirror is optional: put the file ids in the release notes and pass them to `download_data.py --gdrive`.
- [ ] Optional, citable: archive the release on Zenodo for a DOI and add it to the README.

## 3. arXiv

- [x] `python overleaf/neurips2026/make_arxiv.py`; `overleaf/arxiv/clean/main_arxiv.pdf` checked (page 1 author block fixed 2026-09-21).
- [x] v1 uploaded 2026-09-23 as arXiv submission 8105364: `[preprint]` build, abstract says "will be released at" the repository, Comments empty, cs.CV primary with cs.GR and cs.LG cross-lists, arXiv non-exclusive license. The id arrives by e-mail at announcement (14:00 ET cutoff, announced 20:00 ET).
- [x] Paper button and Resources card in `docs/index.html`, and the README link, point at `docs/assets/m-plicits.pdf` (the preprint, images recompressed to 200 dpi, 9.5 MB) since 2026-09-23.
- [ ] After the arXiv id exists: restore the arXiv button in `docs/index.html` (commented placeholder next to the Paper button) and add the arXiv link to the README.
- [ ] v2 on acceptance: switch `repo_sentence` in `make_arxiv.py` back to "are available at", name the venue in Comments (see `overleaf/arxiv/SUBMISSION_NOTES.md`), replace via the arXiv "Replace" action.

## 4. Repository and project page

- [x] Accepted (poster) on 2026-09-24. Repository made public the same day (`gh api -X PATCH repos/dsilvavinicius/m-plicits -F private=false`; the installed `gh` lacks the `--accept-visibility-change-consequences` flag), homepage set to the project page, description and topics added.
- [x] Pages enabled from `main` / `docs` (`gh api -X POST repos/dsilvavinicius/m-plicits/pages -f "source[branch]=main" -f "source[path]=/docs"`); first build finished within a minute, HTTPS enforced. Live at https://dsilvavinicius.github.io/m-plicits/.
- [x] Live checks 2026-09-24: fonts loaded, 39 images without a broken one, video ready (25 s), slider and tabs present, Paper link serves the PDF, social-card image reachable, no console errors.

## 5. Sanity

- [x] 2026-09-24, right after going public: anonymous clone, `python tools/download_data.py --paper-meshes` through the public release URLs (both archives, SHA-256 verified, unpacked: 36 checkpoint folders, 108 paper meshes, renderer and texture assets in place).
- [x] `python reproduce.py all` was verified from a fresh clone on 2026-09-16 (REPRODUCING.md §7); no code changed since.
