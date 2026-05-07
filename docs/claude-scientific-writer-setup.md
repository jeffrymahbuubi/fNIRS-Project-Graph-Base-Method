# Claude Scientific Writer — Bootstrap Guide

A reusable setup guide for adding the [`claude-scientific-writer`](https://github.com/K-Dense-AI/claude-scientific-writer) tool (K-Dense AI) to a new project. This document captures what was learned bootstrapping the tool into the fNIRS GAD paper project so it can be reapplied elsewhere.

---

## What It Is

`claude-scientific-writer` is a deep-research + scientific writing toolkit that runs inside Claude Code. It produces publication-ready papers, posters, grants, clinical reports, literature reviews, and AI-generated figures, with citations grounded in real literature.

**Three usage modes:**

| Mode | When to use | What you get |
|------|-------------|--------------|
| Claude Code skills (copy or plugin) | Interactive writing inside the IDE | All 20+ skills callable as `@skill-name` or `/skill-name` |
| Python CLI (`scientific-writer`) | One-shot PDF generation from prompt + data | LaTeX → PDF pipeline with peer-review pass |
| Python API (`generate_paper(...)`) | Programmatic / scripted document generation | Async streaming progress + result object |

You do **not** have to pick one — they coexist. The skills do the writing, the CLI/API automates the build.

---

## Decision Tree: Pick Your Install Path

```
Do you only need writing skills inside Claude Code?
├── Yes  → Copy-Skills (Path A)         ← simplest, no marketplace
│         or Plugin Install (Path B)    ← upstream-recommended
└── No   → Also need PDF compile / batch generation?
          └── Yes → Add Python CLI/API (Path C)

Do you also want AI-generated figures (schematics, photos)?
└── Yes → Add Figure Scripts (Path D)
```

The fNIRS project uses **Path A + Path D** (skills copied, figures planned).

---

## Prerequisites

| Requirement | Required for | Notes |
|-------------|--------------|-------|
| Claude Code | Paths A, B | Active session |
| `ANTHROPIC_API_KEY` | A, B, C | Already set if Claude Code works |
| `OPENROUTER_API_KEY` | A/B (research-lookup fallback), D (figure gen) | Free tier sufficient for testing |
| `PARALLEL_API_KEY` | A/B (research-lookup primary), `parallel-web` skill | Optional; project falls back gracefully without it |
| Python 3.10–3.12 | C, D | Match upstream constraint |
| LaTeX (`pdflatex`, `bibtex`) | C (when compiling) | TeX Live / MacTeX |
| `requests` Python package | D | `pip install requests` |

`PARALLEL_API_KEY` enables the upstream-recommended `parallel-web` skill (web search, URL extract, deep research). When absent, `research-lookup` automatically routes through Perplexity via OpenRouter — works fine for academic searches.

---

## Path A — Copy Skills (Recommended for new projects)

**Why this path:** The official `/scientific-writer:init` command in the upstream repo has hardcoded developer paths (`/Users/vinayak/...`) that fail outside the maintainer's machine. Copying skills sidesteps that issue.

### A.1 Clone the upstream repo as a reference

```bash
mkdir -p references
cd references
git clone https://github.com/K-Dense-AI/claude-scientific-writer.git
cd ..
```

Treat `references/claude-scientific-writer/` as a read-only source. Do not edit it.

### A.2 Copy the skills you need

There are 24 skills upstream. Copy only what your project will use. A typical scientific-paper project needs six:

```bash
mkdir -p .claude/skills
for skill in scientific-writing research-lookup citation-management \
             peer-review literature-review venue-templates; do
  cp -r references/claude-scientific-writer/skills/$skill .claude/skills/
done
```

Add more as needs arise:

| Need | Copy this skill |
|------|-----------------|
| Clinical case reports | `clinical-reports`, `clinical-decision-support`, `treatment-plans` |
| Conference posters | `latex-posters`, `pptx-posters` |
| Slides / presentations | `scientific-slides` |
| Grants (NSF/NIH/DOE) | `research-grants` |
| Hypothesis papers | `hypothesis-generation` |
| Market research | `market-research-reports` |
| Document conversion | `markitdown`, `document-skills` |
| Web→paper conversion | `paper-2-web` |
| Critical thinking pass | `scientific-critical-thinking`, `scholar-evaluation` |
| Web search via Parallel | `parallel-web` |
| Infographics (PNG) | `infographics` |
| Schematics (AI) | `scientific-schematics` (also see Path D) |
| Photo-style figures | `generate-image` (also see Path D) |

### A.3 Configure API keys

Create a `.env` (or export in shell):

```bash
ANTHROPIC_API_KEY=...      # already set if Claude Code works
OPENROUTER_API_KEY=...     # for research-lookup fallback + figure gen
PARALLEL_API_KEY=...       # optional; enables parallel-web skill
```

Add `.env` to `.gitignore`.

### A.4 Verify install

Restart Claude Code, then in any chat:

```
What skills are available?
```

You should see your copied skills listed. Try a smoke test:

```
@research-lookup Find 3 recent papers on <your topic>. Save to research/sources/smoke_test.md
```

---

## Path B — Official Plugin Install

Use this if upstream has fixed the hardcoded paths or you accept editing the generated `CLAUDE.md` afterward.

```
/plugin marketplace add https://github.com/K-Dense-AI/claude-scientific-writer
/plugin install claude-scientific-writer
```

Restart Claude Code, then:

```
/scientific-writer:init
```

This creates a project-level `CLAUDE.md` populated from `templates/CLAUDE.scientific-writer.md`. Two caveats observed:

1. The init command may try absolute paths like `/Users/vinayak/...`. If it fails, fall back to Path A or pre-copy `templates/CLAUDE.scientific-writer.md` manually.
2. The generated `CLAUDE.md` enforces strict policies (real-citations only, save-to-`sources/`, mandatory graphical abstract, parallel-web routing). Read it before merging into an existing `CLAUDE.md`.

If you already have a `CLAUDE.md`, the init command will offer: back up & replace, merge (append), or cancel.

---

## Path C — Python CLI and API

For automated PDF generation outside the IDE.

### C.1 Install

Pick one:

```bash
# From PyPI
pip install scientific-writer

# Or from source (uv)
git clone https://github.com/K-Dense-AI/claude-scientific-writer.git
cd claude-scientific-writer
uv sync
```

### C.2 CLI usage

```bash
scientific-writer
# or, from source:
uv run scientific-writer
```

Interactive prompt drives generation. Outputs land in `writing_outputs/<timestamp>_<topic>/` with `drafts/`, `references/`, `figures/`, `sources/`, `final/` subfolders and a `progress.md` log.

### C.3 Python API

```python
import asyncio
from scientific_writer import generate_paper

async def main():
    async for update in generate_paper(
        query=(
            "Create a NeurIPS paper on transformers. "
            "Present results.csv (5 architectures, n=200). "
            "Include training_curves.png and ablation.png."
        ),
        data_files=["results.csv", "training_curves.png", "ablation.png"],
        output_dir="./papers",
        track_token_usage=True,
    ):
        if update["type"] == "progress":
            print(f"[{update['stage']}] {update['message']}")
        else:
            print(f"PDF: {update['files']['pdf_final']}")

asyncio.run(main())
```

The `generate_paper` async iterator streams progress and yields a final `result` dict with `pdf_final`, `tex_final`, figure list, citation list, and (optionally) token usage.

---

## Path D — AI Figure Generation

The figure scripts are not auto-installed by the skills. They must be copied into the project so the SKILL.md references resolve.

### D.1 Copy scripts

```bash
mkdir -p scripts/schematics scripts/generate-image
cp -r references/claude-scientific-writer/skills/scientific-schematics/scripts/* scripts/schematics/
cp -r references/claude-scientific-writer/skills/generate-image/scripts/* scripts/generate-image/
pip install requests
```

### D.2 Copy companion skills (if not already done in A.2)

```bash
cp -r references/claude-scientific-writer/skills/scientific-schematics .claude/skills/
cp -r references/claude-scientific-writer/skills/generate-image .claude/skills/
```

### D.3 Run

```bash
# Schematic (CONSORT, network architecture, pathway, etc.)
python scripts/schematics/generate_schematic.py \
  "CONSORT diagram: 450 screened, 312 randomized, 156 per arm" \
  -o figures/consort.png

# Photorealistic / illustrative
python scripts/generate-image/generate_image.py \
  "Microscopy of HEK293 cells expressing fluorescent reporter, professional scientific style" \
  -o figures/microscopy.png
```

Default model is `google/gemini-3-pro-image-preview` (via OpenRouter). Alternates: `black-forest-labs/flux.2-pro`, `black-forest-labs/flux.2-flex`. Pass `--model` to switch.

---

## Parallel API vs OpenRouter Fallback

The upstream `CLAUDE.md` mandates Parallel for all web operations. When `PARALLEL_API_KEY` is absent, the project still works — it just routes differently.

| Operation | With `PARALLEL_API_KEY` | Without (OpenRouter only) |
|-----------|-------------------------|---------------------------|
| Academic paper search | `research-lookup` → Parallel `sonar-pro-search` | `research-lookup` → Perplexity via OpenRouter |
| Generic web search | `parallel-web` skill (`parallel_web.py search`) | Use Claude Code's built-in WebSearch (last resort) |
| URL content extraction | `parallel-web` skill (`parallel_web.py extract`) | Manual via WebFetch |
| Deep research | `parallel-web` skill (`parallel_web.py research`) | Multiple `research-lookup` calls + manual synthesis |
| Citation metadata fix-up | `parallel_web.py extract` on DOI URLs | `research-lookup` on DOI/title queries |

**Practical rule:** if you have `PARALLEL_API_KEY`, copy `parallel-web` skill in A.2 and follow upstream verbatim. If you don't, omit `parallel-web` and accept that `research-lookup` (Perplexity) handles all literature work.

---

## Project Layout Conventions

Upstream expects this structure inside any writing output. Mirror it from day one:

```
<project-root>/
├── .claude/skills/                    # copied skills (Path A/B)
├── scripts/
│   ├── schematics/generate_schematic.py
│   └── generate-image/generate_image.py
├── references/
│   └── claude-scientific-writer/      # read-only upstream clone
└── research/                          # or writing_outputs/, your choice
    └── paper-materials/
        ├── PAPER_SPEC_PLAN.md         # ground truth — read first each session
        ├── progress.md                # auto-generated by skills
        ├── SUMMARY.md                 # final summary
        ├── PEER_REVIEW.md             # peer-review skill output
        ├── drafts/                    # v1_<section>.tex / .md
        ├── references/                # references.bib
        ├── figures/                   # all images
        ├── data/                      # CSVs, XLSX, JSON
        ├── sources/                   # ALL research-lookup / parallel results
        └── final/                     # compiled manuscript.{tex,pdf}
```

**Why a spec plan file:** Upstream `CLAUDE.md` instructs the agent to "research before writing" and "real citations only." Without a single ground-truth document, separate sessions drift on numbers, names, and decisions. Maintain `PAPER_SPEC_PLAN.md` as the authoritative source — read it at the start of every session.

---

## Skill Invocation Pattern

Skills do **not** auto-save. Always specify the output path:

```
/research-lookup <query>. Save to research/paper-materials/sources/papers_<topic>_YYYYMMDD.md

/scientific-writing Write <Section X.Y> per PAPER_SPEC_PLAN.md.
                    ~<N> words. <STYLE> citation style.
                    Save to research/paper-materials/drafts/v1_<section>.tex

/citation-management Verify and generate BibTeX for: <DOI/title list>.
                     Save to research/paper-materials/references/references.bib

/venue-templates Get <VENUE> formatting + LaTeX skeleton.
                 Save to research/paper-materials/drafts/main.tex

/peer-review Evaluate research/paper-materials/drafts/v3_full_manuscript.tex.
             Save to research/paper-materials/PEER_REVIEW.md
```

For figures (Path D):

```bash
python scripts/schematics/generate_schematic.py "<description>" -o figures/<name>.png
```

---

## Multi-Pass Writing Workflow (from upstream)

1. **Pass 1 — Skeleton.** Create the full LaTeX/Markdown structure with section placeholders. Empty `references.bib`. Spec plan locked in.
2. **Pass 2 — Section research+write loop.** For each section:
   - `research-lookup` 5–10 real papers, save to `sources/`.
   - Write the section integrating *only* those citations.
   - Append BibTeX entries as you cite.
   - Log: `[HH:MM:SS] COMPLETED: <Section> — <N> words, <M> citations`.
3. **Pass 3 — Polish.** Write Abstract last. Compile (`pdflatex → bibtex → pdflatex × 2`). PDF formatting review (convert pages to images via `scripts/pdf_to_images.py`, inspect, fix).
4. **Pass 4 — Peer review.** Run `peer-review` skill, save `PEER_REVIEW.md`, address findings.

---

## Quality Rules to Carry Over

These come from upstream `CLAUDE.md`. They matter regardless of install path:

- **Zero placeholder citations.** Every BibTeX entry must point to a verifiable paper. Use `research-lookup` first, fill in `volume`, `pages`, `doi` for every `@article`.
- **Save every research result to `sources/`.** Use the `-o` flag on every script call. Re-read from `sources/` before duplicating queries.
- **Graphical abstract is mandatory** for papers, reviews, reports. Generate as Figure 1 via `scientific-schematics`.
- **Generate figures liberally.** Minimums: 5 for research papers, 4 for reviews, 6 for posters, 4 for grants, 3 for clinical reports.
- **Increment version numbers** when editing existing drafts (`v1_*.tex` → `v2_*.tex`). Never overwrite.
- **Never read PDFs directly.** Convert to images for review.
- **Do not use SPARC orchestrator commands** for paper writing — invoke skills directly. SPARC adds overhead for what is fundamentally an interactive writing task.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `/scientific-writer:init` fails with `/Users/vinayak/...` | Hardcoded path in upstream init command | Use Path A (copy skills manually) |
| `research-lookup` errors on `PARALLEL_API_KEY missing` | Skill prefers Parallel | Confirm `OPENROUTER_API_KEY` set; skill auto-falls back |
| Skills not visible in Claude Code | Missing/invalid YAML frontmatter, or skills not copied | Verify `.claude/skills/<name>/SKILL.md` exists with `name:`/`description:` frontmatter, restart Claude Code |
| `generate_schematic.py: command not found` | Scripts not copied to project | Run Path D step D.1 |
| `pdflatex` errors on missing packages | Incomplete TeX install | `tlmgr install <package>` or use Overleaf for final compile |
| Citations missing `volume`/`pages`/`doi` | Upstream rule violated | Run `parallel_web.py extract` on the DOI URL, or `research-lookup` to find metadata, then update `references.bib` |

---

## Case Study: fNIRS GAD Paper (this project)

A worked example of Path A + Path D applied to a real paper.

**Goal.** Submit "Grid-Based Spatiotemporal Encoding of fNIRS for GAD Classification" to **IEEE TNSRE**.

**What was set up:**

- 6 skills copied: `scientific-writing`, `research-lookup`, `citation-management`, `peer-review`, `literature-review`, `venue-templates`.
- Output folder: `research/paper-materials/` (preferred over `docs/paper/`).
- Spec plan: `research/paper-materials/PAPER_SPEC_PLAN.md` — single source of truth for grid layout (5×7, RBF interpolation), Homer3 pipeline parameters (HPF 0.01 Hz, LPF 0.5 Hz, PPF=6), task epoch lengths, demographics (48 subj: 32 HC / 16 GAD), ViT config (depth=6, dim=64, heads=8), and verified results (HbT 88.4±12.4% acc on GNG, Config C T=256/H=128/W=128).
- Subdirs created: `drafts/`, `references/`, `figures/`, `sources/`, `final/`, `old-result/`.

**API keys actually used:** `ANTHROPIC_API_KEY` (always), `OPENROUTER_API_KEY` (for `research-lookup` Perplexity fallback). `PARALLEL_API_KEY` not set — `parallel-web` skill skipped.

**Skills not copied:** `clinical-reports`, `latex-posters`, `pptx-posters`, `research-grants`, `infographics`, `parallel-web`, etc. — out of scope for this paper.

**Path D status.** Figure scripts identified but not yet copied to `scripts/`; the project's existing `.tif` figures from prior work are sufficient. Documented for future reuse.

**Workflow that worked:**

1. Inspect upstream `references/claude-scientific-writer/`.
2. Copy six skills into `.claude/skills/`.
3. Build `PAPER_SPEC_PLAN.md` with verified facts from code (`data/processor_cli.py`, `src/core/main.py`, Homer3 `.m` script).
4. Section-by-section: `research-lookup` → save to `sources/` → `scientific-writing` → save to `drafts/v1_<section>.md` → log progress.
5. Track all 14 section drafts to completion. Citations queued for `citation-management` pass.
6. Defer `main.tex` (IEEE template), `references.bib`, and Abstract to the final pass.

**Lesson learned.** The biggest leverage was the spec plan, not the tooling. Skills produce uniform output only when fed identical ground-truth context per session. Without `PAPER_SPEC_PLAN.md`, two sessions wrote slightly different age ranges and effect sizes — the spec plan eliminated that drift.

---

## Quick-Start Checklist for a New Project

```
[ ] Clone references/claude-scientific-writer/ as upstream reference
[ ] Decide install path (A: copy / B: plugin / +C / +D)
[ ] Copy required skills into .claude/skills/
[ ] Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY (and PARALLEL_API_KEY if available)
[ ] Add .env to .gitignore
[ ] Create research/<project>/ with subfolders: drafts/, references/, figures/, sources/, final/, data/
[ ] Write <PROJECT>_SPEC_PLAN.md with verified ground-truth facts
[ ] Smoke-test: @research-lookup Save to research/<project>/sources/smoke_test.md
[ ] (Path D) Copy scripts/schematics + scripts/generate-image, pip install requests
[ ] First section: skeleton → research-lookup → scientific-writing → log
```

---

## References

- Upstream repo: https://github.com/K-Dense-AI/claude-scientific-writer
- Project clone (read-only): `references/claude-scientific-writer/`
- Upstream `CLAUDE.md` (full policy): `references/claude-scientific-writer/CLAUDE.md`
- Upstream `README.md` (features + examples): `references/claude-scientific-writer/README.md`
- Plugin template: `references/claude-scientific-writer/templates/CLAUDE.scientific-writer.md`
- Init command source: `references/claude-scientific-writer/commands/scientific-writer-init.md`
- This project's spec plan: `research/paper-materials/PAPER_SPEC_PLAN.md`
