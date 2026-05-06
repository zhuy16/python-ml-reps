# Usage

## Workflow

1. **Select a bucket** — choose `🔵 Basic`, `🧬 Bioinformatics Engineer`, `🏥 Clinical DS`, or `🧩 Integrated` from the radio selector at the top
2. **Read the prompt** — understand what you need to implement (left panel)
3. **Write from memory** — type your solution in the Ace code editor (centre)
4. **Watch the diff** — `🔍 Diff: My Code vs Solution` updates automatically; each line you get right disappears from the diff
5. **Run it** — click **▶ Run Code**; `print()` output and matplotlib plots render inline. Shortcuts: `Ctrl+Enter` / `Cmd+Enter` / `Shift+Enter`.
6. **Run reference** — click **🧪 Run Solution** to execute solution code directly for verification
7. **Toggle solution** — press `Ctrl+S` (or `Cmd+S`) or click **✅ Solution** to load the full solution into the editor; your WIP is preserved. Press again to switch back.
8. **Hard mode diff** — toggle `🧠 Hard mode (hide diff)` to collapse the diff and force genuine recall before peeking
9. **Rate it** — **✅ Easy** if you nailed it, **🔴 Hard** if you struggled
10. **Add a note** — type a mnemonic or key insight in **📌 My Notes** (right panel); persists per question
11. **Navigate** — use the `🗂️ Topics` grid at the bottom of the left panel to jump to any question
12. **Promote tested code** — after a successful run, click **⭐ Use My Code as Solution** if you want your own tested version to become the default solution for that question

## Drilling weak spots

- Enable **🔴 Hard-Only mode** to see only questions you have marked Hard
- **Spaced repetition**: question selection uses weighted sampling so Hard questions appear more often; miss counts boost priority further
- The **miss tracker** shows how many times each question was marked Hard this session

## Bucket guide

| Bucket | Focus | Questions |
|--------|-------|-----------|
| 🔵 Basic | Core ML, Pandas, Python | 10 |
| 🧬 Bioinformatics Engineer | DNA parsing, VCF/BED/FASTQ, variant tables, intervals | 17 |
| 🏥 Clinical DS | Clinical pandas, sklearn pipeline, KM curves, Cox PH | 13 |
| 🧩 Integrated | End-to-end realistic drills (VCF parse → clean → EDA → merge → plot) | 1 |

## Tips

- The editor supports **Tab → 4 spaces** and Python syntax highlighting
- All solutions are self-contained — they define their own data so you can copy-paste and run anywhere
- For survival analysis questions, `lifelines` must be installed (`pip install lifelines`)
