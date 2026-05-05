# Usage

## Workflow

1. **Select a bucket** — choose `🔵 Basic`, `🧬 Bioinformatics Engineer`, or `🏥 Clinical DS` from the radio selector at the top
2. **Read the prompt** — understand what you need to implement (left panel)
3. **Write from memory** — type your solution in the Ace code editor (centre)
4. **Watch the diff** — `🔍 Diff: My Code vs Solution` updates automatically; each line you get right disappears from the diff
5. **Run it** — click **▶ Run Code**; `print()` output and matplotlib plots render inline
6. **Toggle solution** — press `Ctrl+S` (or `Cmd+S`) or click **✅ Solution** to load the full solution into the editor; your WIP is preserved. Press again to switch back.
7. **Hard mode diff** — toggle `🧠 Hard mode (hide diff)` to collapse the diff and force genuine recall before peeking
8. **Rate it** — **✅ Easy** if you nailed it, **🔴 Hard** if you struggled
9. **Add a note** — type a mnemonic or key insight in **📌 My Notes** (right panel); persists per question
10. **Navigate** — use the `🗂️ Topics` grid at the bottom of the left panel to jump to any question

## Drilling weak spots

- Enable **🔴 Hard-Only mode** to see only questions you have marked Hard
- **Spaced repetition**: `🎲 Next Random` (via topic grid navigation) weights Hard questions 4× more than Easy; miss counts boost priority further
- The **miss tracker** shows how many times each question was marked Hard this session

## Bucket guide

| Bucket | Focus | Questions |
|--------|-------|-----------|
| 🔵 Basic | Core ML, Pandas, Python | 10 |
| 🧬 Bioinformatics Engineer | DNA parsing, VCF/BED/FASTQ, variant tables, intervals | 17 |
| 🏥 Clinical DS | Clinical pandas, sklearn pipeline, KM curves, Cox PH | 12 |

## Tips

- The editor supports **Tab → 4 spaces** and Python syntax highlighting
- All solutions are self-contained — they define their own data so you can copy-paste and run anywhere
- For survival analysis questions, `lifelines` must be installed (`pip install lifelines`)
