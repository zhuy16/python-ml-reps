# 🧠 Python ML Reps

**Active-recall training for ML, Pandas & Python — 10 topics, code editor, hint/solution panel.**

* Created by **[Yunhua Zhu](https://www.linkedin.com/in/zhu-yunhua/)**
  * GitHub: https://github.com/zhuy16
* Free software: MIT License

---

## Why

Speed in data science interviews and day-to-day work comes from **muscle memory**, not just knowing concepts.
This app gives you:

- **Active recall** — you see a prompt and have to produce the code from memory
- **Randomness** — questions appear in random order so you can't rely on sequence
- **Immediate feedback** — run your code inline, then compare against the hint and full solution
- **Spaced repetition** — mark questions Easy ✅ or Hard 🔴, then drill hard-only mode to focus on weak spots
- **Miss tracking** — the app counts how many times you've marked each topic Hard so you can see your blind spots

---

## What's inside

10 practice topics across 3 categories:

| # | Topic | Category |
|---|-------|----------|
| 1 | K-Means Clustering | 🔵 ML/Statistics |
| 2 | Logistic Regression | 🔵 ML/Statistics |
| 3 | PCA — Dimensionality Reduction | 🔵 ML/Statistics |
| 4 | Train/Test Split + Pipeline | 🔵 ML/Statistics |
| 5 | Pandas GroupBy & Aggregation | 🟢 Pandas/EDA |
| 6 | Merging Tables (SQL-style Joins) | 🟢 Pandas/EDA |
| 7 | Handling Missing Values | 🟢 Pandas/EDA |
| 8 | Plotting with Matplotlib & Seaborn | 🟢 Pandas/EDA |
| 9 | Simple ML Metrics | 🔵 ML/Statistics |
| 10 | Parsing Messy Files | 🟡 Python Basics |

**App layout:**

```
┌─────────────────┬──────────────────────────┬─────────────────────┐
│  LEFT           │  CENTER                  │  RIGHT              │
│  Task prompt    │  Ace code editor         │  💡 Show Hint       │
│  Category badge │  ▶ Run Code (inline)     │  🔍 Show Solution   │
│  ⏱️ Timer       │  Output + plots          │  ✅ Mark Easy       │
│  🎲 Next Random │  📊 Miss tracker         │  🔴 Mark Hard       │
│  📌 Jump to ... │  🗂️ Topic grid buttons   │  🔴 Hard-only mode  │
└─────────────────┴──────────────────────────┴─────────────────────┘
```

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/zhuy16/python-ml-reps.git
cd python-ml-reps
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run apps/streamlit_app.py
```

The app opens automatically at **http://localhost:8501**.

> If port 8501 is already in use, run on a different port:
> ```bash
> streamlit run apps/streamlit_app.py --server.port 8502
> ```

---

## How to use

1. **Read the prompt** on the left — understand what you need to implement
2. **Write your solution** in the Ace editor (centre) — Tab inserts 4 spaces, syntax highlighting included
3. **Click ▶ Run Code** — `print()` output and matplotlib plots render inline
4. **Peek at 💡 Hint** on the right if you're stuck (shows key imports/approach)
5. **Reveal 🔍 Solution** to compare your answer against the full working code
6. **Rate it** — ✅ Easy if you nailed it, 🔴 Hard if you struggled
7. **Click 🎲 Next Random** or pick a specific topic from the dropdown / grid buttons
8. **Enable 🔴 Hard-Only mode** to drill only your weak spots

---

## Requirements

```
streamlit
streamlit-ace
scikit-learn
pandas
seaborn
matplotlib
numpy
```

---

## Author

Built in 2026 by [Yunhua Zhu](https://github.com/zhuy16).
