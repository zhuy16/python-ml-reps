# Installation

## Requirements

- Python 3.9+
- The dependencies listed in `requirements.txt`:

```
streamlit
streamlit-ace
scikit-learn
pandas
seaborn
matplotlib
numpy
lifelines
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/zhuy16/python-ml-reps.git
cd python-ml-reps
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run apps/streamlit_app.py
```

The app opens at **http://localhost:8501**.

> To use a different port:
> ```bash
> streamlit run apps/streamlit_app.py --server.port 8502
> ```
