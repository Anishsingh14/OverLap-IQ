# 🔍 OverLap-IQ

### High-Dimensional Document Duplicate & Plagiarism Scanner

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/sentence--transformers-Semantic%20Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="sentence-transformers"/>
  <img src="https://img.shields.io/badge/scikit--learn-PCA%20%2B%20Cosine-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-Data%20Viz-11557C?style=for-the-badge&logo=Matplotlib&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/PyMuPDF-PDF%20Parsing-EC1C24?style=for-the-badge" alt="PyMuPDF"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

**OverLap-IQ** is a **backend-only** command-line tool that checks whether an uploaded document (`.pdf` or `.txt`) is a **duplicate, near-duplicate, or plagiarized** version of anything in a reference repository — using **Semantic Sentence Embeddings → PCA → Cosine Similarity**, with an automatic **TF-IDF fallback** so it never breaks even without internet access.

No API keys or manual dataset downloads required. The repository of reference documents ships with the project.

> 🧠 **Catches paraphrasing with completely different vocabulary.** Unlike plain word-overlap methods, the default mode uses a pretrained sentence embedding model that understands *meaning* — so "the feline rested on the mat" and "the cat slept on the rug" are correctly flagged as near-identical, even though they share almost no words.

---

## 📖 Table of Contents

- [How It Works](#-how-it-works-in-plain-english)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Sample Output](#-sample-output)
- [Understanding the Results](#-understanding-the-results)
- [Adding Your Own Reference Documents](#-adding-your-own-reference-documents)
- [Tech Stack](#️-tech-stack)
- [Limitations & Future Improvements](#-limitations--future-improvements)
- [License](#-license)

---

## 🧠 How It Works (In Plain English)

Think of it like a teacher checking if a student copied an essay — except instead of reading every word, the computer converts *meaning* into numbers and measures how close two documents are in that "meaning space."

```
 Your Document (.pdf/.txt)
          │
          ▼
 ① Extract Text                (PyMuPDF for PDFs, plain read for .txt)
          │
          ▼
 ② Vectorization                Semantic Embeddings (all-MiniLM-L6-v2)
          │                     -- or, automatically, TF-IDF if that's
          │                        unavailable (no internet / not installed)
          ▼
 ③ PCA Dimensionality Reduction (compress the vector, drop noise/redundancy)
          │
          ▼
 ④ Cosine Similarity           (measure the "angle" between your doc and every
          │                     reference doc — small angle = similar meaning)
          ▼
 ⑤ Ranked Report + 4 Branded Charts
```

| Step | What happens | Why |
|---|---|---|
| **Text Extraction** | Pulls raw text out of your `.pdf` or `.txt` file | Computers need plain text, not page layouts |
| **Vectorization** | Converts each document into a numeric vector using a pretrained sentence embedding model (or TF-IDF as a fallback) | Gives every document a numeric "fingerprint" of its meaning |
| **PCA** | Compresses that fingerprint into fewer, more informative dimensions | Faster comparison, removes redundant noise |
| **Cosine Similarity** | Measures the angle between two document vectors | Flags near-duplicate/plagiarized content |

### Why Semantic Embeddings (Not Just TF-IDF)

TF-IDF only recognizes documents as similar if they **share actual words**. That means a passage reworded with completely different vocabulary — synonyms, restructured sentences, translated-and-back phrasing — slips right past it, even though the meaning is identical.

The default vectorization here instead uses **`all-MiniLM-L6-v2`**, a small pretrained sentence embedding model trained on millions of paraphrase pairs. It maps sentences to points in a 384-dimensional "meaning space" where paraphrases land close together *regardless of which words were used*. This directly closes the vocabulary-overlap gap that pure TF-IDF has.

**The tool is resilient by design:**

```
Try to load the embedding model
        │
   ┌────┴─────┐
 Success     Fails (no internet on first run, or
   │          package not installed)
   ▼               │
Semantic            ▼
Embeddings      Automatic fallback to TF-IDF
(best accuracy)  (still fully functional, offline)
```

Nothing ever crashes because of this — if the embedding model can't load for any reason, the tool prints a clear notice and continues with TF-IDF automatically. The very first successful run with `sentence-transformers` installed downloads the ~80 MB model once and caches it locally; every run after that is fully offline too.

> 💡 **Key insight:** Because embeddings work on true *meaning-vectors* rather than word identity, they catch a paraphrased paragraph — one that shares almost no exact words with the original — not just copy-pasted text.

---

## 📁 Project Structure

```
OverLap-IQ/
├── main.py                    # ⭐ Main executable — run this
├── generate_repository.py    # Rebuilds the bundled reference repository (optional)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── Reference_Topics/           # 20 bundled reference documents (self-contained, no download)
│   ├── climate_change_a.txt
│   ├── climate_change_b.txt
│   └── ... (18 more)
├── sample_input/
│   └── Sample_Document.txt   # Ready-to-use test document
├── sample_output/             # Example charts from a real run (for reference)
│   ├── 1_pca_scatter.png
│   ├── 2_top_matches_bar.png
│   ├── 3_similarity_heatmap.png
│   └── 4_pca_scree_plot.png
└── output/                    # Created automatically each time you run a scan
```

---

## ⚙️ Installation

**Requirements:** Python 3.9+

```bash
# 1. Clone this repository
git clone <your-repo-url>
cd OverLap-IQ

# 2. Install dependencies
pip install -r requirements.txt
```

That's it — no dataset downloads, no API keys, no external services to configure.

> **About the embedding model:** `sentence-transformers` is included in `requirements.txt` because it's what gives the tool its vocabulary-independent accuracy. The very first time you run a scan, it downloads the small `all-MiniLM-L6-v2` model (~80 MB) once from Hugging Face and caches it locally — every run after that works fully offline. If you'd rather skip this entirely (e.g. for an air-gapped machine), just don't install `sentence-transformers` — the tool detects it's missing and automatically uses TF-IDF instead, no configuration needed.

---

## ▶️ Usage

Simply run the script and follow the interactive prompt:

```bash
python main.py
```

You'll see:

```
========================================================================
  OverLap-IQ
  High-Dimensional Document Duplicate & Plagiarism Scanner
  (Semantic Embeddings / TF-IDF  +  PCA  +  Cosine Similarity)
========================================================================

Tip: try the bundled sample -> sample_input/Sample_Document.txt

Enter the path to your document (.pdf or .txt):
```

Type or paste a file path — for example, to try the bundled sample:

```
sample_input/Sample_Document.txt
```

The tool will:
1. Extract the text
2. Vectorize it with semantic embeddings (or TF-IDF, automatically, if the embedding model isn't available)
3. Compare it against all 20 bundled reference documents
4. Print a ranked similarity report to the terminal
5. Save 4 branded charts to the `output/` folder

**Want to regenerate/reset the bundled repository?**

```bash
python generate_repository.py
```

### Output Format — What You Actually Get

OverLap-IQ produces two things, every run, automatically:

| Output | Where | When |
|---|---|---|
| **Text similarity report** | Printed directly in your terminal | Instantly, as part of the run |
| **4 chart images** (`.png`) | Saved to the `output/` folder | Saved to disk — **not** shown inline in the terminal |

This is a backend/CLI tool, so nothing pops up automatically — after the run finishes, open the `output/` folder yourself to view the 4 charts (any image viewer, VS Code, or your file explorer works). The terminal report gives you the instant readable verdict; the charts give you the visual evidence behind it.

---

## 📊 Sample Output

Below is real output generated by running OverLap-IQ on `sample_input/Sample_Document.txt` (an independent paraphrase of a renewable-energy article) against the bundled repository.

> **Note on these specific sample charts:** they were generated in **TF-IDF fallback mode** (the environment used to build this package had no outbound access to Hugging Face to download the embedding model). Even so, TF-IDF alone correctly caught this particular paraphrase because it still shares a fair amount of vocabulary with the original. With `sentence-transformers` installed and an internet connection, you'll get the semantic embeddings mode instead — same report format, same charts, but stronger recall on paraphrases that share little or no vocabulary.

### Terminal Report

```
========================================================================
  SIMILARITY REPORT
  Vectorization method: TF-IDF (offline fallback)
========================================================================
   1. renewable_energy_a.txt              score=0.8999   [LIKELY DUPLICATE / PLAGIARIZED]
   2. renewable_energy_b.txt              score=0.7760   [MODERATELY SIMILAR]
   3. climate_change_a.txt                score=0.1249   [DISTINCT]
   4. climate_change_b.txt                score=0.0692   [DISTINCT]
   ...
------------------------------------------------------------------------
  Overall verdict : LIKELY DUPLICATE / PLAGIARIZED
  Closest match   : renewable_energy_a.txt  (similarity = 0.8999)
========================================================================
```

Notice it correctly identifies the two renewable-energy reference documents as the closest matches — even though **not a single sentence is copied word-for-word**. It caught the paraphrase. (The report always prints which vectorization method was actually used, so you always know whether you got the embeddings-based or TF-IDF-based result.)

### 1. Document Similarity Map (PCA)

Every document plotted as a point in 2D "meaning space." Your document (★) lands right next to its true matches — visual proof the tool understands semantic closeness, not just keyword overlap.

![PCA Scatter Plot](sample_output/1_pca_scatter.png)

### 2. Top Matches Bar Chart

A quick-glance ranking of the most similar documents, color-coded by verdict tier (duplicate / moderately similar / distinct) with threshold guide lines.

![Top Matches Bar Chart](sample_output/2_top_matches_bar.png)

### 3. Pairwise Similarity Heatmap

Shows how similar *every* top-matching document is to *every other* one, with exact scores annotated on meaningfully similar cells — useful for spotting clusters of duplicate/near-duplicate content across your whole repository at a glance.

![Similarity Heatmap](sample_output/3_similarity_heatmap.png)

### 4. PCA Scree Plot

Shows how much information (variance) each principal component captures, with the point where 90% of variance is retained called out directly on the chart — this is what justifies *how many* dimensions PCA keeps.

![Scree Plot](sample_output/4_pca_scree_plot.png)

---

## 🧾 Understanding the Results

| Similarity Score | Verdict | Meaning |
|---|---|---|
| **≥ 0.85** | 🔴 Likely Duplicate / Plagiarized | Very likely the same content, paraphrased or copied |
| **0.60 – 0.85** | 🟠 Moderately Similar | Shares significant themes/structure — worth a manual look |
| **0.35 – 0.60** | 🔵 Slightly Similar | Some topical overlap, probably coincidental |
| **< 0.35** | ⚪ Distinct | Unrelated content |

> **Note:** After PCA, cosine similarity can range from **-1 to 1** (the raw vectors before PCA are typically non-negative). A negative score just means "unrelated," not "opposite" — there's no real opposite of a topic.

You can tune these thresholds directly in `main.py` inside the `classify()` function.

---

## ➕ Adding Your Own Reference Documents

The bundled repository (20 documents across 12 topics) exists purely to make this project runnable out-of-the-box with zero setup. To build a real-world repository:

1. Drop additional `.txt` files into the `Reference_Topics/` folder, **or**
2. Edit the `DOCUMENTS` dictionary in `generate_repository.py` and re-run it, **or**
3. Point `REPO_DIR` in `main.py` to your own folder of `.txt` files (e.g., a folder of previously submitted essays or published articles).

The more documents in your repository, the more meaningful TF-IDF and PCA become — both need a reasonably sized, diverse corpus to build a useful vocabulary and variance structure. That said, OverLap-IQ is built to degrade gracefully rather than crash on very small repositories (even a single reference document) — PCA dimensionality is automatically clamped to whatever is mathematically valid for the data you give it, and the charts adapt their axes accordingly.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **sentence-transformers** | Semantic embedding model (`all-MiniLM-L6-v2`) — primary vectorization method |
| **scikit-learn** | TF-IDF fallback vectorization, PCA, cosine similarity |
| **NumPy** | Numerical operations |
| **Matplotlib** | All 4 branded visualizations |
| **PyMuPDF (fitz)** | PDF text extraction |

---

## 🚧 Limitations & Future Improvements

The vocabulary-mismatch problem is addressed by the semantic embedding mode, but a few other trade-offs are worth knowing:

- **Document-level, not sentence-level.** The current version compares whole documents. For pinpointing *which paragraph* was copied, add a chunking step (split into sentences/paragraphs before vectorizing) and compare chunk-by-chunk — the embedding model already works at the sentence level internally, so this is a natural extension.
- **No OCR support.** Scanned (image-based) PDFs won't extract text. Adding `pytesseract` + `pdf2image` would cover this case.
- **Small bundled repository.** 20 documents is enough to demonstrate the pipeline correctly, but a production system should be tested against a much larger, domain-specific corpus.
- **One-time model download.** The semantic embeddings mode needs internet access exactly once, on its very first use, to fetch `all-MiniLM-L6-v2`. On a machine with no internet access at all, the tool still works correctly via the automatic TF-IDF fallback — just with the word-overlap limitation TF-IDF inherently has.

---

## 📄 License

This project is released under the MIT License — free to use, modify, and build on.
