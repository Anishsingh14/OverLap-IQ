#!/usr/bin/env python3
"""
================================================================================
 OverLap-IQ
 High-Dimensional Document Duplicate & Plagiarism Scanner
 Method: Semantic Embeddings (or TF-IDF fallback) + PCA + Cosine Similarity
================================================================================

WHAT THIS DOES
---------------
1. Asks you (interactively) for a document to scan (.pdf or .txt).
2. Loads a bundled repository of reference documents (Reference_Topics/ folder) --
   no manual downloads needed, everything is self-contained.
3. Vectorizes all text using SEMANTIC SENTENCE EMBEDDINGS (all-MiniLM-L6-v2),
   which capture MEANING rather than exact word overlap -- so a passage
   reworded with completely different vocabulary is still caught, as long
   as it means the same thing. If the embedding model can't be loaded (no
   internet on first run, or the optional package isn't installed), the
   tool automatically falls back to TF-IDF so it still works offline.
4. Reduces dimensionality using PCA.
5. Computes Cosine Similarity between your document and every reference
   document in the repository.
6. Prints a ranked similarity report and a verdict.
7. Saves 4 visualization charts to the output/ folder.

WHY EMBEDDINGS FIX THE "DIFFERENT VOCABULARY" PROBLEM
-------------------------------------------------------
TF-IDF only recognizes documents as similar if they share actual words.
"The feline rested on the mat" and "The cat slept on the rug" share almost
no vocabulary, so TF-IDF sees them as unrelated. A sentence embedding model,
trained on millions of paraphrase pairs, maps both sentences to nearly the
same point in vector space -- because it has learned MEANING, not just
word identity. That's what closes this gap.

HOW TO RUN
-----------
    python main.py

Then, when prompted, type/paste the path to a .pdf or .txt file. To try it
immediately, use the bundled sample:

    sample_input/Sample_Document.txt

NOTE: the first time you run this with sentence-transformers installed, it
will download the small (~80 MB) embedding model once and cache it locally.
Every run after that is fully offline.
================================================================================
"""

import os
import re
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for any terminal/IDE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# PDF support is optional at import time so .txt files always work even if
# PyMuPDF is not yet installed.
try:
    import pymupdf  # PyMuPDF (modern import name; falls back to legacy 'fitz' below)
    PDF_SUPPORT = True
except ImportError:
    try:
        import fitz as pymupdf  # older PyMuPDF versions only expose 'fitz'
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False


# --------------------------------------------------------------------------
# Paths / Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, "Reference_Topics")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Small, fast, well-regarded sentence embedding model (~80 MB, 384 dimensions).
# Downloaded once on first use and cached locally afterward.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def try_load_embedding_model():
    """
    Attempts to load the semantic embedding model. Returns the model on
    success, or None if unavailable (package not installed, no internet
    on first run, or any other loading error) -- in which case the caller
    falls back to TF-IDF so the tool never hard-fails.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return model
    except Exception:
        return None


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------
def extract_text_from_pdf(path):
    if not PDF_SUPPORT:
        raise RuntimeError(
            "PyMuPDF is not installed, so PDF files can't be read.\n"
            "Install it with:  pip install pymupdf"
        )
    text_parts = []
    doc = pymupdf.open(path)
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError("Unsupported file type. Please provide a .pdf or .txt file.")


def load_repository(repo_dir):
    filepaths = sorted(glob.glob(os.path.join(repo_dir, "*.txt")))
    if not filepaths:
        raise FileNotFoundError(
            f"No repository documents found in '{repo_dir}'.\n"
            "Run 'python generate_repository.py' first to build it."
        )
    names = [os.path.basename(fp) for fp in filepaths]
    texts = [extract_text_from_txt(fp) for fp in filepaths]
    return names, texts


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def clean_text_for_tfidf(text):
    """Lowercase and strip everything except letters/spaces.
    Stopword removal is handled separately by TfidfVectorizer(stop_words='english').
    Used only for the TF-IDF fallback path."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_embedding(text):
    """Minimal cleanup only -- just collapse whitespace. Sentence embedding
    models are trained on natural text (with case and punctuation intact),
    so aggressive cleaning actually hurts their accuracy rather than helping."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------
# Interactive input
# --------------------------------------------------------------------------
def get_user_file():
    print("=" * 72)
    print("  OverLap-IQ")
    print("  High-Dimensional Document Duplicate & Plagiarism Scanner")
    print("  (Semantic Embeddings / TF-IDF  +  PCA  +  Cosine Similarity)")
    print("=" * 72)
    print("\nTip: try the bundled sample -> sample_input/Sample_Document.txt\n")

    while True:
        path = input("Enter the path to your document (.pdf or .txt): ").strip().strip('"').strip("'")
        if not path:
            print("  -> Please enter a file path.\n")
            continue
        if not os.path.isfile(path):
            print(f"  -> Not a valid file: '{path}'. Please check the path and try again.\n")
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".pdf", ".txt"):
            print("  -> Unsupported file type. Please provide a .pdf or .txt file.\n")
            continue
        return path


# --------------------------------------------------------------------------
# Classification thresholds
# --------------------------------------------------------------------------
def classify(score):
    if score >= 0.85:
        return "LIKELY DUPLICATE / PLAGIARIZED"
    elif score >= 0.60:
        return "MODERATELY SIMILAR"
    elif score >= 0.35:
        return "SLIGHTLY SIMILAR"
    else:
        return "DISTINCT"


# --------------------------------------------------------------------------
# Visualizations -- OverLap-IQ brand style
# --------------------------------------------------------------------------
BRAND = {
    "primary": "#4C5FD5",    # indigo -- repository / baseline data
    "accent": "#E5484D",     # coral red -- high similarity / query highlight
    "warning": "#F5A623",    # amber -- moderate similarity
    "muted": "#94A3B8",      # slate gray -- low / distinct
    "text": "#1E293B",       # dark slate -- titles & labels
    "subtext": "#64748B",    # medium slate -- subtitles & annotations
    "grid": "#E2E8F0",       # light gray -- gridlines
    "bg": "#FFFFFF",
}


def _apply_brand_style():
    plt.rcParams.update({
        "figure.facecolor": BRAND["bg"],
        "axes.facecolor": BRAND["bg"],
        "savefig.facecolor": BRAND["bg"],
        "axes.edgecolor": BRAND["grid"],
        "axes.labelcolor": BRAND["text"],
        "xtick.color": BRAND["subtext"],
        "ytick.color": BRAND["subtext"],
        "text.color": BRAND["text"],
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.grid": True,
        "grid.color": BRAND["grid"],
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
    })


def _brand_header(fig, title, subtitle):
    """Consistent two-line branded header used on every chart."""
    fig.text(0.045, 0.965, "OverLap-IQ", fontsize=12, fontweight="bold",
              color=BRAND["primary"], ha="left", va="top")
    fig.text(0.045, 0.925, title, fontsize=15, fontweight="bold",
              color=BRAND["text"], ha="left", va="top")
    fig.text(0.045, 0.895, subtitle, fontsize=10, color=BRAND["subtext"],
              ha="left", va="top", style="italic")


def _brand_footer(fig):
    fig.text(0.985, 0.012, "Generated by OverLap-IQ  |  TF-IDF or Semantic Embeddings + PCA + Cosine Similarity",
              fontsize=7.5, color=BRAND["muted"], ha="right", va="bottom")


def _tier_color(score):
    if score >= 0.85:
        return BRAND["accent"]
    elif score >= 0.60:
        return BRAND["warning"]
    else:
        return BRAND["muted"]


def generate_visualizations(repo_pca, query_pca, repo_names, similarities, pca, order):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _apply_brand_style()

    # Guard against a repository small enough that PCA only produced 1
    # dimension (e.g. a 1-document repository) -- pad a zero second axis
    # so the 2D scatter plot never crashes on repo_pca[:, 1].
    n_dims = repo_pca.shape[1]
    x_repo, x_query = repo_pca[:, 0], query_pca[:, 0]
    if n_dims > 1:
        y_repo, y_query = repo_pca[:, 1], query_pca[:, 1]
        y_label = "Principal Component 2"
    else:
        y_repo, y_query = np.zeros_like(x_repo), np.zeros_like(x_query)
        y_label = "Principal Component 2 (n/a -- only 1 component available)"

    # ---- 1. 2D PCA Scatter Plot -----------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.subplots_adjust(top=0.83, bottom=0.09, left=0.09, right=0.97)

    ax.scatter(
        x_repo, y_repo,
        c=BRAND["primary"], s=100, alpha=0.8,
        label="Repository Documents", edgecolors="white", linewidths=0.8, zorder=3,
    )
    ax.scatter(
        x_query, y_query,
        c=BRAND["accent"], s=320, marker="*",
        label="Your Document", edgecolors="white", linewidths=1.0, zorder=5,
    )
    for i, name in enumerate(repo_names):
        ax.annotate(
            name.replace(".txt", ""), (x_repo[i], y_repo[i]),
            fontsize=7.5, color=BRAND["subtext"],
            xytext=(5, 5), textcoords="offset points", annotation_clip=False,
        )
    ax.margins(x=0.18, y=0.12)
    ax.axhline(0, color=BRAND["grid"], linewidth=0.8, zorder=1)
    ax.axvline(0, color=BRAND["grid"], linewidth=0.8, zorder=1)
    ax.set_xlabel("Principal Component 1", fontsize=10.5)
    ax.set_ylabel(y_label, fontsize=10.5)
    legend = ax.legend(loc="best", frameon=True, fontsize=9.5)
    legend.get_frame().set_edgecolor(BRAND["grid"])
    legend.get_frame().set_linewidth(0.8)

    _brand_header(fig, "Document Similarity Map",
                  "Where your document lands relative to the reference repository, in 2D meaning-space")
    _brand_footer(fig)
    fig.savefig(os.path.join(OUTPUT_DIR, "1_pca_scatter.png"), dpi=160)
    plt.close(fig)

    # ---- 2. Top-N Similarity Bar Chart -----------------------------------
    top_n = min(10, len(repo_names))
    top_idx = order[:top_n]
    top_names = [repo_names[i].replace(".txt", "") for i in top_idx]
    top_scores = [similarities[i] for i in top_idx]
    colors = [_tier_color(s) for s in top_scores]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.22, right=0.95)

    bars = ax.barh(top_names[::-1], top_scores[::-1], color=colors[::-1],
                    edgecolor="white", linewidth=0.6, height=0.62, zorder=3)
    left_lim = min(0, min(top_scores)) - 0.14
    ax.set_xlim(left_lim, 1.08)
    ax.axvline(0, color=BRAND["text"], linewidth=0.9, zorder=2)

    for threshold, label in [(0.35, ""), (0.60, ""), (0.85, "")]:
        ax.axvline(threshold, color=BRAND["grid"], linewidth=1.0, linestyle="--", zorder=1)

    for bar in bars:
        width = bar.get_width()
        label_x = width + 0.018 if width >= 0 else width - 0.018
        ha = "left" if width >= 0 else "right"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
                 va="center", ha=ha, fontsize=8.5, color=BRAND["text"], fontweight="medium")

    ax.set_xlabel("Cosine Similarity Score", fontsize=10.5)

    legend_handles = [
        mpatches.Patch(color=BRAND["accent"], label="≥ 0.85  Likely Duplicate"),
        mpatches.Patch(color=BRAND["warning"], label="0.60–0.85  Moderately Similar"),
        mpatches.Patch(color=BRAND["muted"], label="< 0.60  Slightly Similar / Distinct"),
    ]
    legend = ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8.5)
    legend.get_frame().set_edgecolor(BRAND["grid"])

    _brand_header(fig, f"Top {top_n} Most Similar Documents",
                  "Ranked cosine similarity against every reference document in the repository")
    _brand_footer(fig)
    fig.savefig(os.path.join(OUTPUT_DIR, "2_top_matches_bar.png"), dpi=160)
    plt.close(fig)

    # ---- 3. Pairwise Similarity Heatmap (top matches) --------------------
    n_show = min(12, len(repo_names))
    show_idx = order[:n_show]
    subset = repo_pca[show_idx]
    sim_matrix = cosine_similarity(subset)
    labels = [repo_names[i].replace(".txt", "") for i in show_idx]

    brand_cmap = LinearSegmentedColormap.from_list(
        "overlapiq", ["#FFFFFF", BRAND["primary"], BRAND["accent"]]
    )

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.20, right=0.92)

    im = ax.imshow(sim_matrix, cmap=brand_cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n_show))
    ax.set_yticks(range(n_show))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(False)

    # Annotate only meaningfully similar cells to keep the chart readable
    for r in range(n_show):
        for c in range(n_show):
            val = sim_matrix[r, c]
            if val >= 0.3:
                txt_color = "white" if val >= 0.55 else BRAND["text"]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, label="Cosine Similarity", fraction=0.046, pad=0.03)
    cbar.outline.set_edgecolor(BRAND["grid"])

    _brand_header(fig, "Pairwise Similarity Heatmap",
                  "How similar the top matching documents are to one another")
    _brand_footer(fig)
    fig.savefig(os.path.join(OUTPUT_DIR, "3_similarity_heatmap.png"), dpi=160)
    plt.close(fig)

    # ---- 4. PCA Explained Variance (Scree Plot) --------------------------
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.09, right=0.95)

    ax.bar(range(1, len(var_ratio) + 1), var_ratio, alpha=0.85,
           label="Individual Component", color=BRAND["primary"], zorder=3,
           edgecolor="white", linewidth=0.5)
    ax.plot(range(1, len(cum_var) + 1), cum_var, color=BRAND["accent"],
            marker="o", markersize=4, linewidth=2, label="Cumulative", zorder=4)

    if cum_var.max() >= 0.9:
        idx_90 = int(np.argmax(cum_var >= 0.9)) + 1
        ax.axhline(0.9, color=BRAND["grid"], linewidth=1.0, linestyle="--", zorder=1)
        ax.annotate(f"90% variance at {idx_90} components",
                    xy=(idx_90, 0.9), xytext=(idx_90, 0.9 + 0.06),
                    fontsize=8.5, color=BRAND["subtext"], ha="center")

    ax.set_xlabel("Principal Component", fontsize=10.5)
    ax.set_ylabel("Explained Variance Ratio", fontsize=10.5)
    legend = ax.legend(loc="center right", frameon=True, fontsize=9.5)
    legend.get_frame().set_edgecolor(BRAND["grid"])

    _brand_header(fig, "PCA Explained Variance",
                  "How much information each principal component retains (scree plot)")
    _brand_footer(fig)
    fig.savefig(os.path.join(OUTPUT_DIR, "4_pca_scree_plot.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    query_path = get_user_file()
    print(f"\n[1/6] Reading document: {query_path}")
    query_raw = load_document(query_path)

    word_count = len(query_raw.split())
    if word_count < 5:
        print("  -> Warning: document seems very short; results may be unreliable.")

    print("[2/6] Loading bundled reference repository...")
    repo_names, repo_raw_texts = load_repository(REPO_DIR)
    print(f"       Loaded {len(repo_names)} reference documents.")

    print("[3/6] Vectorizing text...")
    embedding_model = try_load_embedding_model()

    if embedding_model is not None:
        vectorization_mode = "Semantic Embeddings (MiniLM)"
        print("       Using semantic sentence embeddings (all-MiniLM-L6-v2).")
        print("       This catches meaning-based matches even with completely")
        print("       different vocabulary (true paraphrase detection).")
        repo_texts = [clean_text_for_embedding(t) for t in repo_raw_texts]
        query_text = clean_text_for_embedding(query_raw)
        repo_vectors = np.asarray(embedding_model.encode(repo_texts, show_progress_bar=False))
        query_vector = np.asarray(embedding_model.encode([query_text], show_progress_bar=False))
    else:
        vectorization_mode = "TF-IDF (offline fallback)"
        print("       Semantic embedding model unavailable (no internet on first")
        print("       run, or 'sentence-transformers' isn't installed).")
        print("       Falling back to TF-IDF -- still fully functional, but word-")
        print("       overlap based. Install 'sentence-transformers' once with an")
        print("       internet connection for vocabulary-independent matching.")
        repo_texts = [clean_text_for_tfidf(t) for t in repo_raw_texts]
        query_text = clean_text_for_tfidf(query_raw)
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=1)
        repo_vectors = vectorizer.fit_transform(repo_texts).toarray()
        query_vector = vectorizer.transform([query_text]).toarray()

    print("[4/6] Reducing dimensionality with PCA...")
    n_samples, n_features = repo_vectors.shape
    # PCA requires 1 <= n_components <= min(n_samples, n_features). Clamp to
    # that valid range (capped at 50) so tiny repositories never crash --
    # e.g. a repository with just 1 or 2 reference documents.
    n_components = max(1, min(50, n_samples, n_features))
    pca = PCA(n_components=n_components, random_state=42)
    repo_pca = pca.fit_transform(repo_vectors)
    query_pca = pca.transform(query_vector)
    explained = pca.explained_variance_ratio_.sum() * 100
    component_word = "component" if n_components == 1 else "components"
    print(f"       Reduced to {n_components} {component_word} "
          f"(retaining {explained:.1f}% of variance).")

    print("[5/6] Computing cosine similarity...")
    similarities = cosine_similarity(query_pca, repo_pca)[0]
    order = np.argsort(similarities)[::-1]
    ranked_names = [repo_names[i] for i in order]
    ranked_scores = [similarities[i] for i in order]

    print("\n" + "=" * 72)
    print("  SIMILARITY REPORT")
    print(f"  Vectorization method: {vectorization_mode}")
    print("=" * 72)
    top_n = min(10, len(ranked_names))
    for rank in range(top_n):
        name = ranked_names[rank]
        score = ranked_scores[rank]
        print(f"  {rank + 1:2d}. {name:35s} score={score:.4f}   [{classify(score)}]")

    best_score = ranked_scores[0]
    best_match = ranked_names[0]
    print("-" * 72)
    print(f"  Overall verdict : {classify(best_score)}")
    print(f"  Closest match   : {best_match}  (similarity = {best_score:.4f})")
    print("=" * 72)
    print("  Note: after PCA, cosine similarity can range from -1 to 1")
    print("  (the raw vectors before PCA are typically non-negative). Negative")
    print("  or near-zero scores simply mean the documents are unrelated in topic.")

    print("\n[6/6] Generating visualizations...")
    generate_visualizations(repo_pca, query_pca, repo_names, similarities, pca, order)
    print(f"       Charts saved to: {OUTPUT_DIR}")
    print("\nDone.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
