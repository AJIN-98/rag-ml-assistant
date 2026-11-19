"""
ML Knowledge Assistant - a small RAG app over "Introduction to Machine Learning with Python".

Usage:
    pip install pypdf sentence-transformers faiss-cpu google-genai gradio
    export GEMINI_API_KEY=...            # or: export LLM_PROVIDER=openai OPENAI_API_KEY=...
    python app.py                        # web UI
    python app.py --cli                  # plain terminal Q&A

The first run reads the PDF, chunks it, embeds the chunks and saves the index to ./rag_index.
Later runs load that index straight away.
"""


import os, re, json, time
import numpy as np

# ---- settings you can change ----
PDF_PATH      = "intro-to-ml-book.pdf"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"   # 384-dim, 512-token limit, runs fine on CPU
QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "  # recommended by the BGE authors for queries
CHUNK_SIZE    = 500      # tokens (counted with the embedding model's own tokenizer)
CHUNK_OVERLAP = 100      # tokens
TOP_K         = 5

LLM_PROVIDER  = os.environ.get("LLM_PROVIDER", "gemini")   # "gemini" or "openai"
LLM_MODEL     = os.environ.get("LLM_MODEL", "gemini-2.5-flash" if LLM_PROVIDER == "gemini" else "gpt-4o-mini")
INDEX_DIR     = "rag_index"


from pypdf import PdfReader

def load_pdf(path):
    """Return a list of (pdf_page_number, raw_text) - page numbers start at 1."""
    reader = PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


# Layout facts about this particular book (found while exploring it):
FIRST_BODY_PAGE = 9      # PDF page where the Preface starts (before it: cover, copyright, table of contents)
LAST_BODY_PAGE  = 380    # PDF page 381 is where the alphabetical index begins
PRINTED_OFFSET  = 14     # printed page number = PDF page - 14 (Chapter 1 starts on PDF page 15 = printed page 1)

CHAPTER_TITLES = {
    1: "Introduction",
    2: "Supervised Learning",
    3: "Unsupervised Learning and Preprocessing",
    4: "Representing Data and Engineering Features",
    5: "Model Evaluation and Improvement",
    6: "Algorithm Chains and Pipelines",
    7: "Working with Text Data",
    8: "Wrapping Up",
}

# Every page ends with a running footer, e.g. "Supervised Machine Learning Algorithms | 87",
# "88 | Chapter 2: Supervised Learning", or just a bare page number on chapter-opening pages.
FOOTER_RE = re.compile(r"^(?:(?:\d+|[ivxlc]+)\s*\|\s*.+|.+\s*\|\s*(?:\d+|[ivxlc]+)|\d+|[ivxlc]+)$")


def clean_page(text):
    """Remove the footer, re-join hyphenated words and fix the 'Y ou' kerning glitch.
    Code is left alone on purpose (indentation matters and it is useful to retrieve)."""
    lines = text.rstrip().split("\n")
    if lines and FOOTER_RE.match(lines[-1].strip()):
        lines.pop()
    text = "\n".join(lines)
    text = text.replace("\u2010\n", "")                       # "classifica-\ntion" -> "classification"
    text = re.sub(r"\bY (ou(?:r|rs|rself|rselves|Tube)?)\b", r"Y\1", text)  # "Y ou" -> "You"
    text = re.sub(r"[ \t]+\n", "\n", text)                    # trailing spaces
    return text.strip()


def page_label(start, end):
    """Printed page label used in citations."""
    if end < 1:
        return "front matter"
    start = max(start, 1)
    return str(start) if start == end else f"{start}-{end}"


def build_lines(raw_pages):
    """Turn raw PDF pages into a list of body lines, each tagged with chapter and page.
    Returns: list of dicts {text, chapter, pdf_page}"""
    lines, chapter = [], "Preface"
    dangling = None                       # a word that was split across a page break
    for pno, raw in raw_pages:
        if pno < FIRST_BODY_PAGE or pno > LAST_BODY_PAGE:
            continue
        m = re.match(r"\s*CHAPTER (\d+)\b", raw)
        if m:
            chapter = f"Chapter {m.group(1)}: {CHAPTER_TITLES[int(m.group(1))]}"
        cleaned = clean_page(raw)
        if len(cleaned) < 50:             # blank / part-divider pages
            continue
        page_lines = [ln for ln in cleaned.split("\n") if ln.strip()]
        if dangling is not None:          # glue "evalu-" + "ation ..." back together
            dangling["text"] = dangling["text"][:-1] + page_lines.pop(0)
            dangling = None
        for ln in page_lines:
            lines.append({"text": ln, "chapter": chapter, "pdf_page": pno})
        if lines and lines[-1]["text"].endswith("\u2010"):
            dangling = lines[-1]
    return lines


SENT_END = re.compile(r"[.!?:)\]]\s*$")

def chunk_lines(lines, count_tokens, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Pack lines into chunks of at most `size` tokens with ~`overlap` tokens shared between
    neighbours. Chunks never cross a chapter boundary, and a chunk prefers to stop at the end
    of a sentence (if one falls in the last 40% of the window) instead of mid-sentence.

    count_tokens: function taking a list of strings and returning a list of token counts."""
    chunks = []
    # group consecutive lines by chapter
    groups, current = [], []
    for ln in lines:
        if current and ln["chapter"] != current[0]["chapter"]:
            groups.append(current); current = []
        current.append(ln)
    if current:
        groups.append(current)

    for grp in groups:
        tok = count_tokens([l["text"] for l in grp])
        n, i = len(grp), 0
        while i < n:
            j, total = i, 0
            while j < n and total + tok[j] <= size:
                total += tok[j]; j += 1
            if j == i:                       # one enormous line - take it alone
                j = i + 1
            cut = j
            if j < n:                        # try to end on a sentence boundary
                for k in range(j, i, -1):
                    if sum(tok[i:k]) < 0.6 * size:
                        break
                    if SENT_END.search(grp[k - 1]["text"]):
                        cut = k; break
            piece = grp[i:cut]
            start, end = piece[0]["pdf_page"] - PRINTED_OFFSET, piece[-1]["pdf_page"] - PRINTED_OFFSET
            chunks.append({
                "id": len(chunks),
                "chapter": grp[0]["chapter"],
                "page_start": start, "page_end": end,
                "page_label": page_label(start, end),
                "n_tokens": sum(tok[i:cut]),
                "text": "\n".join(l["text"] for l in piece),
            })
            if cut >= n:
                break
            nxt, acc = cut, 0                # step back to create the overlap
            while nxt - 1 > i and acc + tok[nxt - 1] <= overlap:
                nxt -= 1; acc += tok[nxt]
            i = nxt
    return chunks


class Retriever:
    """Embeds the chunks, keeps them in a FAISS index and answers similarity searches."""

    def __init__(self, embedder, chunks, embeddings=None):
        import faiss
        self.embedder, self.chunks = embedder, chunks
        if embeddings is None:
            embeddings = embedder.encode([c["text"] for c in chunks], batch_size=32,
                                         normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])   # inner product == cosine for unit vectors
        self.index.add(embeddings)
        self.embeddings = embeddings

    def search(self, query, k=TOP_K):
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [dict(self.chunks[i], score=float(s), rank=r + 1)
                for r, (s, i) in enumerate(zip(scores[0], ids[0])) if i != -1]

    def save(self, folder=INDEX_DIR):
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "embeddings.npy"), self.embeddings)
        with open(os.path.join(folder, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)

    @classmethod
    def load(cls, embedder, folder=INDEX_DIR):
        with open(os.path.join(folder, "chunks.json")) as f:
            chunks = json.load(f)
        return cls(embedder, chunks, np.load(os.path.join(folder, "embeddings.npy")))


SYSTEM_PROMPT = """You are a teaching assistant for the book "Introduction to Machine Learning with Python" by Andreas Mueller and Sarah Guido.

Answer the student's question using ONLY the numbered context passages that are provided.

Rules:
1. If the passages do not contain enough information to answer the question, reply exactly: "I couldn't find that in the book." and stop. Do not guess and do not use outside knowledge, even if you know the answer.
2. After each claim, cite the passage(s) it came from with the passage number in square brackets, like [1] or [2][4].
3. Keep the answer short and clear: one short paragraph, or a few bullet points if the question has several parts. Only include code if it appears in the passages.
4. If the passages answer only part of the question, answer that part and say what is missing."""


def build_prompt(query, hits):
    blocks = [f"[{n}] ({h['chapter']}, p. {h['page_label']})\n{h['text']}" for n, h in enumerate(hits, 1)]
    return ("Context passages:\n\n" + "\n\n".join(blocks) +
            f"\n\nQuestion: {query}\n\nAnswer (use only the passages above and cite them like [1]):")


def generate(prompt):
    """Send the prompt to the LLM. temperature=0 so answers stay as close to the context as possible."""
    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(
            model=LLM_MODEL, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0))
        return resp.text.strip()
    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()                      # reads OPENAI_API_KEY
        resp = client.chat.completions.create(
            model=LLM_MODEL, temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()
    raise ValueError("LLM_PROVIDER must be 'gemini' or 'openai'")


_STOP = set("this that with from have been were will would could should also which their there these those about into than then them they what when where while your does done only more most such some other being because between using used uses each both very much many".split())

def grounding_report(answer, hits):
    """Cheap grounding evidence: which passages were cited, and how many of the answer's content
    words actually occur in the retrieved text (a lexical proxy, not a proof)."""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(hits)})
    context_words = set(re.findall(r"[a-z_]{4,}", " ".join(h["text"] for h in hits).lower()))
    answer_words = [w for w in re.findall(r"[a-z_]{4,}", re.sub(r"\[\d+\]", "", answer.lower())) if w not in _STOP]
    overlap = sum(w in context_words for w in answer_words) / max(1, len(answer_words))
    return {"cited_passages": cited, "context_word_overlap": round(overlap, 2)}


def ask(query, k=TOP_K):
    hits = retriever.search(query, k)
    answer = generate(build_prompt(query, hits))
    return {"query": query, "answer": answer, "sources": hits, "grounding": grounding_report(answer, hits)}


def show(result, preview_chars=220):
    print("Q:", result["query"], "\n")
    print("A:", result["answer"], "\n")
    print("Grounding:", result["grounding"])
    print("Retrieved passages:")
    for i, h in enumerate(result["sources"], 1):
        snippet = " ".join(h["text"].split())[:preview_chars]
        print(f"  [{i}] {h['chapter']} | p. {h['page_label']} | score {h['score']:.3f}\n      {snippet}...")


def launch_app(share=False):
    """Small web UI: type a question, get a cited answer plus the passages it was based on."""
    import gradio as gr

    def respond(question, k):
        if not question or not question.strip():
            return "Please type a question first.", ""
        res = ask(question.strip(), int(k))
        sources = "\n\n".join(
            f"**[{i}] {h['chapter']}, p. {h['page_label']}** (similarity {h['score']:.2f})\n\n> "
            + " ".join(h["text"].split())[:350] + "..."
            for i, h in enumerate(res["sources"], 1))
        g = res["grounding"]
        note = f"\n\n---\n*Cited passages: {g['cited_passages']} | answer words found in context: {g['context_word_overlap']:.0%}*"
        return res["answer"] + note, sources

    demo = gr.Interface(
        fn=respond,
        inputs=[gr.Textbox(label="Your question", lines=2, placeholder="e.g. Why is cross-validation better than a single split?"),
                gr.Slider(1, 10, value=TOP_K, step=1, label="Passages to retrieve (k)")],
        outputs=[gr.Markdown(label="Answer"), gr.Markdown(label="Retrieved passages")],
        title="ML Knowledge Assistant",
        description="Answers come only from 'Introduction to Machine Learning with Python' (Mueller & Guido).",
        examples=[["What is the difference between a random forest and gradient boosted trees?", 5],
                  ["Why should I use a Pipeline together with GridSearchCV?", 5],
                  ["How do I choose the number of clusters for k-means?", 5]])
    demo.launch(share=share)


retriever = None      # set by init_retriever()


def init_retriever():
    global retriever
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    embedder.max_seq_length = 512
    if os.path.exists(os.path.join(INDEX_DIR, "chunks.json")):
        retriever = Retriever.load(embedder)
        print(f"Loaded index with {len(retriever.chunks)} chunks from ./{INDEX_DIR}")
        return
    print("Building the index (first run only)...")
    tok = embedder.tokenizer
    count = lambda strs: [len(x) for x in tok(strs, add_special_tokens=False)["input_ids"]]
    chunks = chunk_lines(build_lines(load_pdf(PDF_PATH)), count)
    retriever = Retriever(embedder, chunks)
    retriever.save()
    print(f"Indexed {len(chunks)} chunks.")


def cli():
    print("Ask a question about the book (empty line to quit).")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        show(ask(q))


if __name__ == "__main__":
    import sys
    init_retriever()
    cli() if "--cli" in sys.argv else launch_app()
