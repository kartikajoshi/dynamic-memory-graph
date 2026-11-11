# Dynamic Memory Graph (DMG)

An external, persistent knowledge graph that gives small LLMs long-term conversational memory.

Small models like **Llama 3.2 3B** have a short context window (2048 tokens), so they forget what a user said a few turns ago. DMG fixes this without fine-tuning. It extracts facts from every user message as structured triples, stores them in a graph, and at query time injects only the relevant facts into the prompt instead of the full chat history.

> Course project, Artificial Intelligence, BML Munjal University.

---

## How it works

```
User message
    │
    ▼
1. Query Rewriter    →  resolves pronouns ("its side effects" → "Metformin side effects")
    │
    ▼
2. Extractor         →  pulls (subject, predicate, object, object_type) triples
    │
    ▼
3. Memory Manager    →  ADD / UPDATE / DELETE facts, links entity aliases, deduplicates edges
    │
    ▼
4. Graph Store       →  NetworkX graph, saved to data/graph.json
    │
    ▼
5. Retriever         →  BM25 + dense search + graph expansion, fused with RRF
    │
    ▼
6. LLM (Ollama)      →  answers using only the retrieved facts
```

**Entity types:** `PERSON`, `CONDITION`, `DRUG`, `PLACE`, `PREFERENCE`

**Example:**

| User says | Stored triple |
|---|---|
| "I'm diabetic and take Metformin" | `(User, has, Diabetes, CONDITION)` |
| | `(User, takes, Metformin, DRUG)` |

Twenty turns later, "What are its side effects?" is rewritten to "What are the side effects of Metformin?", and the stored facts are retrieved and passed to the model.

---

## Tech stack

| Purpose | Tool |
|---|---|
| Language | Python 3.10+ |
| LLM | Ollama, `llama3.2:3b` (runs locally) |
| Graph + persistence | NetworkX (JSON via `node_link_data`) |
| Keyword retrieval | rank-bm25 |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| UI | Streamlit |

---

## Project structure

```
dynamic-memory-graph/
├── README.md
├── requirements.txt
├── config.py                   # shared settings for all modules
├── problem_demo.py             # baseline vs DMG comparison demo
├── main.py                     # integration pipeline + CLI chat
├── app.py                      # Streamlit chat UI
│
├── modules/
│   ├── query_rewriter.py       # coreference resolution (Abhishek)
│   ├── extractor.py            # entity + triple extraction (Arnav)
│   ├── memory_manager.py       # ADD / UPDATE / DELETE, alias + dedup (Aditya)
│   ├── graph_store.py          # NetworkX backend, expand, rerank (Arpit)
│   ├── retriever.py            # BM25 + FAISS + graph → RRF fusion (Kartika)
│   └── graph_viz.py            # graph visualization helper (Arpit)
│
├── evaluation/
│   ├── eval.py                 # Recall@k + ablation study (Kartika)
│   └── test_conversations.json # test data for evaluation
│
├── data/                       # graph.json is created here at runtime
└── screenshots/                # graph visualizations and UI screenshots
```

---

## Setup

**1. Install Ollama and pull the model**

Download Ollama from https://ollama.com, then run:

```bash
ollama pull llama3.2:3b
```

**2. Clone the repo and install dependencies**

```bash
git clone https://github.com/kartikajoshi/dynamic-memory-graph.git
cd dynamic-memory-graph
python -m venv venv
```

Activate the virtual environment:

- Windows: `venv\Scripts\activate`
- macOS / Linux: `source venv/bin/activate`

Then install:

```bash
pip install -r requirements.txt
```

**3. Run**

Terminal chat:
```bash
python main.py
```

Streamlit UI:
```bash
streamlit run app.py
```

Problem demonstration (baseline vs DMG):
```bash
python problem_demo.py
```

Evaluation:
```bash
python -m evaluation.eval
python -m evaluation.eval --skip-llm   # quick check without Ollama
```

Graph visualization:
```bash
python -m modules.graph_viz
```

---

## Screenshots

> Screenshots will be added after the full pipeline is tested end-to-end.

| Screenshot | Description |
|---|---|
| `screenshots/graph.png` | Memory graph after a sample conversation |
| (Streamlit UI) | Chat interface with pipeline details sidebar |
| (Problem demo) | Side-by-side baseline vs DMG output |

---

## Team

| Member | Responsibility |
|---|---|
| Abhishek | Query rewriter (`modules/query_rewriter.py`) |
| Arnav | Entity and triple extractor (`modules/extractor.py`) |
| Aditya | Memory manager (`modules/memory_manager.py`) |
| Arpit | Graph store, reranker, visualization (`modules/graph_store.py`, `modules/graph_viz.py`) |
| Kartika | Retriever, evaluation, integration, UI, config, docs |

---

## Status

- [x] Project structure and dependencies
- [x] Query rewriter (coreference resolution + graceful fallback)
- [x] Extractor (few-shot extraction + retries + type fixing)
- [x] Memory manager (ADD / UPDATE / DELETE + aliases + edge dedup)
- [x] Graph store (NetworkX + multi-hop expansion + recency rerank)
- [x] Retriever (BM25 → RRF fusion with dense + graph)
- [x] Integration pipeline (main.py CLI)
- [x] Problem demonstration (baseline vs DMG)
- [x] Evaluation (Recall@k + ablation)
- [x] Streamlit UI
- [x] Graph visualization
- [ ] Final README with architecture diagram and screenshots
