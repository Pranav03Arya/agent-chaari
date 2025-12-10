# Agent Chaari: Indian Temple Travel Chatbot

Agent Chaari is a Streamlit-powered chatbot that helps travelers explore Indian temples.
It pairs a curated dataset with semantic search (Sentence Transformers + FAISS) and a
FLAN-T5 generator to answer questions, surface temple details, and sketch simple trip
ideas.

## What the project does
- Semantic retrieval over `final_temple_dataset_2.json` with state/deity filters.
- Intent-aware responses: temple info, discovery, simple trip plans, and budget tips.
- Distance lookups for popular cities when data is available in each record.
- Streamlit chat UI with optional map of matched temples.
- Reusable backend in `chatbot_backend.py` for programmatic access or other frontends.

## Why it is useful
- Gives quick, conversational access to temple overviews, legends, and visiting guides.
- Balances rules and embeddings to stay on-domain and reduce irrelevant answers.
- Works offline after initial model downloads; no external API keys required.
- Provides a baseline for extending to other cultural or travel datasets.

## Project layout
- `app.py` — Streamlit chat interface and embedding/index bootstrap.
- `chatbot_backend.py` — Intent detection, entity extraction, retrieval, and generation
  helpers.
- `final_temple_dataset_2.json` — Canonical v3.1 dataset used by the app (do not edit
  manually).
- `combined_temple_dataset_for_chatbot.json`, `final_temple_dataset.json` — Earlier data
  drops retained for reference.
- Notebooks (`executed.ipynb`, `chatbot_evaluation.ipynb`, `Untitled-1.ipynb`) — Data
  prep and evaluation explorations.
- Media (`CS678_Project.mp4`, `Streamlit ... .mp4`) — Demo/reference recordings.

## Getting started
### Prerequisites
- Python 3.10+ recommended.
- A virtual environment (see below).
- First run will download Hugging Face models (`all-MiniLM-L6-v2`, `google/flan-t5-small`);
  allow a few minutes and network access.

### Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install streamlit pandas sentence-transformers faiss-cpu transformers
```

If you prefer a pinned `requirements.txt`, add one alongside `app.py` and install with
`pip install -r requirements.txt`.

### Run the chat app
```powershell
streamlit run app.py
```

- The app loads `final_temple_dataset_2.json` from the repo root by default.
- Use sidebar filters to narrow by state or deity; toggle map display as needed.
- Ask about a temple, state, or deity (e.g., "tell me about Brihadeeswarar Temple",
  "Shiva temples in Tamil Nadu", "distance from Mumbai to Siddhivinayak").

### Programmatic usage
Reuse the backend without the UI:
```python
from pathlib import Path
import json
import chatbot_backend as backend
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

data = json.loads(Path("final_temple_dataset_2.json").read_text(encoding="utf-8"))
temples = []
for rec in data:
    text = backend.build_search_text(
        {
            "name": rec.get("name"),
            "state": rec.get("state"),
            "deities": rec.get("deities"),
            "sections": rec.get("sections"),
            "summary": rec.get("summary", ""),
        }
    )
    temples.append(
        {
            "name": rec.get("name"),
            "state": rec.get("state"),
            "deities": backend.to_list(rec.get("deities")),
            "text": text,
            "raw": rec,
        }
    )

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = encoder.encode(
    [t["text"] for t in temples],
    convert_to_numpy=True,
    normalize_embeddings=True,
)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
intent_embeddings = backend.prepare_intent_embeddings(encoder)

chatbot = backend.TempleChatbot(
    temples=temples,
    encoder=encoder,
    faiss_index=index,
    tokenizer=tokenizer,
    generator_model=generator,
    intent_ex_embeddings=intent_embeddings,
)

print(chatbot.answer("Tell me about Kedarnath Temple")["reply"])
```



## Maintainers and contributions
- Maintainer: Pranav Arya and Karthik Sanvelly.

