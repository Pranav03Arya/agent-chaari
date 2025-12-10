from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - handled in UI
    faiss = None

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import chatbot_backend as backend


DATA_PATH = Path("final_temple_dataset_2.json")  # v3.1 dataset
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"


def load_raw_data(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array of temple records.")
    return data


def normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    # Core identifiers
    name = rec.get("name") or rec.get("temple_name") or ""
    state = rec.get("state") or rec.get("region") or ""
    location = rec.get("location") or rec.get("city") or rec.get("place") or ""

    # Clean deities (v3.1 usually already has good lists, but keep to_list for safety)
    deities = backend.to_list(rec.get("deities") or rec.get("deity"))

    # Sections: in v3.1 they are typically under rec["sections"]
    sections = rec.get("sections") or {}
    if not sections:
        # Fallback for any legacy rows that still keep fields flat
        sections = {field: rec.get(field, "") for field in backend.TEXT_FIELDS}

    # Coordinates are nested under "coordinates" when available
    coords = rec.get("coordinates") or {}
    lat = coords.get("lat") or rec.get("lat") or rec.get("latitude")
    lng = coords.get("lng") or rec.get("lng") or rec.get("longitude")

    # Build the semantic text for embeddings using backend helper
    search_text = backend.build_search_text(
        {
            "name": name,
            "state": state,
            "deities": deities,
            "sections": sections,
            "summary": rec.get("summary", ""),
        }
    )
    return {
        "id": rec.get("id") or rec.get("temple_id") or rec.get("slug") or name,
        "name": name,
        "state": state,
        "location": location,
        "deities": deities,
        "lat": lat,
        "lng": lng,
        "text": search_text,
        # keep full v3.1 record in raw so backend can access distances_km, sections, etc.
        "raw": rec,
    }


@st.cache_resource(show_spinner=False)
def load_chatbot_resources() -> Tuple[pd.DataFrame, backend.TempleChatbot]:
    if faiss is None:
        raise ImportError("FAISS is not installed; cannot build the semantic index.")

    raw_data = load_raw_data(DATA_PATH)
    normalized = [normalize_record(r) for r in raw_data]
    df = pd.DataFrame(normalized)
    df["search_text"] = df["text"].fillna("")

    encoder = SentenceTransformer(EMBED_MODEL_NAME)
    corpus_texts = df["search_text"].fillna("").tolist()
    embeddings = encoder.encode(
        corpus_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    intent_emb = backend.prepare_intent_embeddings(encoder)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    chatbot = backend.TempleChatbot(
        temples=df.to_dict(orient="records"),
        encoder=encoder,
        faiss_index=index,
        tokenizer=tokenizer,
        generator_model=gen_model,
        intent_ex_embeddings=intent_emb,
    )
    return df, chatbot


def render_sidebar(df: pd.DataFrame) -> Dict[str, Any]:
    st.sidebar.header("Filters")
    states = ["All"] + sorted({s for s in df["state"].dropna().tolist() if s})
    selected_state = st.sidebar.selectbox("State", options=states, index=0)
    unique_deities = sorted({d for lst in df["deities"].tolist() for d in (lst or []) if d})
    selected_deities = st.sidebar.multiselect("Deity", options=unique_deities, default=[])
    show_map = st.sidebar.checkbox("Show map", value=True)
    return {
        "state": None if selected_state == "All" else selected_state,
        "deities": selected_deities,
        "show_map": show_map,
    }


def apply_filters_to_query(user_message: str, state: Optional[str], deities: List[str]) -> str:
    enriched = user_message
    if state:
        enriched += f" in {state}"
    if deities:
        enriched += " about " + ", ".join(deities)
    return enriched


def main():
    st.title("Agent Chaari — Indian Temple Travel Chatbot")

    try:
        df, chatbot = load_chatbot_resources()
    except Exception as exc:  # pragma: no cover - UI surfacing
        st.error(f"Chatbot initialization failed: {exc}")
        st.stop()

    filters = render_sidebar(df)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi, Im Agent Chaari, an Indian temple travel chatbot. "
                    "Ask me about temples, states, deities, or simple temple trips."
                ),
            }
        ]

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            sections = msg.get("sections") or {}
            if sections:
                st.markdown("### Temple Details")

                overview = sections.get("overview")
                story = sections.get("story")
                visiting = sections.get("visiting_guide")
                architecture = sections.get("architecture")
                scripture = sections.get("scripture_mentions")

                if overview:
                    st.markdown("**Overview**")
                    st.write(overview)

                if story:
                    st.markdown("**Story**")
                    st.write(story)

                if visiting:
                    st.markdown("**Visiting Guide**")
                    st.write(visiting)

                if architecture:
                    st.markdown("**Architecture**")
                    st.write(architecture)

                if scripture:
                    st.markdown("**Scripture Mentions**")
                    st.write(scripture)

    user_prompt = st.chat_input("Ask about temples, trips, or deities...")
    if user_prompt:
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        st.chat_message("user").write(user_prompt)

        query_text = apply_filters_to_query(user_prompt, filters["state"], filters["deities"])
        result = chatbot.answer(query_text)

        reply_text = result.get("reply") or (
            "I don't have enough context yet, but I can help if you mention a temple, state, or deity."
        )
        sections = result.get("sections") or {}

        assistant_msg = {
            "role": "assistant",
            "content": reply_text,
        }
        if sections:
            assistant_msg["sections"] = sections

        st.session_state["messages"].append(assistant_msg)

        # Render the latest assistant message (same structure as history loop)
        with st.chat_message("assistant"):
            st.write(reply_text)

            if sections:
                st.markdown("### Temple Details")

                overview = sections.get("overview")
                story = sections.get("story")
                visiting = sections.get("visiting_guide")
                architecture = sections.get("architecture")
                scripture = sections.get("scripture_mentions")

                if overview:
                    st.markdown("**Overview**")
                    st.write(overview)

                if story:
                    st.markdown("**Story**")
                    st.write(story)

                if visiting:
                    st.markdown("**Visiting Guide**")
                    st.write(visiting)

                if architecture:
                    st.markdown("**Architecture**")
                    st.write(architecture)

                if scripture:
                    st.markdown("**Scripture Mentions**")
                    st.write(scripture)

        if filters["show_map"]:
            hits = chatbot.retrieve(
                query_text,
                k=5,
                state_filter=filters["state"],
                deity_filter=filters["deities"][0] if filters["deities"] else None,
            )
            map_rows = []
            for h in hits:
                raw = h.get("raw") or {}
                lat = raw.get("lat") or raw.get("latitude") or h.get("lat")
                lng = raw.get("lng") or raw.get("longitude") or h.get("lng")
                if lat is None or lng is None:
                    continue
                map_rows.append({"lat": float(lat), "lon": float(lng), "name": h.get("name", "")})
            if map_rows:
                st.map(pd.DataFrame(map_rows))

if __name__ == "__main__":
    main()
