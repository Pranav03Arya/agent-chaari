from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


TEXT_FIELDS = ["overview", "story", "visiting_guide", "architecture", "scripture_mentions"]
# Mapping from user query phrases to distances_km keys in the dataset
CITY_KEY_MAP = {
    "new delhi": "from_new_delhi",
    "delhi": "from_new_delhi",
    "mumbai": "from_mumbai",
    "bombay": "from_mumbai",
    "chennai": "from_chennai",
    "madras": "from_chennai",
    "kolkata": "from_kolkata",
    "calcutta": "from_kolkata",
    "bangalore": "from_bangalore",
    "bengaluru": "from_bangalore",
}

# Intent examples kept small and deterministic for embedding fallback
INTENT_EXAMPLES = {
    "TEMPLE_INFO": [
        "Tell me about Kedarnath temple",
        "Give details on this temple",
        "History of the temple",
        "Temple overview",
    ],
    "FIND_TEMPLES": [
        "temples in Tamil Nadu",
        "find Shiva temples near river",
        "list famous Vishnu shrines",
        "recommend temples to visit",
    ],
    "PLAN_TRIP": [
        "plan a 3 day trip",
        "help me plan a pilgrimage",
        "itinerary for temples",
        "trip plan with timings",
    ],
    "ITINERARY_COST": [
        "budget for a 2 day temple trip",
        "what is the cost for visiting",
        "estimate expenses for pilgrimage",
        "trip cost for temples",
    ],
    "SMALL_TALK": [
        "hello",
        "how are you",
        "thanks",
        "good morning",
    ],
    "UNKNOWN": [
        "random question",
        "not sure",
    ],
}

RULE_KEYWORDS = {
    "ITINERARY_COST": ["cost", "budget", "price", "expense"],
    "PLAN_TRIP": ["plan", "itinerary", "trip", "travel"],
    "TEMPLE_INFO": ["about", "details", "history", "information", "info"],
    "FIND_TEMPLES": ["find", "show", "list", "recommend", "suggest", "near"],
    "SMALL_TALK": ["hello", "hi", "thanks", "thank you", "good morning", "good evening"],
}


def to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        return parts or [value.strip()]
    return [str(value).strip()]


def build_search_text(item: Dict[str, Any]) -> str:
    """Concatenate salient fields for semantic search (v3.1 schema)."""
    parts: List[str] = []

    # Basic identifiers
    name = (item.get("name") or item.get("temple_name") or "").strip()
    state = (item.get("state") or item.get("region") or "").strip()
    deities = to_list(item.get("deities") or item.get("deity"))
    base_values = [name, state, ", ".join(deities)]
    for val in base_values:
        if val:
            parts.append(val)

    # v3.1 sections
    sections = item.get("sections") or {}
    if not sections:
        # Fallback for legacy callers that still pass flat fields
        sections = {field: item.get(field, "") for field in TEXT_FIELDS}

    # Pull individual section texts
    overview = sections.get("overview", "")
    story = sections.get("story", "")
    visiting = sections.get("visiting_guide", "")
    arch = sections.get("architecture", "")
    scripture = sections.get("scripture_mentions", "")

    # v3.1 summary is short and high-quality; include it explicitly
    summary = item.get("summary", "")

    for val in [summary, overview, story, visiting, arch, scripture]:
        if not val:
            continue
        if isinstance(val, list):
            sval = " ".join(str(v).strip() for v in val if str(v).strip())
        else:
            sval = str(val).strip()
        if sval:
            parts.append(sval)

    return " | ".join(parts).strip()


def extract_days_from_text(msg_lower: str) -> Optional[int]:
    match = re.search(r"(\\d+)\\s*day(?:s)?", msg_lower)
    return int(match.group(1)) if match else None


def generate_answer(
    context_text: str,
    user_message: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    max_new_tokens: int = 180,
) -> str:
    # Model max length is 512 tokens; reserve ~150 tokens for prompt template and user message
    max_context_tokens = 350
    
    # Truncate context if needed
    context_tokens = tokenizer.encode(context_text, add_special_tokens=False)
    if len(context_tokens) > max_context_tokens:
        # Truncate to fit within limit
        truncated_tokens = context_tokens[:max_context_tokens]
        context_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        # Add ellipsis to indicate truncation
        context_text = context_text.rstrip() + "..."
    
    prompt = textwrap.dedent(
        f"""
        You are a helpful Indian temple travel assistant.
        Use ONLY the following context to answer:
        {context_text}
        Question: {user_message}
        Answer in 4-6 friendly sentences.
        """
    ).strip()
    try:
        # Use truncation=True as safety net (max_length=512 is default for flan-t5-small)
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception:
        if context_text.strip():
            return "Here's what I can share from what I know: " + context_text[:400]
        return "I don't have enough context yet, but I can help if you mention a temple, state, or deity."


def _rule_based_intent(text_lower: str) -> Optional[str]:
    # Special pattern: "name ... temples in ..."
    if "temples in" in text_lower and "name" in text_lower:
        return "FIND_TEMPLES"
    for intent, keywords in RULE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return None


def prepare_intent_embeddings(
    encoder: SentenceTransformer, intent_examples: Dict[str, List[str]] = INTENT_EXAMPLES
) -> Dict[str, Any]:
    return {
        intent: encoder.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)
        for intent, phrases in intent_examples.items()
    }


def detect_intent(
    user_message: str,
    encoder: SentenceTransformer,
    intent_ex_embeddings: Dict[str, Any],
) -> str:
    if not user_message or not user_message.strip():
        return "UNKNOWN"
    text = user_message.strip()
    text_lower = text.lower()
    rule_intent = _rule_based_intent(text_lower)
    if rule_intent:
        return rule_intent
    q_emb = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    best_intent, best_score = "UNKNOWN", -1.0
    for intent, emb in intent_ex_embeddings.items():
        scores = emb @ q_emb  # cosine similarity because normalized
        top = float(scores.max())
        if top > best_score:
            best_intent, best_score = intent, top
    return best_intent if best_score >= 0.4 else "UNKNOWN"


def extract_entities(
    user_message: str,
    temple_names: Iterable[str],
    state_names: Iterable[str],
    deity_names: Iterable[str],
    max_hits: int = 5,
) -> Dict[str, Any]:
    msg_lower = user_message.lower() if user_message else ""

    def _find(candidates: Iterable[str]) -> List[str]:
        hits = [c for c in candidates if c and c.lower() in msg_lower]
        return hits[:max_hits]

    temples = _find(temple_names)
    states = _find(state_names)
    deities = _find(deity_names)

    return {
        "temples": temples,
        "states": states,
        "deities": deities,
        "days": extract_days_from_text(msg_lower),
        "attributes": {
            "overview": ("overview" in msg_lower or "summary" in msg_lower),
            "story": ("story" in msg_lower or "legend" in msg_lower),
            "visiting_guide": (
                "visit" in msg_lower or "visiting" in msg_lower or "how to reach" in msg_lower or "guide" in msg_lower
            ),
            "architecture": "architecture" in msg_lower,
            "scripture_mentions": ("scripture" in msg_lower or "mentioned in" in msg_lower),
        },
    }


def search_temples(
    query: str,
    encoder: SentenceTransformer,
    index,
    metadata: List[Dict[str, Any]],
    k: int = 5,
    state_filter: Optional[str] = None,
    deity_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        return []
    query_vec = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_vec, k=k * 5)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        rec = metadata[idx]
        if state_filter and state_filter.lower() not in rec.get("state", "").lower():
            continue
        if deity_filter:
            deities = rec.get("deities") or []
            if not any(deity_filter.lower() in str(d).lower() for d in deities):
                continue
        raw = rec.get("raw", rec) or {}
        sections = raw.get("sections") or rec.get("sections")
        if sections is None:
            # Fallback for any legacy structure without a sections dict
            sections = {field: raw.get(field, "") for field in TEXT_FIELDS}

        results.append(
            {
                "name": rec.get("name", ""),
                "state": rec.get("state", ""),
                "score": float(dist),
                "raw": raw,
                "sections": sections,
            }
        )
        if len(results) >= k:
            break
    return results


class TempleChatbot:
    def __init__(
        self,
        temples: List[Dict[str, Any]],
        encoder: SentenceTransformer,
        faiss_index,
        tokenizer: AutoTokenizer,
        generator_model: AutoModelForSeq2SeqLM,
        intent_ex_embeddings: Optional[Dict[str, Any]] = None,
    ):
        self.temples = temples
        self.encoder = encoder
        self.index = faiss_index
        self.tokenizer = tokenizer
        self.generator_model = generator_model
        self.intent_ex_embeddings = intent_ex_embeddings or prepare_intent_embeddings(encoder)
        self.temple_names = sorted({t.get("name") for t in temples if t.get("name")})
        self.state_names = sorted({t.get("state") for t in temples if t.get("state")})
        self.deity_names = sorted({d for t in temples for d in (t.get("deities") or []) if d})

    # NEW: helper to map a city name in the message to a distances_km key
    def _extract_city_from_message(self, user_message: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Look for a known city phrase in the user message and map it to the
        appropriate distances_km key (from_new_delhi, from_mumbai, etc).

        Returns (distance_key, pretty_city_label) or (None, None) if nothing matches.
        Matches longer phrases first to avoid partial matches.
        """
        msg_lower = (user_message or "").lower()
        # Sort by phrase length (longest first) to match "new delhi" before "delhi"
        sorted_phrases = sorted(CITY_KEY_MAP.items(), key=lambda x: -len(x[0]))
        for phrase, key in sorted_phrases:
            if phrase in msg_lower:
                pretty = phrase.title()
                # normalise common variants
                if phrase == "delhi":
                    pretty = "New Delhi"
                elif phrase == "bombay":
                    pretty = "Mumbai"
                elif phrase == "madras":
                    pretty = "Chennai"
                elif phrase == "bengaluru":
                    pretty = "Bangalore"
                return key, pretty
        return None, None

    def detect_intent(self, user_message: str) -> str:
        return detect_intent(user_message, self.encoder, self.intent_ex_embeddings)

    def extract_entities(self, msg: str) -> Dict[str, Any]:
        msg_l = msg.lower()

        temple_names = sorted([t["name"] for t in self.temples if t.get("name")], key=lambda x: -len(x))
        found_temples = [n for n in temple_names if n.lower() in msg_l]

        states_found = [s for s in self.state_names if s.lower() in msg_l]
        deities_found = [d for d in self.deity_names if d.lower() in msg_l]

        wants_overview = "overview" in msg_l or "summary" in msg_l
        wants_story = "story" in msg_l or "legend" in msg_l
        wants_guide = ("visit" in msg_l or "visiting" in msg_l or "how to reach" in msg_l or "guide" in msg_l)
        wants_arch = "architecture" in msg_l
        wants_scripture = "scripture" in msg_l or "mentioned in" in msg_l

        return {
            "temples": found_temples,
            "states": states_found,
            "deities": deities_found,
            "days": extract_days_from_text(msg_l),
            "attributes": {
                "overview": wants_overview,
                "story": wants_story,
                "visiting_guide": wants_guide,
                "architecture": wants_arch,
                "scripture_mentions": wants_scripture,
            },
        }

    def retrieve(
        self,
        query: str,
        k: int = 5,
        state_filter: Optional[str] = None,
        deity_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return search_temples(
            query=query,
            encoder=self.encoder,
            index=self.index,
            metadata=self.temples,
            k=k,
            state_filter=state_filter,
            deity_filter=deity_filter,
        )

    def build_context(self, temple_records: List[Dict[str, Any]]) -> str:
        """
        Build detailed context for generation from v3.1 schema:
        summary + sections + optional distances.
        """
        ctx_parts: List[str] = []

        for rec in temple_records:
            raw = rec.get("raw") or {}
            sections = raw.get("sections") or {}

            if not sections:
                sections = {field: raw.get(field, "") for field in TEXT_FIELDS}

            parts: List[str] = []

            # Add summary first if available
            summary = raw.get("summary")
            if isinstance(summary, str) and summary.strip():
                parts.append(f"Summary:\n{summary.strip()}")

            # Emphasize overview + visiting guide first, then story, etc.
            for key in ("overview", "visiting_guide", "story", "architecture", "scripture_mentions"):
                val = sections.get(key)
                if not val:
                    continue
                sval = str(val).strip()
                if not sval:
                    continue
                label = key.replace("_", " ").title()
                parts.append(f"{label}:\n{sval}")

            # Append approximate distances, if present
            distances = raw.get("distances_km") or {}
            if distances:
                snippets = []
                for city, km in distances.items():
                    try:
                        km_val = float(km)
                        snippets.append(f"{city}: ~{int(km_val)} km")
                    except Exception:
                        continue
                if snippets:
                    parts.append("Approximate distances:\n" + ", ".join(snippets))

            info_block = "\n\n".join(parts).strip()
            name = rec.get("name", "") or raw.get("name", "")
            state = rec.get("state", "") or raw.get("state", "")
            deities = raw.get("deities") or rec.get("deities") or []
            deities_str = ", ".join(str(d) for d in deities if str(d).strip())

            ctx_parts.append(
                f"Name: {name}\nState: {state}\nDeities: {deities_str}\n\n{info_block}"
            )

        return "\n\n---\n\n".join([c for c in ctx_parts if c.strip()])

    def generate(self, context_records: List[Dict[str, Any]], user_message: str) -> str:
        context_text = self.build_context(context_records) if context_records else ""
        return generate_answer(
            context_text or "No specific context available.",
            user_message,
            tokenizer=self.tokenizer,
            model=self.generator_model,
        )

    def answer(self, user_message: str) -> Dict[str, Any]:
        intent = self.detect_intent(user_message)
        entities = self.extract_entities(user_message)
        used_temples: List[str] = []
        sections: Dict[str, Any] = {}
        msg_l = user_message.lower()
        is_distance_question = (
            "distance" in msg_l 
            or "how far" in msg_l 
            or "km from" in msg_l 
            or "kilometers from" in msg_l
        )
        attr_requested = any(entities.get("attributes", {}).values()) if entities.get("attributes") else False

        # Domain guard:
        # - If user asks for attributes (overview/story/guide), we ALWAYS stay in TEMPLE_INFO domain.
        # - For other queries with no temple/state/deity, we treat as UNKNOWN (unless SMALL_TALK).
        if not (entities["temples"] or entities["states"] or entities["deities"]):
            if intent != "SMALL_TALK" and not attr_requested:
                intent = "UNKNOWN"

        # Any explicit attribute request OR distance question should go through TEMPLE_INFO logic
        if attr_requested or is_distance_question:
            intent = "TEMPLE_INFO"

        state_filter = entities.get("states", [None])[0] if entities.get("states") else None
        deity_filter = entities.get("deities", [None])[0] if entities.get("deities") else None

        if intent == "TEMPLE_INFO":
            query_text = entities["temples"][0] if entities.get("temples") else user_message
            hits = self.retrieve(query_text, k=5, state_filter=state_filter, deity_filter=deity_filter)
            if not hits and entities.get("temples"):
                hits = self.retrieve(user_message, k=5, state_filter=state_filter, deity_filter=deity_filter)
            if not hits:
                reply = (
                    "I couldn't find matching temples for that description. "
                    "Try rephrasing with the temple name, state, or deity."
                )
                return {
                    "reply": reply,
                    "intent": intent,
                    "entities": entities,
                    "used_temples": used_temples,
                    "sections": sections,
                }
            else:
                primary = hits[0]
                raw = primary.get("raw", {})
                sections = primary.get("sections") or raw.get("sections", {}) or {}

                # 1) DISTANCE QUESTIONS: "how far", "distance from X" (prioritize this)
                if is_distance_question:
                    distances = raw.get("distances_km") or {}
                    if distances:
                        # Use the helper to find which city the user asked about
                        dist_key, city_label = self._extract_city_from_message(user_message)

                        if dist_key and dist_key in distances:
                            try:
                                km_val = float(distances[dist_key])
                                reply = (
                                    f"{primary.get('name', 'This temple')} is approximately "
                                    f"{int(km_val)} km from {city_label}. "
                                    "This is an approximate distance; actual travel distance may vary."
                                )
                            except Exception:
                                reply = (
                                    f"I have distance information for {primary.get('name', 'this temple')}, "
                                    "but couldn't parse the exact number cleanly."
                                )
                        else:
                            # Fallback: list all known distances for this temple
                            snippets = []
                            for key, km in distances.items():
                                try:
                                    km_val = float(km)
                                    # keys look like 'from_new_delhi' -> 'New Delhi'
                                    city = key.replace("from_", "").replace("_", " ").title()
                                    snippets.append(f"{city}: ~{int(km_val)} km")
                                except Exception:
                                    continue
                            if snippets:
                                reply = (
                                    f"I couldn't match the exact city you asked for, "
                                    f"but here are approximate distances for {primary.get('name', 'this temple')}:\n"
                                    + ", ".join(snippets)
                                )
                            else:
                                reply = (
                                    "I couldn't extract a clean distance value, "
                                    "but I can still help with an overview or visiting guide."
                                )
                    else:
                        reply = (
                            "I don't have distance data for this temple, "
                            "but I can share its overview or visiting details if you'd like."
                        )

                # 2) ATTRIBUTE QUERIES: overview / summary / story / visiting guide etc.
                # If the user explicitly asked for these, answer directly from sections.
                elif attr_requested and sections:
                    attrs = entities.get("attributes", {})
                    chosen_keys = [k for k, v in attrs.items() if v] or ["overview"]

                    text_blocks: List[str] = []
                    for key in chosen_keys:
                        val = sections.get(key)
                        if not val:
                            continue
                        label = key.replace("_", " ").title()
                        text_blocks.append(f"{label}:\n{str(val).strip()}")

                    if text_blocks:
                        reply = "\n\n".join(text_blocks)
                    else:
                        # Fallback to overview or summary if specific key missing
                        fallback = sections.get("overview") or raw.get("summary")
                        if fallback:
                            reply = str(fallback).strip()
                        else:
                            reply = (
                                f"I know about {primary.get('name', 'this temple')}, "
                                "but I don't have a detailed overview or story stored yet."
                            )

                # 3) GENERAL TEMPLE INFO: default FLAN-T5 behaviour
                else:
                    # Limit to top 2 records for context
                    ctx = self.build_context(hits[:2])
                    if ctx:
                        reply = generate_answer(
                            ctx,
                            user_message,
                            tokenizer=self.tokenizer,
                            model=self.generator_model,
                        )
                    else:
                        reply = (
                            f"I know about {primary.get('name', 'this temple')} "
                            f"in {primary.get('state', '')}, "
                            "but I don't have detailed sections like overview or story yet."
                        )

                used_temples = [h.get("name", "") for h in hits]

                return {
                    "reply": reply,
                    "intent": intent,
                    "entities": entities,
                    "used_temples": used_temples,
                    "sections": sections,
                }

        elif intent == "FIND_TEMPLES":
            hits = self.retrieve(user_message, k=5, state_filter=state_filter, deity_filter=deity_filter)
            if not hits and (state_filter or deity_filter):
                hits = self.retrieve("popular temples", k=5)
            if hits:
                lines_out = []
                for h in hits:
                    raw = h.get("raw") or {}
                    deities = raw.get("deities") or []
                    line = f"- {h['name']} ({h['state']})"
                    if deities:
                        line += f" | Deities: {', '.join(deities)}"
                    lines_out.append(line)
                reply = "Here are some temples you might like:\n" + "\n".join(lines_out)
            else:
                reply = "I couldn't find exact matches, but here are some similar temples"
            used_temples = [h.get("name", "") for h in hits] if hits else []
            return {
                "reply": reply,
                "intent": intent,
                "entities": entities,
                "used_temples": used_temples,
                "sections": {},
            }

        elif intent == "PLAN_TRIP":
            days = entities.get("days") or 2
            hits = self.retrieve(user_message, k=max(days, 5), state_filter=state_filter, deity_filter=deity_filter)
            if not hits and state_filter:
                hits = self.retrieve(state_filter, k=max(days, 5))
            if not hits:
                alt_hits = self.retrieve("popular pilgrimage temples", k=max(days, 3))
                if alt_hits:
                    hits = alt_hits
                else:
                    reply = "I couldn't find exact matches, but here are some similar temples"
                    return {"reply": reply, "intent": intent, "entities": entities, "used_temples": [], "sections": {}}
            plan_lines = []
            for i in range(days):
                rec = hits[i % len(hits)] if hits else None
                if rec:
                    raw = rec.get("raw") or {}
                    loc = raw.get("location") or raw.get("city") or rec.get("state", "")
                    plan_lines.append(
                        f"Day {i+1}: Visit {rec.get('name', 'a temple')} in {rec.get('state', '')} around {loc or 'the area'}"
                    )
                else:
                    plan_lines.append(f"Day {i+1}: Explore nearby temples or local sites.")
            reply = "Here's a simple plan:\n" + "\n".join(plan_lines)
            used_temples = [h.get("name", "") for h in hits] if hits else []
            return {
                "reply": reply,
                "intent": intent,
                "entities": entities,
                "used_temples": used_temples,
                "sections": {},
            }

        elif intent == "ITINERARY_COST":
            reply = (
                "A modest trip often costs around INR 2k-4k per day for stay, food, and local travel; "
                "adjust upward for private transport or premium lodging. Prices vary by season and city."
            )
            used_temples = []
            return {
                "reply": reply,
                "intent": intent,
                "entities": entities,
                "used_temples": used_temples,
                "sections": {},
            }

        elif intent == "SMALL_TALK":
            reply = (
                "Hi there! Im Agent Chaari, an Indian temple travel assistant. "
                "I can help you discover temples, plan simple trips, or share temple details. "
                "What would you like to explore?"
            )
            used_temples = []
            return {
                "reply": reply,
                "intent": intent,
                "entities": entities,
                "used_temples": used_temples,
                "sections": {},
            }

        else:
            reply = (
                "Im mainly designed to help with Indian temples and temple trips. "
                "Try asking about a temple, a state, or a deity."
            )
            used_temples = []
            return {
                "reply": reply,
                "intent": intent,
                "entities": entities,
                "used_temples": used_temples,
                "sections": {},
            }
