"""Lightweight entity tagger for forecast questions.

Tags each question with a primary domain and a list of named entities.
No external dependencies — pure keyword matching for domain, capitalised-
phrase extraction for entities. Can be upgraded to LLM-based extraction
later without changing the stored schema.

Schema stored per ForecastSnapshot:
    domain   : str   — one of DOMAINS (or "other")
    entities : list  — deduplicated named entities extracted from the question
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ── Domain taxonomy ──────────────────────────────────────────────────────────

DOMAINS = [
    "geopolitics",
    "elections",
    "economics",
    "crypto",
    "technology",
    "sports",
    "entertainment",
    "science",
    "society",
    "other",
]

# Each domain: list of lowercase keywords/phrases. Scored by total hits.
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "geopolitics": [
        "war", "peace", "nato", "russia", "ukraine", "china", "taiwan", "iran",
        "israel", "gaza", "conflict", "military", "sanction", "treaty", "ceasefire",
        "invasion", "troops", "nuclear", "missile", "diplomacy", "g7", "g20",
        "zelenskyy", "putin", "xi jinping", "geopolit", "foreign policy",
        "coup", "border", "territory", "annexation", "occupied", "ally", "alliance",
    ],
    "elections": [
        "election", "vote", "ballot", "primary", "president", "governor", "senator",
        "congress", "parliament", "poll", "candidate", "campaign", "democrat",
        "republican", "majority", "seat", "referendum", "inaugurati",
        "midterm", "runoff", "caucus", "electoral",
    ],
    "economics": [
        "fed", "interest rate", "inflation", "cpi", "gdp", "recession", "unemployment",
        "federal reserve", "rate cut", "rate hike", "treasury", "bond", "yield",
        "earnings", "revenue", "profit", "acquisition", "merger", "tariff", "trade",
        "export", "import", "dollar", "euro", "yen", "stock market", "s&p", "dow",
        "nasdaq", "ipo", "valuation", "fiscal", "monetary", "central bank",
    ],
    "crypto": [
        "bitcoin", "ethereum", "crypto", "blockchain", "btc", "eth", "nft",
        "defi", "microstrategy", "coinbase", "binance", "altcoin", "stablecoin",
        "solana", "dogecoin", "xrp", "web3", "token", "wallet", "halving",
        "mining", "proof of", "dao", "smart contract",
    ],
    "technology": [
        "artificial intelligence", "artificial general intelligence", "agi",
        "large language model", "llm", "openai", "anthropic", "deepmind",
        "gpt", "gemini", "claude", "apple", "microsoft", "meta", "nvidia",
        "semiconductor", "chip", "quantum", "autonomous", "self-driving",
        "robotics", "drone", "data center", "cloud", "cybersecurity", "hack", "breach",
    ],
    "sports": [
        "world cup", "nba", "nfl", "mlb", "nhl", "championship", "finals",
        "olympic", "soccer", "football", "basketball", "baseball", "tennis",
        "wimbledon", "super bowl", "playoff", "league", "tournament",
        "grand prix", "formula 1", "ufc", "boxing", "golf", "pga", "fifa",
        "premier league", "champions league", "retire", "season", "draft",
    ],
    "entertainment": [
        "movie", "film", "box office", "oscar", "grammy", "emmy", "award",
        "album", "tour", "concert", "celebrity", "married", "divorce", "tv show",
        "release", "netflix", "streaming", "prison break", "kardashian",
        "taylor swift", "beyoncé", "singer", "actor", "actress", "director",
        "sequel", "franchise", "trailer",
    ],
    "science": [
        "climate", "temperature", "sea level", "carbon", "emission", "nasa",
        "spacex", "rocket", "mars", "moon", "space station", "vaccine", "drug",
        "fda", "clinical trial", "pandemic", "virus", "earthquake", "hurricane",
        "wildfire", "species", "genome", "crispr", "particle",
        "population", "life expectancy", "mortality", "fertility", "birth rate",
        "energy", "renewable", "solar", "nuclear power", "fusion",
    ],
    "society": [
        "social media", "reddit", "twitter", "facebook", "instagram", "tiktok",
        "platform", "users", "monthly active", "app", "internet", "online",
        "poverty", "inequality", "housing", "homelessness", "crime",
        "education", "university", "school", "literacy",
        "healthcare", "hospital", "insurance", "opioid",
        "immigration", "refugee", "border crossing",
        "charity", "donation", "philanthropy", "givewell",
    ],
}

# Keywords meant to match token prefixes rather than whole tokens.
# Example: "geopolit*" should match "geopolitical", and "retire*" should
# match "retired" / "retirement".
_DOMAIN_PREFIX_KEYWORDS = {
    "geopolit",
    "inaugurati",
    "retire",
}

# ── Entity extraction ────────────────────────────────────────────────────────

# Known entities to always capture regardless of capitalisation.
_KNOWN_ENTITIES = [
    "United States", "US", "USA", "UK", "EU", "NATO", "UN", "WHO", "IMF",
    "Russia", "Ukraine", "China", "Taiwan", "Iran", "Israel", "Gaza",
    "North Korea", "South Korea", "India", "Pakistan", "Germany", "France",
    "Brazil", "Argentina", "Spain", "England", "Portugal", "Japan", "Saudi Arabia",
    "Federal Reserve", "Fed", "S&P 500", "Dow Jones", "Nasdaq",
    "Bitcoin", "Ethereum", "BTC", "ETH",
    "OpenAI", "Anthropic", "Google", "Apple", "Microsoft", "Meta", "Tesla",
    "Amazon", "Nvidia", "SpaceX", "NASA",
    "MicroStrategy", "Coinbase", "Binance",
    "FIFA", "NBA", "NFL", "MLB", "NHL", "Olympics",
    "Trump", "Biden", "Putin", "Zelenskyy", "Xi Jinping",
]

_KNOWN_LOWER = {e.lower(): e for e in _KNOWN_ENTITIES}

# Capitalised phrase pattern: 1-4 consecutive Title-Case words, excluding
# common sentence starters and question words.
_TITLE_PATTERN = re.compile(
    r'\b([A-Z][a-z]{1,20})(?:\s+[A-Z][a-z]{1,20}){0,3}\b'
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "will", "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "would", "could", "should",
    "may", "might", "shall", "can", "what", "when", "where", "who", "which",
    "how", "why", "if", "by", "before", "after", "during", "until", "since",
    "in", "on", "at", "to", "from", "of", "for", "with", "about", "without",
    "and", "or", "but", "not", "no", "yes", "any", "all", "new", "first",
    "next", "last", "this", "that", "than", "more", "most", "over", "under",
    # months and abbreviated months
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # common question/date words that produce false positives
    "between", "worldwide", "global", "season", "announce",
    "achieve", "release", "win", "lose", "reach", "meet", "sign", "pass",
    "become", "remain", "stay", "enter", "leave", "join", "exit",
}


def _extract_entities(text: str) -> List[str]:
    """Extract named entities from a question string."""
    # Strip leading question words that generate false title-case positives.
    text = re.sub(r'^\s*(?:Will|When|What|Who|How|Which|Is|Are|Does|Did|Can|Could|Would|Should)\s+', '', text)
    found: Dict[str, str] = {}  # lower → canonical

    # Known entities first (highest precision).
    # Use word-boundary regex so "UN" doesn't match inside "under", "until", etc.
    for key, canonical in _KNOWN_LOWER.items():
        pattern = r'(?<![a-z])' + re.escape(key) + r'(?![a-z])'
        if re.search(pattern, text, re.IGNORECASE):
            found[key] = canonical

    # Capitalised phrases not already covered.
    for m in _TITLE_PATTERN.finditer(text):
        phrase = m.group(0)
        phrase_lower = phrase.lower()
        # Skip stop-words and already-found knowns.
        words = phrase_lower.split()
        if all(w in _STOP_WORDS for w in words):
            continue
        if phrase_lower not in found:
            found[phrase_lower] = phrase

    # Deduplicate: remove phrases that are substrings of longer found phrases.
    canonical_list = list(found.values())
    result: List[str] = []
    for e in canonical_list:
        el = e.lower()
        if not any(el != o.lower() and el in o.lower() for o in canonical_list):
            result.append(e)

    return sorted(set(result))


def _score_domain(text: str) -> Tuple[str, Dict[str, int]]:
    """Return (best_domain, {domain: score}) for a question string."""
    lower = text.lower()
    tokens = _TOKEN_PATTERN.findall(lower)
    token_set = set(tokens)
    scores: Dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if _keyword_matches(lower, tokens, token_set, kw))
        if score:
            scores[domain] = score
    if not scores:
        return "other", scores
    best = max(scores, key=lambda d: scores[d])
    return best, scores


def _keyword_matches(lower: str, tokens: List[str], token_set: set[str], keyword: str) -> bool:
    """Token-aware keyword matching for domain scoring.

    Raw substring matching causes false positives for short symbols such as
    "eth" inside "netherlands". Match ordinary keywords as whole tokens or
    contiguous token phrases, with a small explicit allowlist for true stems.
    """
    kw = keyword.lower()
    if " " in kw:
        parts = _TOKEN_PATTERN.findall(kw)
        if not parts:
            return False
        width = len(parts)
        return any(tokens[i:i + width] == parts for i in range(len(tokens) - width + 1))
    if kw in _DOMAIN_PREFIX_KEYWORDS:
        return any(tok.startswith(kw) for tok in tokens)
    return kw in token_set


# ── Public API ───────────────────────────────────────────────────────────────

def tag_question(question: str, category: Optional[str] = None) -> Dict:
    """Return tagging dict for a forecast question.

    Returns:
        {
            "domain":   str,        # primary domain label
            "entities": list[str],  # extracted named entities
        }
    """
    if not question:
        return {"domain": "other", "entities": []}

    domain, _scores = _score_domain(question)

    # If the market's own category field is available and maps to a domain,
    # use it as a tiebreaker when the keyword score is low.
    if category and domain == "other":
        cat_lower = (category or "").lower()
        for d in DOMAINS:
            if d in cat_lower or cat_lower in d:
                domain = d
                break

    entities = _extract_entities(question)
    return {"domain": domain, "entities": entities}
