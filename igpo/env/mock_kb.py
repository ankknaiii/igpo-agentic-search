"""Tiny knowledge base for Colab / offline agentic search demos.

Each document has a title + text. Search ranks by simple keyword overlap.
This avoids Serper/Bing API dependency while preserving multi-hop structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    aliases: tuple[str, ...] = ()


# Multi-hop friendly toy corpus inspired by HotpotQA-style composition.
DOCUMENTS: list[Document] = [
    Document(
        title="College Lovers",
        text=(
            "College Lovers is a 1930 American comedy film directed by John G. Adolfi. "
            "It was released on October 5, 1930."
        ),
        aliases=("college lovers film",),
    ),
    Document(
        title="The Dixie Flyer",
        text=(
            "The Dixie Flyer is a 1926 American silent action film directed by Charles J. Hunt. "
            "It was released in 1926."
        ),
        aliases=("dixie flyer film",),
    ),
    Document(
        title="John G. Adolfi",
        text=(
            "John Griffith Adolfi (February 19, 1888 – May 11, 1933) was an American actor "
            "and film director who directed College Lovers."
        ),
        aliases=("john adolfi", "adolfi"),
    ),
    Document(
        title="Charles J. Hunt",
        text=(
            "Charles J. Hunt (April 8, 1881 – November 3, 1954) was an American film director "
            "who directed The Dixie Flyer."
        ),
        aliases=("charles hunt",),
    ),
    Document(
        title="Augusta Marie of Holstein-Gottorp",
        text=(
            "Augusta Marie of Holstein-Gottorp (6 February 1649 – 25 April 1728) was a German "
            "noblewoman. She was a daughter of Frederick III, Duke of Holstein-Gottorp and "
            "Duchess Marie Elisabeth of Saxony."
        ),
        aliases=("augusta marie",),
    ),
    Document(
        title="Duchess Marie Elisabeth of Saxony",
        text=(
            "Duchess Marie Elisabeth of Saxony (22 November 1610 – 24 October 1684) was a "
            "duchess consort of Holstein-Gottorp. She was the mother of Augusta Marie of "
            "Holstein-Gottorp."
        ),
        aliases=("marie elisabeth of saxony", "marie elisabeth"),
    ),
    Document(
        title="Paris",
        text=(
            "Paris is the capital and largest city of France. It is located in northern France "
            "on the River Seine."
        ),
        aliases=("city of paris",),
    ),
    Document(
        title="France",
        text=(
            "France is a country in Western Europe. Its capital is Paris. The official language "
            "is French and the currency is the euro."
        ),
    ),
    Document(
        title="Eiffel Tower",
        text=(
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was designed "
            "by Gustave Eiffel's company and completed in 1889."
        ),
        aliases=("tour of eiffel",),
    ),
    Document(
        title="Gustave Eiffel",
        text=(
            "Alexandre Gustave Eiffel (15 December 1832 – 27 December 1923) was a French civil "
            "engineer. His company designed the Eiffel Tower."
        ),
        aliases=("eiffel",),
    ),
    Document(
        title="Yangtze",
        text=(
            "The Yangtze is the longest river in Asia and the third-longest in the world. "
            "It flows entirely within China."
        ),
        aliases=("chang jiang", "yangtze river"),
    ),
    Document(
        title="China",
        text=(
            "China is a country in East Asia. Its capital is Beijing. The Yangtze is the "
            "longest river in China."
        ),
        aliases=("prc",),
    ),
    Document(
        title="Beijing",
        text="Beijing is the capital of China. It is a major political and cultural center.",
        aliases=("peking",),
    ),
    Document(
        title="Alan Turing",
        text=(
            "Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician "
            "and computer scientist. He is widely considered the father of theoretical "
            "computer science and artificial intelligence."
        ),
        aliases=("turing",),
    ),
    Document(
        title="University of Cambridge",
        text=(
            "The University of Cambridge is a collegiate research university in Cambridge, "
            "England. Alan Turing studied there."
        ),
        aliases=("cambridge university", "cambridge"),
    ),
]


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def search_kb(query: str, *, top_k: int = 3) -> list[dict]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scored: list[tuple[float, Document]] = []
    for doc in DOCUMENTS:
        hay = _tokenize(doc.title + " " + doc.text + " " + " ".join(doc.aliases))
        overlap = len(q_tokens & hay)
        # Mild title boost for more realistic retrieval ranking.
        title_hit = 1.5 if any(t in _tokenize(doc.title) for t in q_tokens) else 0.0
        score = overlap + title_hit
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: (-x[0], x[1].title))
    results = []
    for score, doc in scored[:top_k]:
        results.append(
            {
                "title": doc.title,
                "snippet": doc.text,
                "score": float(score),
            }
        )
    return results


def format_search_results(results: list[dict]) -> str:
    if not results:
        return "No results found."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\n{r['snippet']}")
    return "\n\n".join(parts)
