"""Offline retrieval corpus for agentic search experiments.

The corpus supports multi-hop question answering without external API calls.
Optional retrieval noise may be injected to approximate imperfect ranking.

References:
    Wang et al., IGPO, ICLR 2026.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Single document in the offline retrieval corpus."""

    title: str
    text: str
    aliases: tuple[str, ...] = ()


DOCUMENTS: list[Document] = [
    Document(title='College Lovers', text='College Lovers is a 1930 American comedy film directed by John G. Adolfi. It was released on October 5, 1930.', aliases=('college lovers film',)),
    Document(title='The Dixie Flyer', text='The Dixie Flyer is a 1926 American silent action film directed by Charles J. Hunt. It was released in 1926.', aliases=('dixie flyer film',)),
    Document(title='John G. Adolfi', text='John Griffith Adolfi (February 19, 1888 – May 11, 1933) was an American film director who directed College Lovers.', aliases=('john adolfi', 'adolfi',)),
    Document(title='Charles J. Hunt', text='Charles J. Hunt (April 8, 1881 – November 3, 1954) was an American film director who directed The Dixie Flyer.', aliases=('charles hunt',)),
    Document(title='Augusta Marie of Holstein-Gottorp', text='Augusta Marie of Holstein-Gottorp (6 February 1649 – 25 April 1728) was a German noblewoman, daughter of Frederick III, Duke of Holstein-Gottorp and Duchess Marie Elisabeth of Saxony.', aliases=('augusta marie',)),
    Document(title='Duchess Marie Elisabeth of Saxony', text='Duchess Marie Elisabeth of Saxony (22 November 1610 – 24 October 1684) was a duchess consort of Holstein-Gottorp and mother of Augusta Marie of Holstein-Gottorp.', aliases=('marie elisabeth of saxony', 'marie elisabeth',)),
    Document(title='Paris', text='Paris is the capital and largest city of France. It is located in northern France on the River Seine.', aliases=('city of paris',)),
    Document(title='France', text='France is a country in Western Europe. Its capital is Paris. The official language is French and the currency is the euro.', aliases=()),
    Document(title='Eiffel Tower', text="The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was designed by Gustave Eiffel's company and completed in 1889.", aliases=('tower of eiffel',)),
    Document(title='Gustave Eiffel', text='Alexandre Gustave Eiffel (15 December 1832 – 27 December 1923) was a French civil engineer. His company designed the Eiffel Tower.', aliases=('eiffel',)),
    Document(title='Yangtze', text='The Yangtze is the longest river in Asia and the third-longest in the world. It flows entirely within China.', aliases=('chang jiang', 'yangtze river',)),
    Document(title='China', text='China is a country in East Asia. Its capital is Beijing. The Yangtze is the longest river in China.', aliases=('prc',)),
    Document(title='Beijing', text='Beijing is the capital of China. It is a major political and cultural center.', aliases=('peking',)),
    Document(title='Alan Turing', text='Alan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician and computer scientist, widely considered the father of theoretical computer science and artificial intelligence.', aliases=('turing',)),
    Document(title='University of Cambridge', text='The University of Cambridge is a collegiate research university in Cambridge, England. Alan Turing studied there.', aliases=('cambridge university', 'cambridge',)),
    Document(title='London', text='London is the capital and largest city of England and the United Kingdom. It stands on the River Thames.', aliases=()),
    Document(title='United Kingdom', text='The United Kingdom is a country in Northwestern Europe. Its capital is London.', aliases=('uk', 'britain',)),
    Document(title='River Thames', text='The River Thames is a river that flows through southern England, including London.', aliases=('thames',)),
    Document(title='Berlin', text='Berlin is the capital and largest city of Germany.', aliases=()),
    Document(title='Germany', text='Germany is a country in Central Europe. Its capital is Berlin.', aliases=()),
    Document(title='Tokyo', text="Tokyo is the capital of Japan and one of the world's most populous metropolitan areas.", aliases=()),
    Document(title='Japan', text='Japan is an island country in East Asia. Its capital is Tokyo.', aliases=()),
    Document(title='Mount Fuji', text='Mount Fuji is the highest mountain in Japan. It is located on Honshu island near Tokyo.', aliases=('fuji',)),
    Document(title='Nile', text='The Nile is a major north-flowing river in northeastern Africa and is commonly regarded as the longest river in Africa.', aliases=()),
    Document(title='Egypt', text='Egypt is a country spanning the northeast corner of Africa. Its capital is Cairo. The Nile flows through Egypt.', aliases=()),
    Document(title='Cairo', text='Cairo is the capital of Egypt and the largest city in the Arab world.', aliases=()),
    Document(title='Amazon River', text='The Amazon River in South America is the largest river by discharge volume of water in the world.', aliases=('amazon',)),
    Document(title='Brazil', text='Brazil is the largest country in South America. Its capital is Brasília. Much of the Amazon River basin lies in Brazil.', aliases=()),
    Document(title='Brasília', text='Brasília is the capital of Brazil.', aliases=('brasilia',)),
    Document(title='Isaac Newton', text='Sir Isaac Newton (25 December 1642 – 20 March 1726/27) was an English mathematician and physicist who formulated the laws of motion and universal gravitation. He studied at the University of Cambridge.', aliases=('newton',)),
    Document(title='Marie Curie', text='Marie Skłodowska Curie (7 November 1867 – 4 July 1934) was a physicist and chemist who conducted pioneering research on radioactivity. She worked extensively in Paris, France.', aliases=('curie',)),
    Document(title='Radioactivity', text='Radioactivity is the process by which an unstable atomic nucleus loses energy by radiation. Marie Curie conducted pioneering research on radioactivity.', aliases=()),
    Document(title='Albert Einstein', text='Albert Einstein (14 March 1879 – 18 April 1955) was a theoretical physicist known for developing the theory of relativity. He was born in Ulm, Germany.', aliases=('einstein',)),
    Document(title='Theory of relativity', text='The theory of relativity encompasses special and general relativity, developed by Albert Einstein.', aliases=('relativity',)),
    Document(title='Ulm', text='Ulm is a city in the German state of Baden-Württemberg. Albert Einstein was born in Ulm.', aliases=()),
    Document(title='Ada Lovelace', text="Augusta Ada King, Countess of Lovelace (10 December 1815 – 27 November 1852), was an English mathematician known for work on Charles Babbage's Analytical Engine.", aliases=('ada lovelace',)),
    Document(title='Charles Babbage', text='Charles Babbage (26 December 1791 – 18 October 1871) was an English mathematician and inventor who originated the concept of a programmable computer.', aliases=('babbage',)),
    Document(title='Analytical Engine', text='The Analytical Engine was a proposed mechanical general-purpose computer designed by Charles Babbage. Ada Lovelace wrote notes on the engine.', aliases=()),
    Document(title='Shakespeare', text='William Shakespeare (26 April 1564 – 23 April 1616) was an English playwright and poet. He was born in Stratford-upon-Avon.', aliases=()),
    Document(title='Stratford-upon-Avon', text='Stratford-upon-Avon is a market town in England, known as the birthplace of William Shakespeare.', aliases=('stratford',)),
    Document(title='Hamlet', text='Hamlet is a tragedy written by William Shakespeare between 1599 and 1601.', aliases=()),
    Document(title='Leonardo da Vinci', text='Leonardo da Vinci (15 April 1452 – 2 May 1519) was an Italian polymath of the Renaissance. He painted the Mona Lisa.', aliases=('da vinci', 'leonardo',)),
    Document(title='Mona Lisa', text='The Mona Lisa is a portrait painting by Leonardo da Vinci, housed in the Louvre Museum in Paris, France.', aliases=()),
    Document(title='Louvre Museum', text="The Louvre Museum in Paris, France, is the world's largest art museum and a historic landmark. It houses the Mona Lisa.", aliases=('louvre',)),
    Document(title='Apollo 11', text='Apollo 11 was the spaceflight that first landed humans on the Moon on 20 July 1969. The mission was crewed by Neil Armstrong, Buzz Aldrin, and Michael Collins.', aliases=()),
    Document(title='Neil Armstrong', text='Neil Alden Armstrong (5 August 1930 – 25 August 2012) was an American astronaut and the first person to walk on the Moon during Apollo 11.', aliases=('armstrong',)),
    Document(title='Moon', text="The Moon is Earth's only natural satellite. Apollo 11 achieved the first crewed lunar landing.", aliases=()),
    Document(title='DNA', text='DNA is a molecule that carries genetic instructions. Its double-helix structure was proposed by James Watson and Francis Crick.', aliases=()),
    Document(title='James Watson', text='James Dewey Watson (born 6 April 1928) is an American molecular biologist who co-discovered the structure of DNA with Francis Crick.', aliases=('watson',)),
    Document(title='Francis Crick', text='Francis Harry Compton Crick (8 June 1916 – 28 July 2004) was a British molecular biologist who co-discovered the structure of DNA with James Watson.', aliases=('crick',)),
    Document(title='Photosynthesis', text='Photosynthesis is the process used by plants to convert light energy into chemical energy. Chlorophyll is central to the process.', aliases=()),
    Document(title='Chlorophyll', text='Chlorophyll is a green pigment found in plants that is essential for photosynthesis.', aliases=()),
    Document(title='Pacific Ocean', text="The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.", aliases=('pacific',)),
    Document(title='Atlantic Ocean', text='The Atlantic Ocean is the second-largest ocean on Earth, separating the Americas from Europe and Africa.', aliases=('atlantic',)),
    Document(title='Great Wall of China', text='The Great Wall of China is a series of fortifications built across northern China. It is a UNESCO World Heritage Site.', aliases=('great wall',)),
]


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def search_kb(query: str, *, top_k: int = 3, noise: bool = True) -> list[dict]:
    """Rank corpus documents by keyword overlap.

    Args:
        query: Search query string.
        top_k: Maximum number of documents to return.
        noise: When True, one unrelated document may be appended if available.

    Returns:
        Ranked list of document dictionaries with title, snippet, and score.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scored: list[tuple[float, Document]] = []
    for doc in DOCUMENTS:
        hay = _tokenize(doc.title + " " + doc.text + " " + " ".join(doc.aliases))
        overlap = len(q_tokens & hay)
        title_hit = 1.5 if any(t in _tokenize(doc.title) for t in q_tokens) else 0.0
        score = overlap + title_hit
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: (-x[0], x[1].title))
    results = [
        {"title": doc.title, "snippet": doc.text, "score": float(score)}
        for score, doc in scored[:top_k]
    ]

    if noise and DOCUMENTS:
        top_titles = {r["title"] for r in results}
        distractors = [d for d in DOCUMENTS if d.title not in top_titles]
        if distractors:
            d = random.choice(distractors)
            results.append({"title": d.title, "snippet": d.text, "score": 0.0})
    return results


def format_search_results(results: list[dict]) -> str:
    """Format retrieval results for insertion into the dialogue context."""
    if not results:
        return "No results found."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\n{r['snippet']}")
    return "\n\n".join(parts)
