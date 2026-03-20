"""LLM response generation with grounding context.

Takes a user query + retrieved documents and generates a grounded
response. Can use OpenAI GPT-4 (live mode) or template-based
generation (offline mode) for testing without API keys.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    query: str
    response: str
    model: str
    sources_used: int
    grounded: bool  # whether the generator had sources to ground against


SYSTEM_PROMPT = """You are a financial research assistant. Answer the user's question
based ONLY on the provided source documents. If the sources don't contain
enough information to answer confidently, say so explicitly.

Rules:
- Only state facts that are supported by the source documents
- Include specific numbers and data points from the sources
- If sources conflict, mention both perspectives
- Never make up data, projections, or recommendations
- Use hedging language for any uncertain claims
- Do not provide personalized investment advice

Source documents:
{sources}"""


class ResponseGenerator:
    def __init__(self, model: str = "gpt-4o", use_fallback: bool = True):
        self.model = model
        self.use_fallback = use_fallback
        self._llm = None

        if not use_fallback:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate

                self._llm = ChatOpenAI(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    temperature=0.1,
                    max_tokens=800,
                )
                self._prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    ("human", "{query}"),
                ])
                self._chain = self._prompt | self._llm
            except ImportError:
                log.warning("langchain not available, using fallback generator")
                self.use_fallback = True

    async def generate(self, query: str, source_contents: list[str]) -> GenerationResult:
        log.info(f"generating response for: '{query[:50]}...' with {len(source_contents)} sources")

        if not source_contents:
            return GenerationResult(
                query=query,
                response=(
                    "I don't have sufficient verified information to answer this "
                    "question confidently. The available sources do not contain "
                    "relevant data for this query."
                ),
                model="fallback",
                sources_used=0,
                grounded=False,
            )

        if self.use_fallback:
            return self._fallback_generate(query, source_contents)

        # Live LLM generation
        sources_text = "\n\n---\n\n".join(
            f"[Source {i+1}]: {s}" for i, s in enumerate(source_contents)
        )

        try:
            resp = await self._chain.ainvoke({
                "sources": sources_text,
                "query": query,
            })
            return GenerationResult(
                query=query,
                response=resp.content,
                model=self.model,
                sources_used=len(source_contents),
                grounded=True,
            )
        except Exception as e:
            log.error(f"LLM generation failed: {e}, falling back")
            return self._fallback_generate(query, source_contents)

    def _fallback_generate(self, query: str, sources: list[str]) -> GenerationResult:
        """Template-based response generation for offline testing.
        
        Extracts key facts from sources and composes a response.
        Not as fluent as LLM output but demonstrates the pipeline flow
        and allows quality gate testing without API costs.
        """
        query_lower = query.lower()

        # Extract relevant sentences from sources
        relevant = []
        for source in sources:
            sentences = [s.strip() for s in source.split(".") if len(s.strip()) > 20]
            for sent in sentences:
                # Simple relevance: query terms appear in sentence
                q_terms = set(query_lower.split())
                s_terms = set(sent.lower().split())
                if len(q_terms & s_terms) >= 2:
                    relevant.append(sent)

        if not relevant:
            # Fall back to first sentences of each source
            for source in sources[:3]:
                first_sent = source.split(".")[0].strip()
                if len(first_sent) > 20:
                    relevant.append(first_sent)

        # Compose response
        unique_facts = list(dict.fromkeys(relevant))[:5]  # deduplicate, keep order

        if not unique_facts:
            response = (
                "Based on the available sources, I was unable to find specific "
                "information directly addressing this query. The documents cover "
                "related topics but may not contain the exact data requested."
            )
        else:
            response_parts = []
            response_parts.append("Based on the available sources:")
            for i, fact in enumerate(unique_facts):
                clean = fact.strip().rstrip(".")
                response_parts.append(f"{clean}.")
            response_parts.append(
                "Note that this information is based on available sources and "
                "may not reflect the most current data. Consider consulting "
                "additional sources for a complete picture."
            )
            response = " ".join(response_parts)

        return GenerationResult(
            query=query,
            response=response,
            model="fallback_template",
            sources_used=len(sources),
            grounded=True,
        )
