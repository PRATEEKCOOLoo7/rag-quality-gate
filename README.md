# RAG Quality Gate

A production-pattern RAG (Retrieval-Augmented Generation) pipeline with a built-in quality gate that validates every AI response for hallucination, factual grounding, tone compliance, and adversarial input detection before delivery.

Inspired by the "Do Not Embarrass Me" design principle — every output is checked before it reaches a user.

## Architecture

```
User Query
    │
    ▼
┌──────────────────┐
│  Input Sanitizer  │ ← Prompt injection detection, jailbreak prevention
└────────┬─────────┘
         │ (clean query)
         ▼
┌──────────────────┐     ┌──────────────────┐
│  RAG Retriever    │────▶│  Vector Store     │
│  (Hybrid Search)  │     │  (Pinecone)       │
└────────┬─────────┘     └──────────────────┘
         │ (context chunks)
         ▼
┌──────────────────┐
│  LLM Generator    │ ← Generates grounded response
└────────┬─────────┘
         │ (raw response)
         ▼
┌──────────────────────────────────────────┐
│            QUALITY GATE                   │
│                                           │
│  ✓ Hallucination Detection               │
│  ✓ Factual Grounding Score               │
│  ✓ Tone & Compliance Check               │
│  ✓ Confidence Threshold                  │
│  ✓ Prohibited Content Filter             │
│                                           │
│  Pass → Deliver    Fail → Fallback/Human │
└──────────────────────────────────────────┘
```

## Key Features

- **Input Sanitization**: Detects and blocks prompt injection attempts, jailbreak patterns, and malicious inputs before they reach the LLM
- **Hybrid RAG Retrieval**: Combines dense (embedding) and sparse (keyword) retrieval for better context coverage
- **Hallucination Detection**: Cross-references every claim in the LLM response against retrieved source documents
- **Factual Grounding Score**: Quantifies how well the response is supported by retrieved evidence
- **Tone & Compliance**: Validates output tone and checks for prohibited financial/legal claims
- **Confidence-Based Fallback**: Low-confidence responses route to a safe fallback instead of delivering potentially wrong answers

## Tech Stack

- **LangChain** — LLM orchestration
- **OpenAI GPT-4** — Generation
- **Pinecone** — Vector store
- **sentence-transformers** — Embedding model
- Python 3.11+

## Project Structure

```
rag-quality-gate/
├── README.md
├── requirements.txt
├── config.py
├── main.py                     # Demo runner
├── pipeline/
│   ├── __init__.py
│   ├── input_sanitizer.py      # Prompt injection & jailbreak detection
│   ├── retriever.py            # Hybrid RAG retrieval
│   ├── generator.py            # LLM response generation
│   └── quality_gate.py         # Multi-check validation layer
├── checks/
│   ├── __init__.py
│   ├── hallucination.py        # Claim-vs-source verification
│   ├── grounding.py            # Factual grounding scorer
│   ├── compliance.py           # Tone & prohibited content
│   └── confidence.py           # Confidence threshold logic
└── tests/
    ├── test_sanitizer.py       # Adversarial input tests
    ├── test_quality_gate.py
    └── test_hallucination.py
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/rag-quality-gate.git
cd rag-quality-gate
pip install -r requirements.txt

export OPENAI_API_KEY=your_key
export PINECONE_API_KEY=your_key

python main.py
pytest tests/ -v
```

## Design Decisions

- **Quality gate at delivery, not retrieval**: The gate checks the final generated response, not the retrieved chunks. This catches hallucinations the LLM introduces even when retrieval is perfect.
- **Input sanitization as first layer**: Adversarial inputs are blocked before they consume LLM tokens or pollute the retrieval context.
- **Grounding score, not binary pass/fail**: A continuous 0-1 grounding score lets downstream systems make nuanced decisions (e.g., show low-grounding responses with a disclaimer rather than blocking entirely).

