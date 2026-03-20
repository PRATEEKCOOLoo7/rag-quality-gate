"""Simulated knowledge base representing real financial documents.

In production these come from Pinecone/ChromaDB/Weaviate.
For offline testing and demo, we use this curated corpus
of realistic financial content covering major tech companies,
market analysis, and economic indicators.

Each document has:
- id: unique document identifier
- source: origin system (SEC filing, analyst report, news, CRM)
- content: the actual text
- metadata: structured fields for filtering
"""

DOCUMENTS = [
    {
        "id": "doc_aapl_10q_2024q4",
        "source": "sec_filing",
        "content": (
            "Apple Inc reported fourth quarter revenue of 94.9 billion dollars "
            "representing a 6 percent increase year over year. iPhone revenue was "
            "46.2 billion dollars while Services revenue reached an all-time "
            "quarterly record of 25.0 billion dollars up 12 percent year over year. "
            "Greater China revenue was 15.0 billion, a decline of 3 percent. The "
            "company returned over 29 billion dollars to shareholders through "
            "dividends and share repurchases during the quarter."
        ),
        "metadata": {"company": "Apple", "ticker": "AAPL", "doc_type": "10-Q", "quarter": "Q4 2024"},
    },
    {
        "id": "doc_aapl_analyst_2024",
        "source": "analyst_report",
        "content": (
            "Apple maintains a premium valuation with a price-to-earnings ratio "
            "of approximately 28.5 compared to the S&P 500 average of 22. The "
            "company holds 162 billion dollars in cash and marketable securities "
            "providing substantial financial flexibility. Analysts maintain a "
            "consensus Buy rating with an average 12-month price target of 245 "
            "dollars. Key risks include increasing regulatory pressure in the "
            "European Union digital markets act and slowing demand in China."
        ),
        "metadata": {"company": "Apple", "ticker": "AAPL", "doc_type": "analyst_report"},
    },
    {
        "id": "doc_nvda_10q_2024q3",
        "source": "sec_filing",
        "content": (
            "NVIDIA Corporation reported record third quarter revenue of 35.1 "
            "billion dollars up 94 percent year over year driven by accelerated "
            "demand for AI infrastructure. Data Center revenue reached 30.8 "
            "billion a 112 percent increase. The company shipped its first "
            "Blackwell architecture GPUs to customers including major cloud "
            "providers. Gross margin was 74.6 percent. The company guided for "
            "fourth quarter revenue of approximately 37.5 billion dollars."
        ),
        "metadata": {"company": "NVIDIA", "ticker": "NVDA", "doc_type": "10-Q", "quarter": "Q3 2024"},
    },
    {
        "id": "doc_nvda_competition",
        "source": "market_intelligence",
        "content": (
            "NVIDIA faces growing competition in the AI accelerator market from "
            "AMD with its MI300X GPU and custom silicon from major cloud providers "
            "including Google TPU v5 Amazon Trainium and Microsoft Maia. However "
            "NVIDIA maintains dominant market share estimated at 80 percent in "
            "training workloads due to its CUDA software ecosystem and networking "
            "capabilities via InfiniBand and NVLink. Supply constraints for "
            "Blackwell GPUs are expected to persist through mid 2025."
        ),
        "metadata": {"company": "NVIDIA", "ticker": "NVDA", "doc_type": "competitive_analysis"},
    },
    {
        "id": "doc_macro_rates_2024",
        "source": "economic_research",
        "content": (
            "The Federal Reserve held the federal funds rate steady at 4.25 to "
            "4.50 percent at its December 2024 meeting citing persistent inflation "
            "above the 2 percent target. The 10 year Treasury yield stands at "
            "approximately 4.3 percent. Markets are pricing in two rate cuts in "
            "2025 with the first expected in June. Higher rates continue to "
            "pressure growth stock valuations and increase corporate borrowing "
            "costs particularly for highly leveraged companies."
        ),
        "metadata": {"topic": "monetary_policy", "doc_type": "economic_analysis"},
    },
    {
        "id": "doc_fintech_trends",
        "source": "industry_report",
        "content": (
            "The wealth management technology market is projected to reach 12.5 "
            "billion dollars by 2027 growing at 15 percent annually. Key trends "
            "include AI-powered portfolio rebalancing robo-advisory platforms "
            "and automated compliance monitoring. Registered Investment Advisors "
            "managing under 5 billion in assets are the fastest adopting segment "
            "driven by the need to compete with larger firms on technology while "
            "maintaining personalized client relationships."
        ),
        "metadata": {"sector": "fintech", "doc_type": "industry_report"},
    },
    {
        "id": "doc_crm_meridian",
        "source": "crm",
        "content": (
            "Meridian Financial Group is a mid-market wealth management firm with "
            "2.1 billion in assets under management. Based in Chicago with 85 "
            "employees and 12 financial advisors. They recently hired a CTO from "
            "Fidelity Investments to lead digital transformation. Current tech "
            "stack includes Salesforce CRM Orion portfolio management and "
            "Riskalyze for risk assessment. Annual technology budget estimated at "
            "1.5 million dollars. Primary pain point is manual client reporting "
            "which takes advisors 6 hours per week per client."
        ),
        "metadata": {"company": "Meridian Financial Group", "doc_type": "crm_record"},
    },
    {
        "id": "doc_crm_apex",
        "source": "crm",
        "content": (
            "Apex Venture Partners is a Series B focused VC firm with a 450 "
            "million dollar fund. Managing partner David Kim has been vocal about "
            "AI-native investment tools at recent conferences. The firm invested "
            "in 3 AI startups in the last quarter including an AI due diligence "
            "platform. They use Carta for cap table management and Affinity CRM. "
            "Decision making is committee-based with 4 general partners."
        ),
        "metadata": {"company": "Apex Venture Partners", "doc_type": "crm_record"},
    },
]


def search(query: str, top_k: int = 5, filter_source: str = None) -> list[dict]:
    """Simple keyword search over the document corpus.
    
    In production this would be a vector similarity search against
    Pinecone/ChromaDB/Weaviate. For testing and demo we use term overlap
    ranking which is good enough to demonstrate the pipeline.
    """
    query_terms = set(query.lower().split())
    scored = []

    for doc in DOCUMENTS:
        if filter_source and doc["source"] != filter_source:
            continue

        content_terms = set(doc["content"].lower().split())
        overlap = len(query_terms & content_terms)
        if overlap > 0:
            scored.append((overlap, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def get_by_company(company: str) -> list[dict]:
    """Retrieve all documents for a specific company."""
    company_lower = company.lower()
    return [
        doc for doc in DOCUMENTS
        if company_lower in doc.get("content", "").lower()
        or company_lower in doc.get("metadata", {}).get("company", "").lower()
    ]
