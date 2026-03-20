"""Financial compliance checker for RAG-generated responses.

Ensures that AI-generated financial content does not contain
prohibited claims (guaranteed returns, risk-free assertions)
or miss required disclaimers.
"""

import re
from dataclasses import dataclass, field


PROHIBITED = [
    (r"guaranteed?\s+returns?", "guaranteed_returns"),
    (r"risk[\s-]?free", "risk_free"),
    (r"you\s+will\s+(?:definitely|certainly|surely)\s+(?:make|earn)", "assured_profit"),
    (r"100\s*%\s+safe", "absolute_safety"),
    (r"can'?t\s+lose", "no_loss"),
    (r"guaranteed?\s+profit", "guaranteed_profit"),
    (r"no\s+risk\s+(?:at\s+all|whatsoever)", "no_risk"),
    (r"(?:buy|invest)\s+(?:now|immediately|today)", "urgency_pressure"),
    (r"insider\s+(?:info|tip|knowledge)", "insider_ref"),
]

REQUIRED_HEDGES = [
    "may", "might", "could", "potentially", "consider",
    "past performance", "not guaranteed", "risk tolerance",
    "consult", "based on available",
]

SAFE_FALLBACK = (
    "I don't have enough verified information to answer this confidently. "
    "Please consult a qualified financial advisor for personalized guidance."
)


@dataclass
class ComplianceResult:
    compliant: bool
    violations: list[str] = field(default_factory=list)
    has_hedging: bool = False
    score: float = 1.0


class FinancialComplianceChecker:
    def check(self, text: str) -> ComplianceResult:
        lower = text.lower()
        violations = []

        for pattern, label in PROHIBITED:
            if re.search(pattern, lower):
                violations.append(label)

        hedge_count = sum(1 for h in REQUIRED_HEDGES if h in lower)
        has_hedging = hedge_count >= 2

        compliant = len(violations) == 0
        score = 1.0 if compliant else max(0.0, 1.0 - 0.3 * len(violations))

        return ComplianceResult(
            compliant=compliant, violations=violations,
            has_hedging=has_hedging, score=round(score, 3),
        )
