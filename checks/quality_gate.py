"""Quality gate that orchestrates all output checks.

Runs hallucination detection, compliance checking, relevance scoring,
and response quality checks. Produces a single pass/fail decision
with a continuous overall score.

If the gate fails, the response is replaced with a safe fallback
rather than delivering potentially wrong/harmful content.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from checks.hallucination import HallucinationChecker
from checks.compliance import FinancialComplianceChecker, SAFE_FALLBACK

log = logging.getLogger(__name__)


@dataclass
class CheckDetail:
    name: str
    passed: bool
    score: float
    details: str = ""


@dataclass
class GateDecision:
    passed: bool
    score: float
    checks: list[CheckDetail]
    response: str
    fallback_used: bool = False
    issues: list[str] = field(default_factory=list)


class QualityGate:
    def __init__(self, confidence_threshold: float = 0.65):
        self.threshold = confidence_threshold
        self.hallucination = HallucinationChecker()
        self.compliance = FinancialComplianceChecker()

    def evaluate(self, response: str, sources: list[str], query: str) -> GateDecision:
        checks = []

        # 1. Hallucination / grounding
        grounding = self.hallucination.check(response, sources)
        checks.append(CheckDetail(
            name="hallucination", passed=grounding.grounded,
            score=grounding.score,
            details="" if grounding.grounded else f"{grounding.total_claims - grounding.verified_claims} unverified claims",
        ))

        # 2. Compliance
        comp = self.compliance.check(response)
        checks.append(CheckDetail(
            name="compliance", passed=comp.compliant,
            score=comp.score,
            details="" if comp.compliant else f"violations: {comp.violations}",
        ))

        # 3. Relevance to query
        rel_score = self._relevance_score(response, query)
        rel_passed = rel_score >= 0.3
        checks.append(CheckDetail(
            name="relevance", passed=rel_passed, score=rel_score,
            details="" if rel_passed else "response may not address the query",
        ))

        # 4. Response quality (length, repetition)
        qual = self._quality_score(response)
        checks.append(CheckDetail(
            name="quality", passed=qual >= 0.5, score=qual,
            details="" if qual >= 0.5 else "response too short or repetitive",
        ))

        # 5. Source availability
        has_sources = len(sources) > 0
        checks.append(CheckDetail(
            name="sources_available", passed=has_sources,
            score=1.0 if has_sources else 0.0,
            details="" if has_sources else "no source documents available",
        ))

        # Aggregate
        overall = sum(c.score for c in checks) / len(checks)
        all_pass = all(c.passed for c in checks)
        issues = [c.details for c in checks if not c.passed and c.details]

        use_fallback = not all_pass or overall < self.threshold

        if use_fallback:
            log.warning(f"GATE FAIL score={overall:.3f} issues={issues}")

        return GateDecision(
            passed=not use_fallback,
            score=round(overall, 4),
            checks=checks,
            response=response if not use_fallback else SAFE_FALLBACK,
            fallback_used=use_fallback,
            issues=issues,
        )

    @staticmethod
    def _relevance_score(response: str, query: str) -> float:
        q_terms = set(w.lower() for w in query.split() if len(w) > 3 and w.isalpha())
        r_terms = set(w.lower() for w in response.split() if len(w) > 3 and w.isalpha())
        if not q_terms:
            return 0.7
        overlap = len(q_terms & r_terms)
        return min(overlap / len(q_terms) * 1.5, 1.0)

    @staticmethod
    def _quality_score(response: str) -> float:
        if len(response) < 30:
            return 0.2
        if len(response) > 5000:
            return 0.4
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        if len(sentences) > 2:
            unique_ratio = len(set(sentences)) / len(sentences)
            if unique_ratio < 0.6:
                return 0.3
        return 1.0
