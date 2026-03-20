"""Hallucination detection via claim-source grounding verification.

Extracts individual claims (sentences) from the LLM response and
checks whether each claim is supported by the retrieved source
documents. Produces a continuous grounding score (0-1) rather
than binary pass/fail.

A score of 0.85 means "well grounded, minor unverifiable details."
A score of 0.40 means "significant portions have no source support."
"""

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ClaimVerification:
    claim: str
    grounded: bool
    overlap_ratio: float
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class GroundingResult:
    grounded: bool
    score: float
    total_claims: int
    verified_claims: int
    unverified: list[str] = field(default_factory=list)
    claim_details: list[ClaimVerification] = field(default_factory=list)


class HallucinationChecker:
    def __init__(self, min_claim_length: int = 20, term_min_length: int = 4,
                 match_threshold: float = 0.2, grounding_threshold: float = 0.6):
        self.min_claim_length = min_claim_length
        self.term_min_length = term_min_length
        self.match_threshold = match_threshold
        self.grounding_threshold = grounding_threshold

    def check(self, response: str, sources: list[str]) -> GroundingResult:
        if not sources:
            return GroundingResult(
                grounded=False, score=0.0, total_claims=0,
                verified_claims=0, unverified=["no source documents"],
            )

        # Build source vocabulary
        source_combined = " ".join(sources).lower()
        source_terms = set(
            w for w in source_combined.split()
            if len(w) >= self.term_min_length and w.isalpha()
        )

        # Build source number set (for financial data grounding)
        source_numbers = set()
        for source in sources:
            for word in source.split():
                cleaned = word.strip("$,%().").replace(",", "")
                try:
                    float(cleaned)
                    source_numbers.add(cleaned)
                except ValueError:
                    pass

        # Extract claims
        claims = self._extract_claims(response)
        if not claims:
            return GroundingResult(
                grounded=True, score=0.8, total_claims=0, verified_claims=0,
            )

        verified = 0
        unverified = []
        claim_details = []

        for claim in claims:
            terms = [
                w.lower() for w in claim.split()
                if len(w) >= self.term_min_length and w.isalpha()
            ]

            # Also extract numbers from claim for financial grounding
            claim_numbers = set()
            for word in claim.split():
                cleaned = word.strip("$,%().").replace(",", "")
                try:
                    float(cleaned)
                    claim_numbers.add(cleaned)
                except ValueError:
                    pass

            if not terms and not claim_numbers:
                verified += 1
                claim_details.append(ClaimVerification(
                    claim=claim[:80], grounded=True, overlap_ratio=1.0,
                ))
                continue

            # Term overlap check
            matched = [t for t in terms if t in source_terms]
            term_ratio = len(matched) / len(terms) if terms else 0

            # Number grounding check (bonus for matching financial figures)
            number_match = len(claim_numbers & source_numbers) > 0 if claim_numbers else False
            number_bonus = 0.2 if number_match else 0

            effective_ratio = min(term_ratio + number_bonus, 1.0)

            is_grounded = effective_ratio >= self.match_threshold
            if is_grounded:
                verified += 1
            else:
                unverified.append(claim[:80])

            claim_details.append(ClaimVerification(
                claim=claim[:80], grounded=is_grounded,
                overlap_ratio=round(effective_ratio, 3),
                matched_terms=matched[:5],
            ))

        score = verified / len(claims)
        grounded = score >= self.grounding_threshold

        if not grounded:
            log.warning(
                f"grounding FAIL: {score:.2f} ({verified}/{len(claims)} verified)"
            )

        return GroundingResult(
            grounded=grounded, score=round(score, 4),
            total_claims=len(claims), verified_claims=verified,
            unverified=unverified, claim_details=claim_details,
        )

    def _extract_claims(self, text: str) -> list[str]:
        raw = text.replace("\n", " ").split(".")
        return [
            s.strip() for s in raw
            if len(s.strip()) >= self.min_claim_length
        ]
