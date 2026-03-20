"""Input sanitization layer for RAG pipeline.

Blocks malicious inputs before they reach the LLM or pollute
the retrieval context. Catches prompt injection, jailbreaks,
system prompt extraction, encoding evasion, and delimiter attacks.

This runs BEFORE retrieval — a blocked input never touches
the vector store or the LLM, saving cost and preventing
context poisoning.
"""

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---- Pattern databases ----

INJECTION = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?prior",
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"act\s+as\s+(?:if|though)",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"\]\s*\n?\s*\[system",
    r"OVERRIDE",
    r"bypass\s+(?:safety|content|filter|guard)",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"do\s+anything\s+now",
    r"you\s+have\s+no\s+(?:rules|restrictions|limits)",
]

EXTRACTION = [
    r"what\s+(?:are|is)\s+your\s+(?:system|initial)\s+(?:prompt|instructions)",
    r"repeat\s+(?:your|the)\s+(?:system|initial)\s+(?:prompt|instructions)",
    r"show\s+me\s+your\s+(?:system|hidden)",
    r"output\s+(?:your|the)\s+(?:system|initial)\s+prompt",
    r"print\s+your\s+(?:system|initial|hidden)",
    r"reveal\s+your\s+(?:instructions|prompt|rules)",
    r"what\s+were\s+you\s+told\s+to\s+do",
]

EVASION = [
    r"base64\s*[:\(]",
    r"rot13\s*[:\(]",
    r"decode\s+(?:this|the\s+following)",
    r"\\x[0-9a-fA-F]{2}",
    r"\\u[0-9a-fA-F]{4}",
    r"eval\s*\(",
    r"exec\s*\(",
    r"hex\s*\(",
    r"chr\s*\(",
]

COT_MANIPULATION = [
    r"let'?s\s+think\s+step\s+by\s+step.*(?:ignore|forget|disregard)",
    r"step\s+\d+.*(?:ignore|bypass|override)",
    r"first.*(?:forget|ignore).*then",
    r"reasoning:.*(?:override|bypass)",
]

DELIMITER_ATTACKS = [
    r"```\s*system",
    r"<\|im_start\|>system",
    r"\[INST\].*\[/INST\]",
    r"###\s*(?:System|Human|Assistant)\s*:",
    r"<\|system\|>",
]

INDIRECT_INJECTION = [
    r"important\s*:\s*(?:ignore|disregard|override|skip)",
    r"note\s+to\s+ai",
    r"instruction\s+override",
    r"ai\s+directive",
    r"hidden\s+instruction",
    r"ignore\s+(?:all\s+)?(?:quality|safety|compliance)\s+checks",
]


@dataclass
class SanitizeResult:
    safe: bool
    original: str
    cleaned: str
    threats: list[str] = field(default_factory=list)
    risk: float = 0.0
    category: str = ""

    @property
    def blocked(self) -> bool:
        return not self.safe


class InputSanitizer:
    MAX_LEN = 4000
    CHAR_RATIO_LIMIT = 0.25

    def __init__(self):
        self._pattern_groups = [
            ("injection", INJECTION, 0.4),
            ("extraction", EXTRACTION, 0.5),
            ("evasion", EVASION, 0.3),
            ("cot_manipulation", COT_MANIPULATION, 0.4),
            ("delimiter_attack", DELIMITER_ATTACKS, 0.35),
            ("indirect_injection", INDIRECT_INJECTION, 0.45),
        ]

    def check(self, text: str) -> SanitizeResult:
        threats = []
        risk = 0.0
        categories_hit = set()

        # Length check
        if len(text) > self.MAX_LEN:
            threats.append(f"length:{len(text)}")
            risk += 0.15

        # Empty/whitespace check
        if not text.strip():
            return SanitizeResult(safe=True, original=text, cleaned=text)

        # Pattern scanning
        lower = text.lower()
        for category, patterns, weight in self._pattern_groups:
            hits = [p[:45] for p in patterns if re.search(p, lower)]
            if hits:
                categories_hit.add(category)
                for h in hits:
                    threats.append(f"{category}:{h}")
                risk += weight * len(hits)

        # Special character ratio
        ratio = self._char_ratio(text)
        if ratio > self.CHAR_RATIO_LIMIT:
            threats.append(f"suspicious_chars:{ratio:.2f}")
            risk += 0.1

        # Multi-line with role markers (common in indirect injection)
        if re.search(r"\n\s*(?:system|assistant|human)\s*:", lower):
            threats.append("role_marker_in_input")
            risk += 0.35
            categories_hit.add("role_injection")

        risk = min(risk, 1.0)
        safe = risk < 0.3

        primary_category = ""
        if categories_hit:
            # Highest-severity category first
            priority = ["extraction", "injection", "indirect_injection",
                        "cot_manipulation", "delimiter_attack", "evasion", "role_injection"]
            for p in priority:
                if p in categories_hit:
                    primary_category = p
                    break

        if not safe:
            log.warning(f"BLOCKED risk={risk:.2f} cat={primary_category} threats={len(threats)}")

        return SanitizeResult(
            safe=safe, original=text,
            cleaned=text if safe else "",
            threats=threats, risk=round(risk, 4),
            category=primary_category,
        )

    def check_retrieved_context(self, documents: list[str]) -> list[SanitizeResult]:
        """Scan retrieved documents for indirect injection.
        
        Attackers can embed malicious instructions in documents
        that get retrieved by RAG and injected into the LLM context.
        This checks each retrieved chunk for embedded instructions.
        """
        results = []
        for doc in documents:
            result = self.check(doc)
            if not result.safe:
                result.category = f"indirect:{result.category}"
            results.append(result)
        return results

    @staticmethod
    def _char_ratio(text: str) -> float:
        if not text:
            return 0.0
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special / len(text)
