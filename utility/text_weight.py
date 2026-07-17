"""
utility/text_weight.py
Keyword-based complexity scorer for the boss.py orchestrator.

Bug fixes vs original:
  - Word count modifier was broken: `if x > 50 or x > 100` is identical to
    `if x > 50`, so the >100 tier never added +4 — only +2. Fixed to two-tier
    if/elif with correct point values matching the boss prompt spec.
  - Function name typo: `scoreing` → `scoring`.
"""


# ── keyword lists ──────────────────────────────────────────────────────────────

_COMPLEX_HIGH = [
    "entire system", "full app", "production ready", "multiple files",
    "architecture", "scalable", "microservice", "microservices", "real time",
    "concurrent", "distributed", "high availability", "fault tolerant",
    "load balancing", "horizontal scaling", "event driven", "message queue",
    "kubernetes", "docker orchestration", "multi-tenant", "enterprise grade",
    "mission critical", "zero downtime", "disaster recovery", "full stack",
    "end-to-end pipeline", "ci/cd pipeline", "service mesh", "orchestration",
]

_COMPLEX_MID = [
    "api", "database", "authentication", "integrate", "optimize",
    "refactor", "component", "module", "webhook", "rest api", "graphql",
    "orm", "migration", "caching", "middleware", "state management",
    "routing", "third party integration", "background job", "queue",
    "cron job", "search functionality", "pagination", "file upload",
    "notification system", "email service", "websocket", "rate limiting",
]

_COMPLEXITY_LOW = [
    "fix typo", "rename", "change color", "simple", "basic", "quick",
    "one line", "small", "tweak", "minor change", "css fix", "update text",
    "adjust spacing", "add comment", "small bug", "quick fix",
    "cosmetic change", "button label", "font size", "padding", "margin",
]

_LARGE_SCOPE = [
    "entire", "whole", "complete", "all", "every", "full", "end to end",
    "from scratch", "comprehensive", "across the board", "system-wide",
    "overall", "everything", "ground up", "top to bottom",
]

_HIGH_COST_FAILURE = [
    "security", "auth", "password", "payment", "encrypt", "token",
    "permission", "login", "private", "sensitive", "pii", "gdpr",
    "compliance", "credit card", "ssn", "two factor", "2fa", "oauth",
    "session management", "role based access", "rbac", "audit log",
    "data breach", "vulnerability", "sql injection", "xss", "csrf", "hipaa",
]


def scoring(prompt: str) -> int:
    """
    Return a complexity score for *prompt* using keyword weights and a
    two-tier word-count modifier.

    Tier weights:
      complex_high       → +3 per match
      complex_mid        → +2 per match
      complexity_low     → +1 per match
      large_scope        → +2 per match
      high_cost_failure  → +4 per match
      word count >  50   → +2
      word count > 100   → +4  (replaces the +2; not additive)
    """
    score = 0
    p = prompt.lower()

    for kw in _COMPLEX_HIGH:
        if kw in p:
            score += 3

    for kw in _COMPLEX_MID:
        if kw in p:
            score += 2

    for kw in _COMPLEXITY_LOW:
        if kw in p:
            score += 1

    for kw in _LARGE_SCOPE:
        if kw in p:
            score += 2

    for kw in _HIGH_COST_FAILURE:
        if kw in p:
            score += 4

    # Two-tier word-count modifier.
    # Original code used `if word_count > 50 or word_count > 100` which is
    # logically identical to `if word_count > 50`, so the +4 tier for long
    # prompts was never applied. Fixed to a proper if/elif.
    word_count = len(p.split())
    if word_count > 100:
        score += 4
    elif word_count > 50:
        score += 2

    return score


# ── backwards-compat alias for any callers still using the typo'd name ────────
scoreing = scoring  # noqa: N816