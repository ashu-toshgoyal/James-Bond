def scoreing(prompt:str):
    score = 0
    p = prompt.lower()

    complex_high = [
    "entire system", "full app", "production ready", "multiple files",
    "architecture", "scalable", "microservice", "microservices", "real time",
    "concurrent", "distributed", "high availability", "fault tolerant",
    "load balancing", "horizontal scaling", "event driven", "message queue",
    "kubernetes", "docker orchestration", "multi-tenant", "enterprise grade",
    "mission critical", "zero downtime", "disaster recovery", "full stack",
    "end-to-end pipeline", "ci/cd pipeline", "service mesh", "orchestration"
]

    complex_mid = [
    "api", "database", "authentication", "integrate", "optimize",
    "refactor", "component", "module", "webhook", "rest api", "graphql",
    "orm", "migration", "caching", "middleware", "state management",
    "routing", "third party integration", "background job", "queue",
    "cron job", "search functionality", "pagination", "file upload",
    "notification system", "email service", "websocket", "rate limiting"
    ]

    complexity_low = [
    "fix typo", "rename", "change color", "simple", "basic", "quick",
    "one line", "small", "tweak", "minor change", "css fix", "update text",
    "adjust spacing", "add comment", "small bug", "quick fix",
    "cosmetic change", "button label", "font size", "padding", "margin"
    ]

    large_scope = [
    "entire", "whole", "complete", "all", "every", "full", "end to end",
    "from scratch", "comprehensive", "across the board", "system-wide",
    "overall", "everything", "ground up", "top to bottom"
]

    high_cost_failure = [
    "security", "auth", "password", "payment", "encrypt", "token",
    "permission", "login", "private", "sensitive", "pii", "gdpr",
    "compliance", "credit card", "ssn", "two factor", "2fa", "oauth",
    "session management", "role based access", "rbac", "audit log",
    "data breach", "vulnerability", "sql injection", "xss", "csrf", "hipaa"
]

    for i in complex_high:
        if i in p:
            score+=3

    for i in complex_mid:
        if i in p:
            score+=2

    for i in complexity_low:
        if i in p:
            score+=1

    for i in large_scope:
        if i in p:
            score+=2

    for i in high_cost_failure:
        if i in p:
            score+=4


    word_count = len(p.split())
    if word_count > 50 or word_count > 100:
        score+=2
    
    return score
