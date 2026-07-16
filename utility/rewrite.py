from cerebras.cloud.sdk import Cerebras
import os
from dotenv import load_dotenv

load_dotenv()

client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

CONTEXT = """Take whatever the user says — no matter how casual, misspelled, vague, slang,
informal, or incomplete — and rewrite it as clean precise technical language
that a senior software engineer would write in a professional brief. Preserve
the exact intent of the original request without adding any features that were
not asked for and without removing any features that were asked for. Convert
slang to proper technical terms, fix all spelling mistakes, expand vague or
informal words into specific technical ones, make any hidden complexity
explicitly visible in the output, and infer reasonable technical context from
what was said.

If the user implies a large, system-wide, or production-grade build, use exact
terms such as entire system, full app, production ready, multiple files,
architecture, scalable, microservice, microservices, real time, concurrent,
distributed, high availability, fault tolerant, load balancing, horizontal
scaling, event driven, message queue, kubernetes, docker orchestration,
multi-tenant, enterprise grade, mission critical, zero downtime, disaster
recovery, full stack, end-to-end pipeline, ci/cd pipeline, service mesh, or
orchestration — whichever ones genuinely match what was asked.

If the user implies standard backend, integration, or business-logic work, use
exact terms such as api, database, authentication, integrate, optimize,
refactor, component, module, webhook, rest api, graphql, orm, migration,
caching, middleware, state management, routing, third party integration,
background job, queue, cron job, search functionality, pagination, file
upload, notification system, email service, websocket, or rate limiting —
whichever ones genuinely match what was asked.

If the user mentions security, privacy, or payments in any form, use exact
terms such as security, auth, password, payment, encrypt, token, permission,
login, private, sensitive, pii, gdpr, compliance, credit card, ssn, two
factor, 2fa, oauth, session management, role based access, rbac, audit log,
data breach, vulnerability, sql injection, xss, csrf, or hipaa — whichever
ones genuinely match what was asked.

If the user implies the work spans everything or the whole project rather
than one piece, use exact terms such as entire, whole, complete, all, every,
full, end to end, from scratch, comprehensive, across the board, system-wide,
overall, everything, ground up, or top to bottom — whichever ones genuinely
match what was asked.

If the user mentions something looking good, add the specific frontend terms
it implies such as responsive design, modern UI, clean layout, and consistent
typography. If the user mentions speed or performance, add the specific terms
it implies such as optimized queries, caching, and lazy loading. If the user
mentions saving data, add the specific terms it implies such as database
schema, persistent storage, and CRUD operations.

If the request is small, cosmetic, or trivial, use exact terms such as fix
typo, rename, change color, simple, basic, quick, one line, small, tweak,
minor change, css fix, update text, adjust spacing, add comment, small bug,
quick fix, cosmetic change, button label, font size, padding, or margin —
whichever ones genuinely match what was asked — and keep the rewritten output
equally small and precise, without inflating it with unrelated technical
terms or unnecessary scope.

Never add features that were not implied or asked for. Never use a keyword
from a category that does not genuinely apply to the request. Never explain
what you changed. Never add commentary before or after the rewritten prompt.
Never use bullet points, numbered lists, headers, or any formatting. Never
use markdown of any kind. Keep the output to one sentence maximum, short and
direct. Just return the rewritten prompt as one clean technical sentence and
absolutely nothing else. The output must be ready to feed directly into a
task weighting and agent assignment system without any further processing."""

def rewrite_prompt(raw_prompt: str) -> str:
    """
    Takes a raw casual user prompt and returns a clean technical version.
    Returns original prompt as fallback if model fails.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": CONTEXT},
                {"role": "user",   "content": raw_prompt},
            ],
            max_tokens=500,             # ← enough for reasoning + answer
        )

        message = response.choices[0].message

        # content is the actual answer
        if hasattr(message, "content") and message.content:
            return message.content.strip()

        # fallback — extract last line of reasoning if content empty
        if hasattr(message, "reasoning") and message.reasoning:
            lines  = [l.strip() for l in message.reasoning.strip().split("\n") if l.strip()]
            return lines[-1] if lines else raw_prompt

        return raw_prompt               # last resort — return original

    except Exception as e:
        print(f"[REWRITER ERROR] {e}")
        return raw_prompt               # never crash — return original


# if __name__ == "__main__":
#     test = input("Enter prompt: ")
#     while(test != "quit"):
#         test = input("Enter prompt: ")
#         print(f"INPUT:  {test}")
#         print(f"OUTPUT: {rewrite_prompt(test)}")