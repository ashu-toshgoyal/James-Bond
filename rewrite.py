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
what was said. If the user mentions security in any form add the specific
security terms it implies such as JWT, password hashing, input validation,
rate limiting. If the user mentions something looking good add the specific
frontend terms it implies such as responsive design, modern UI, clean layout,
consistent typography. If the user mentions speed or performance add the
specific terms it implies such as optimized queries, caching, lazy loading.
If the user mentions saving data add the specific terms it implies such as
database schema, persistent storage, CRUD operations. Never add features
that were not implied or asked for. Never explain what you changed. Never
add commentary before or after the rewritten prompt. Never use bullet points,
numbered lists, headers, or any formatting. Never use markdown of any kind.
Keep the output to one sentence maximum, short and direct.
Just return the rewritten prompt as one clean technical sentence
and absolutely nothing else. The output must be ready to feed directly into
a task weighting and agent assignment system without any further processing."""


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


if __name__ == "__main__":
    test = "yo make me a sick login page that looks clean and is super secure"

    print(f"INPUT:  {test}")
    print(f"OUTPUT: {rewrite_prompt(test)}")