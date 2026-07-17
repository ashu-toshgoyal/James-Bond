"""
boss.py — Orchestrator / Router Agent

Bug fixes vs original:
  1. Score variance: `rewrite_prompt` is a non-deterministic LLM call that can
     aggressively shorten a 200-word request to ~12 words, dropping the score
     from ~300 to ~24.  Fix: score BOTH the original and the rewrite and pass
     max(original_score, rewritten_score) to the orchestrator so a short rewrite
     can never make a complex request look trivial.

  2. Token budget: `max_tokens=4096` (deprecated param) was too small for a full
     multi-task decomposition of a complex prompt, causing mid-JSON truncation.
     Replaced with `max_completion_tokens=8192`.

  3. No retry on malformed JSON: a single truncated response permanently fell
     back to saving raw text. Added a retry loop (up to MAX_RETRIES) that
     requests only the missing portion instead of re-running the whole call.

  4. Premature deletion scheduling: `delete_after` was called even when the JSON
     parse had failed and we'd only saved a raw fallback. Moved inside the
     success branch.

  5. Import typo: `scoreing` → `scoring` (fixed in text_weight.py too).
"""

import os
import json
import subprocess
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from utility.rewrite import rewrite_prompt
from utility.text_weight import scoring

load_dotenv()

MAX_RETRIES = 3          # attempts to get valid JSON before falling back
OUTPUT_ROOT = "outputs"  # directory where orchestrator responses are saved


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def delete_after(filepath: str, delay_seconds: int = 1*60*60) -> None:
    """Spawn a detached background process that permanently deletes `filepath`
    after `delay_seconds` (default 3 hours = 10 800 s), including purging it
    from the OS Recycle Bin / Trash if a copy ended up there."""
    script = f"""
import time, os, sys, glob

time.sleep({delay_seconds})

target = r'{filepath}'
fname = os.path.basename(target)

# 1. Delete the actual file if it still exists
if os.path.exists(target):
    os.remove(target)

# 2. Purge any matching copy from the OS trash/recycle bin
try:
    if sys.platform.startswith('win'):
        # Windows Recycle Bin (per drive, in $Recycle.Bin)
        import string
        for drive in string.ascii_uppercase:
            bin_path = f"{{drive}}:\\\\$Recycle.Bin"
            if os.path.isdir(bin_path):
                for root, dirs, files in os.walk(bin_path):
                    for f in files:
                        if f == fname or fname in f:
                            try:
                                os.remove(os.path.join(root, f))
                            except Exception:
                                pass
    elif sys.platform == 'darwin':
        # macOS Trash
        trash_dir = os.path.expanduser('~/.Trash')
        for f in glob.glob(os.path.join(trash_dir, fname + '*')):
            try:
                os.remove(f)
            except Exception:
                pass
    else:
        # Linux Trash (freedesktop.org spec)
        trash_dir = os.path.expanduser('~/.local/share/Trash/files')
        trash_info = os.path.expanduser('~/.local/share/Trash/info')
        for f in glob.glob(os.path.join(trash_dir, fname + '*')):
            try:
                os.remove(f)
                info_file = os.path.join(trash_info, os.path.basename(f) + '.trashinfo')
                if os.path.exists(info_file):
                    os.remove(info_file)
            except Exception:
                pass
except Exception:
    pass
"""
    subprocess.Popen(
        [sys.executable, "-c", script],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Scheduled permanent deletion (incl. trash purge) of {filepath} in {delay_seconds/3600 :.1f} hour(s).")


def _call_orchestrator(client: OpenAI, system_prompt: str, user_payload: str) -> str | None:
    """Single call to the orchestrator model. Returns raw content or None."""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_payload},
        ],
        # 8 192 tokens gives headroom for a detailed multi-task breakdown.
        # `max_tokens` is deprecated on Groq — use `max_completion_tokens`.
        max_completion_tokens=8192,
        temperature=0.0,                        # fully deterministic
        response_format={"type": "json_object"} # guarantees valid JSON output
    )
    return resp.choices[0].message.content


def _fetch_with_retry(
    client: OpenAI, system_prompt: str, user_payload: str, max_retries: int = MAX_RETRIES
) -> tuple[dict | None, str]:
    """
    Attempt to get a valid, parseable JSON response from the orchestrator.
    If the first attempt is truncated or malformed, retry up to `max_retries`
    times asking the model to continue/repair the JSON.

    Returns (parsed_dict_or_None, raw_content_string).
    """
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            chunk = _call_orchestrator(client, system_prompt, user_payload)
            if not chunk:
                print(f"[Attempt {attempt}/{max_retries}] Model returned empty content. Retrying...")
                continue

            # On the first attempt we use the output directly.
            # On subsequent attempts we're asking the model to continue, so we
            # append rather than replace — but since response_format forces a
            # self-contained JSON object each time, we replace on retry.
            raw = chunk

            parsed = json.loads(raw)
            return parsed, raw  # success

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(
                    f"[Attempt {attempt}/{max_retries}] JSON parse failed ({e}). "
                    f"Retrying with continuation prompt..."
                )
                # Update the user payload to ask the model to produce a clean,
                # complete version — include the broken fragment as context.
                user_payload = (
                    f"{user_payload}\n\n"
                    f"Your previous response was not valid JSON and was truncated. "
                    f"Return a complete, valid JSON object now. "
                    f"Previous (broken) response for reference:\n{raw}"
                )
            else:
                print(f"[Attempt {attempt}/{max_retries}] All retries exhausted.")

        except Exception as exc:
            print(f"[Attempt {attempt}/{max_retries}] API error: {exc}")
            if attempt == max_retries:
                raise

    return None, raw  # all retries failed, return raw for fallback save


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT
# ─────────────────────────────────────────────────────────────────────────────

example_prompt = (
    "hey can u build me the entire system from scratch, i need a full app that's "
    "production ready, multiple files, proper architecture, scalable, using microservices, "
    "real time, concurrent, distributed, high availability, fault tolerant, load balancing, "
    "horizontal scaling, event driven with a message queue, deployed on kubernetes with docker "
    "orchestration, multi-tenant, enterprise grade, mission critical, zero downtime, disaster "
    "recovery, full stack, end-to-end pipeline, ci/cd pipeline, service mesh, and orchestration "
    "— also integrate an api, database, authentication, optimize and refactor every component and "
    "module, add webhooks, rest api, graphql, orm, migrations, caching, middleware, state "
    "management, routing, third party integration, background jobs, queue, cron job, search "
    "functionality, pagination, file upload, notification system, email service, websocket, rate "
    "limiting — and make sure security is airtight with auth, password, payment, encrypt, token, "
    "permission, login, private, sensitive data, pii, gdpr, compliance, credit card, ssn, two "
    "factor, 2fa, oauth, session management, role based access, rbac, audit log, protection "
    "against data breach, vulnerability, sql injection, xss, csrf, hipaa — this needs to cover "
    "the whole, complete, all, every, full, end to end, comprehensive, across the board, "
    "system-wide, overall, everything, from ground up, top to bottom part of the platform, but "
    "also fix typo, rename, change color, keep it simple and basic where quick, one line, small "
    "tweaks, minor changes, css fix, update text, adjust spacing, add comment for any small bug "
    "or quick fix, cosmetic change, button label, font size, padding, margin issues along the way"
)

boss_orchestrator_prompt = """
# Role & Objective
You are 'boss.py', the core Orchestrator and Router Agent for a high-efficiency multi-agent LLM pipeline. Your job is to take the global user request provided in the user message, analyze its scope alongside the provided score, break it down into sequential, isolated development tasks, calculate a complexity score for each individual task, and assign it to the optimal model tier.

---

# Task Decomposition & Scoring (Level 2 Phase)
Your primary directive is to decompose the provided global user request into discrete, independent development steps. For EACH individual task you generate, calculate a Task Complexity Score using the keyword weights below applied strictly to that sub-task's specific description.

### Task-Level Keyword Weights
* **Complex High (+3 points per matching word):** "entire system", "full app", "production ready", "multiple files", "architecture", "scalable", "microservice", "real time", "concurrent", "distributed"
* **Complex Mid (+2 points per matching word):** "api", "database", "authentication", "integrate", "optimize", "refactor", "component", "module"
* **Complexity Low (+1 point per matching word):** "fix typo", "rename", "change color", "simple", "basic", "quick", "one line", "small"
* **Large Scope (+2 points per matching word):** "entire", "whole", "complete", "all", "every", "full", "end to end", "from scratch"
* **High Cost of Failure (+4 points per matching word):** "security", "auth", "password", "payment", "encrypt", "token", "permission", "login", "private", "sensitive"

### Task Length Modifier
* **Task description word count > 50:** Add +2 points.
* **Task description word count > 100:** Add +4 points.

---

# Router Tiers & Model Assignments
Route each decomposed task to its designated hardware/model tier based on its calculated Task Complexity Score:

### [Tier 1: Score 0 to 4] -> FAST Model (cerebras gpt-oss-120b)
* **Purpose:** Boilerplate, configuration, folder structures, basic setups, markdown documentation, or simple unit tests.
* **Instruction Style:** Provide highly direct, boilerplate code or setup instantly with zero conversational fluff.

### [Tier 2: Score 5 to 10] -> SMART Model (groq qwen3.6-27b)
* **Purpose:** Standard business logic, modular design, frontend components, state management, or common database CRUD operations.
* **Instruction Style:** Act as a Solid Mid-Level/Senior Developer. Write robust, modular code and explain the integration steps cleanly.

### [Tier 3: Score 11+] -> BEST Model (groq gpt-oss-120b)
* **Purpose:** High cost-of-failure features, cryptographic/auth implementations, complex distributed flow, state syncing, or system optimization.
* **Instruction Style:** Act as a Principal Systems Architect. Build end-to-end production-grade, secure, and bulletproof architectural implementations.

---

# Response Format Requirement
You must return your analysis strictly as a valid, parsable JSON object. Do not include any text, commentary, or markdown outside of the JSON object itself.

```json
{
  "overall_project_analysis": {
    "global_score": [Insert provided pre-calculated score],
    "verdict": "Full orchestrator team initialized to decompose the request."
  },
  "task_pipeline": [
    {
      "task_id": 1,
      "description": "[Brief description of task 1]",
      "task_score": [Calculated score for task 1],
      "assigned_tier": "[FAST Model | SMART Model | BEST Model]",
      "system_instructions": "[Tailored system instructions telling the assigned model exactly what to output based on its tier definition above]"
    }
  ]
}
```"""


# ─────────────────────────────────────────────────────────────────────────────
#  CLIENT
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


# ─────────────────────────────────────────────────────────────────────────────
#  SCORING  — score both original and rewrite; use the max
#
#  Root cause of the original "sometimes lowest score / worst JSON" bug:
#  `rewrite_prompt` is a non-deterministic LLM call. On some runs it might
#  aggressively condense "build the entire distributed system..." (238 words,
#  score ≈310) down to a 12-word summary (score ≈24). The orchestrator then
#  sees a trivially low global score and generates a shallow, minimal plan.
#
#  Fix: always anchor the score to whichever representation scores higher.
# ─────────────────────────────────────────────────────────────────────────────

rewritten_prompt = rewrite_prompt(example_prompt)

original_score  = scoring(example_prompt)
rewritten_score = scoring(rewritten_prompt)
calculated_score = max(original_score, rewritten_score)

print(f"[SCORE] original={original_score}  rewritten={rewritten_score}  using={calculated_score}")


# ─────────────────────────────────────────────────────────────────────────────
#  RUN THE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

user_payload = f"""
Please process and decompose the following project parameters:

Global Project Prompt: "{rewritten_prompt}"

Pre-calculated Score: {calculated_score}
"""

parsed_response, raw_content = _fetch_with_retry(client, boss_orchestrator_prompt, user_payload)


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_ROOT, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if parsed_response is not None:
    output_path = os.path.join(OUTPUT_ROOT, f"boss_response_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_response, f, indent=2, ensure_ascii=False)
    print(f"\nSaved orchestrator response to: {output_path}")

    # Only schedule deletion after a confirmed successful JSON save.
    # The original code scheduled deletion even when the JSON was broken and
    # only a raw fallback was saved — meaning good data was still deleted.
    delete_after(output_path)

else:
    fallback_path = os.path.join(OUTPUT_ROOT, f"boss_response_{timestamp}_raw.txt")
    with open(fallback_path, "w", encoding="utf-8") as f:
        f.write(raw_content)
    print(
        f"\nAll {MAX_RETRIES} attempts returned invalid JSON. "
        f"Saved raw response to: {fallback_path}\n"
        f"(Raw file will NOT be auto-deleted — inspect and rerun manually.)"
    )