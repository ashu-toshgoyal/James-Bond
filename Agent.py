import os
import json
import subprocess
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from utility.rewrite import rewrite_prompt
from utility.text_weight import scoreing

load_dotenv()


def delete_after(filepath, delay_seconds= 2 * 60):
    """Spawn a detached background process that deletes `filepath` after `delay_seconds`."""
    deleter_script = (
        f"import time, os; "
        f"time.sleep({delay_seconds}); "
        f"os.path.exists(r'{filepath}') and os.remove(r'{filepath}')"
    )
    subprocess.Popen(
        [sys.executable, "-c", deleter_script],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Scheduled deletion of {filepath} in {delay_seconds / 60:.1f} Min(s).")


# Initialize client to target Groq's OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

example_prompt = """hey can u build me the entire system from scratch, i need a full app that's production ready, multiple files, proper architecture, scalable, using microservices, real time, concurrent, distributed, high availability, fault tolerant, load balancing, horizontal scaling, event driven with a message queue, deployed on kubernetes with docker orchestration, multi-tenant, enterprise grade, mission critical, zero downtime, disaster recovery, full stack, end-to-end pipeline, ci/cd pipeline, service mesh, and orchestration — also integrate an api, database, authentication, optimize and refactor every component and module, add webhooks, rest api, graphql, orm, migrations, caching, middleware, state management, routing, third party integration, background jobs, queue, cron job, search functionality, pagination, file upload, notification system, email service, websocket, rate limiting — and make sure security is airtight with auth, password, payment, encrypt, token, permission, login, private, sensitive data, pii, gdpr, compliance, credit card, ssn, two factor, 2fa, oauth, session management, role based access, rbac, audit log, protection against data breach, vulnerability, sql injection, xss, csrf, hipaa — this needs to cover the whole, complete, all, every, full, end to end, comprehensive, across the board, system-wide, overall, everything, from ground up, top to bottom part of the platform, but also fix typo, rename, change color, keep it simple and basic where quick, one line, small tweaks, minor changes, css fix, update text, adjust spacing, add comment for any small bug or quick fix, cosmetic change, button label, font size, padding, margin issues along the way"""



rewritten_prompt = rewrite_prompt(example_prompt)
calculated_score = scoreing(rewritten_prompt)

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
You must return your analysis strictly as a valid, parsable JSON block. Do not include any markdown commentary outside of the JSON block.

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

user_payload = f"""
Please process and decompose the following project parameters:

Global Project Prompt: "{rewritten_prompt}"

Pre-calculated Score: {calculated_score}
"""

# Executing the request through Groq's openai/gpt-oss-120b (current flagship model
# on GroqCloud as of mid-2026)
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": boss_orchestrator_prompt},
        {"role": "user", "content": user_payload},
    ],
    max_tokens=4096,  # Headroom for all per-task system instructions
    temperature=0.0,
    response_format={"type": "json_object"},
)

raw_content = response.choices[0].message.content
if raw_content is None:
    raise RuntimeError("Model returned no content in the response.")


# --- Save the response to a JSON file ---
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"boss_response_{timestamp}.json")
fallback_path = os.path.join(output_dir, f"boss_response_{timestamp}_raw.txt")

try:
    parsed_response = json.loads(raw_content)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_response, f, indent=2, ensure_ascii=False)
    print(f"\nSaved orchestrator response to: {output_path}")
except json.JSONDecodeError as e:
    # Fall back to saving the raw text so nothing is lost, even if the model
    # returned malformed/truncated JSON
    with open(fallback_path, "w", encoding="utf-8") as f:
        f.write(raw_content)
    print(f"\nWarning: response was not valid JSON ({e}). Saved raw text to: {fallback_path}")

# --- Schedule auto-deletion of the saved file after 3 hours ---
file_to_delete = output_path if os.path.exists(output_path) else fallback_path
delete_after(file_to_delete)