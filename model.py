from dotenv import load_dotenv
from openai import OpenAI
from google import genai
import os
import re
from datetime import datetime
import json
import time
import requests

load_dotenv()

# ── model registry — all verified working models only ─────────────────────────
Model = {
    "fast":        "openai/gpt-oss-20b",        # replaces deprecated llama-3.1-8b
    "smart":       "openai/gpt-oss-120b",        # replaces gemini (not working)
    "heavy":       "openai/gpt-oss-120b",        # replaces gemini (not working)
    "code":        "qwen/qwen3.6-27b",           # replaces deprecated qwen3-32b
    "code_backup": "openai/gpt-oss-120b",        # fallback if qwen busy
    "math":        "openai/gpt-oss-120b",        # groq only — gemini not working
    "creative":    "openai/gpt-oss-120b",        # groq only — gemini not working
    "translate":   "openai/gpt-oss-20b",         # replaces deprecated llama-3.3-70b
    "system":      "openai/gpt-oss-20b",         # replaces deprecated llama-3.3-70b
}

# ── groq client ────────────────────────────────────────────────────────────────
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# ── gemini openai-compat client (kept as optional fallback) ───────────────────
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ── native google genai client ────────────────────────────────────────────────
gemini_native = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── all modes use groq — gemini kept as optional fallback only ─────────────────
GROQ_MODES = {
    "fast", "smart", "heavy", "code", "code_backup",
    "math", "creative", "translate", "system"
}

SESSION_FILE = "session.json"
TIMING_FILE  = "time.json"

SYSTEM_PROMPT = """
You are an AI assistant in the Jarvis mold — composed, precise, quietly competent.
You don't perform enthusiasm and you don't narrate your own helpfulness. You just
handle things, efficiently, and say exactly as much as is useful and not one
sentence more.

You never open with "Sure!", "Great question!", "Of course!", "Certainly!" or
similar filler. You simply answer.


FORMAT — the only rule that never bends

Decide first whether the reply needs a code block.

If it does:
<code>
...
</code>
<response>
...
</response>

If it doesn't, skip the <code> block entirely and reply with just:
<response>
...
</response>

Never write the word "None" or any placeholder where code would go. Either
there's a real code block, or there's no code block at all — nothing in between.

- <code> gets raw runnable code only. No explanations, no markdown fences.
- <response> gets ONLY plain spoken English text. Never put code, file contents,
  file paths, or technical output inside <response>. If you catch yourself
  writing code or a file path inside <response>, stop and delete it.
- The response block should sound like someone explaining what they built out
  loud — no code whatsoever, not even a single line or filename in code format.


DEFAULT TO SUBSTANCE — match depth to what was asked

- Quick factual question: two to three sentences with context, not just the raw answer.
- Code request: build the complete thing in <code>. In <response>, explain what
  you built, why you made key decisions, what to watch out for, and what to do next.
  Be thorough — the person needs to understand what they are working with.
- Explain X / help me think through X: give a proper explanation with reasoning,
  examples, and analogies where they help. Do not stop when the point is technically
  made — make sure it actually lands.
- Deep dive / research / full analysis: go long and cover it properly. Multiple
  angles, trade-offs, real depth. Written like a person, not a report.
- Creative writing: write the actual piece in full with genuine craft and length
  appropriate to the form.
- Conversational question: respond like a knowledgeable friend — warm, complete,
  not clipped. Add relevant context the person probably wants even if they did
  not explicitly ask for it.
- Never cut an answer short just to seem efficient. Completeness is more valuable
  than brevity. If in doubt, say more not less.


HOW YOU WRITE CODE

Complete, production-ready, runs as-is. Comments explain why not what.
Modern idioms. Handle edge cases, validate inputs. Never hardcode secrets.
Never write anything built to attack a system the person doesn't own.


PERSONALITY

Dry wit when it fits — not performed. You have opinions and say if an approach
is flawed, briefly, then offer the better one. Don't over-apologize. If
something can't be done, say so and move on to what can.


WHAT YOU WON'T DO

No malware, exploits, or anything built to harm systems or people. Nothing
illegal. A fictional wrapper doesn't change the answer.
"""

KEYWORD_MAP = {
    "fast": [
        "quick", "fast", "simple", "brief", "short", "tell me", "what is",
        "who is", "when", "where", "define", "give me", "just", "only",
        "one line", "quickly", "summary", "tldr", "in short", "meaning of",
        "definition", "quick answer", "simple answer", "in one word",
        "one word answer", "yes or no", "quick fact", "fact check",
        "how many", "how much", "how far", "how long", "how old",
        "what year", "what time", "current", "today", "right now",
        "instantly", "asap", "in a nutshell", "briefly", "short answer",
        "quick note", "quick tip", "simple explanation", "basic", "basics",
        "at a glance", "snapshot", "concise", "to the point",
        "straightforward", "plain answer", "one sentence", "single word",
        "name of", "spell", "pronounce", "abbreviation", "full form",
        "acronym for", "stands for", "capital of", "population of",
        "distance between", "currency of", "time zone", "date of",
        "born on", "died on", "founded in", "located in", "quick fix",
        "quick check", "fast answer", "rapid", "instant", "immediately",
        "right away", "in brief", "sum up", "condense", "shorten",
        "skip the details", "in a word", "one word", "quick recap",
        "express version", "no fluff", "no explanation", "skip explanation",
        "cut to the chase", "bottom line", "key point", "main point",
        "headline", "fast facts", "trivia", "random fact", "did you know",
        "quick trivia", "speed answer", "rapid fire", "lightning answer",
        "in a flash", "real quick", "super quick", "super fast",
        "in seconds", "30 second answer", "elevator pitch", "gist",
        "essence of", "core idea", "main idea", "key takeaway", "takeaway",
        "highlights", "recap", "recap quickly", "name only", "just the name",
        "just the number", "exact number", "exact date", "precise answer",
        "direct answer", "straight answer", "no context needed",
        "skip intro", "skip background", "without details",
        "minimal answer", "short and sweet", "snap answer", "fact only",
        "raw fact", "one liner", "single line", "tiny answer",
        "micro answer", "tiny summary", "headline only", "title only",
        "label", "tag", "id of", "code name of", "abbrev", "short form",
        "short name", "common name", "nickname", "alias", "aka",
        "what's it called", "called what", "known as", "referred to as",
        "year of", "month of", "day of", "hour of", "exact time",
        "current time", "current date", "today's date", "fast please",
        "quick please", "make it short", "keep it short", "keep it brief",
        "shorten this", "trim this", "compress this", "tldr please",
        "give tldr", "1 line answer", "single word reply", "quick one",
        "fast one", "two words or less", "short reply", "no long answer"
    ],
    "smart": [
        "explain", "help", "how to", "suggest", "idea", "plan", "should i",
        "recommend", "difference", "why", "understand", "guide", "steps",
        "teach", "show me", "walk me", "approach", "strategy", "best way",
        "advice", "opinion", "think", "improve", "review", "clarify",
        "clarification", "can you explain", "help me understand",
        "what should i do", "how does it work", "how does this work",
        "give advice", "your thoughts", "your opinion", "feedback",
        "suggestion", "recommendation", "tips for", "tips on", "guidance",
        "mentor", "coach me", "walk through", "talk me through",
        "break it down", "simplify", "make it simple", "elaborate a bit",
        "in simple terms", "layman's terms", "eli5", "explain like",
        "help me decide", "decision", "choose between", "which is better",
        "what's better", "how can i", "ways to", "methods to",
        "options for", "what are my options", "next steps", "action plan",
        "roadmap", "framework", "outline a plan", "brainstorm",
        "ideas for", "suggest ways", "how should i", "what's the best",
        "point of view", "perspective", "insight", "reasoning",
        "rationale", "justify", "convince me", "persuade",
        "argument for", "argument against", "debate", "discuss",
        "discussion", "let's talk about", "thoughts on", "view on",
        "stance on", "take on", "is it a good idea", "worth it",
        "pros or cons", "what do you suggest", "what's the deal with",
        "help me figure out", "figure out", "figuring out",
        "make sense of", "make sense", "help me with", "assist me",
        "assist with", "guide me", "guide me through", "lead me",
        "lay out", "lay it out", "spell it out", "unpack", "unpack this",
        "demystify", "decode", "decode this", "deconstruct",
        "what's going on with", "what's happening with", "navigate",
        "navigating", "handle this", "deal with this", "cope with",
        "manage this", "manage", "tackle this", "approach this",
        "best approach", "best practice", "best practices",
        "right way to", "correct way to", "proper way", "wrong way",
        "common mistake", "avoid mistakes", "do's and don'ts",
        "checklist for", "template for", "blueprint", "playbook",
        "game plan", "strategy for", "tactic", "tactics for",
        "method for", "technique", "techniques for",
        "how should this work", "expectation", "what to expect",
        "what should happen", "is this normal", "is this ok",
        "is this fine", "is this good", "is this bad", "good idea or bad idea",
        "smart move", "wise choice", "right choice", "second opinion",
        "another perspective", "different angle", "alternative view",
        "counter argument", "rebuttal", "weigh in", "weigh in on",
        "chime in", "input on", "input needed", "need your input",
        "need help deciding", "torn between", "stuck between",
        "can't decide", "help me choose", "which one should i pick",
        "which option", "evaluate options", "compare options",
        "rank these", "rate this idea", "critique this",
        "constructive feedback", "honest feedback", "be honest",
        "tell me honestly", "what do you recommend",
        "recommend something", "your recommendation", "advise me",
        "advise on", "counsel", "counseling on", "life advice",
        "career advice", "relationship advice", "study tips",
        "productivity tips", "time management", "motivation",
        "how to improve", "ways to improve", "self improvement",
        "growth mindset", "personal development"
    ],
    "heavy": [
        "research", "deep", "analyze", "analyse", "compare", "detail",
        "study", "report", "summarize", "summarise", "thesis",
        "pros and cons", "in depth", "breakdown", "elaborate",
        "comprehensive", "thorough", "investigate", "evaluate", "assess",
        "critical", "academic", "extensive", "full analysis", "deep dive",
        "explain everything", "in detail", "literature review",
        "case study", "white paper", "dissertation", "meta analysis",
        "systematic review", "data analysis", "statistical analysis",
        "root cause analysis", "swot analysis", "market research",
        "competitive analysis", "trend analysis", "historical analysis",
        "comparative study", "exhaustive", "all details",
        "full breakdown", "detailed report", "detailed explanation",
        "point by point", "line by line", "deep research",
        "in-depth review", "deep evaluation", "rigorous", "methodology",
        "hypothesis", "peer reviewed", "citations", "references",
        "sources", "bibliography", "scholarly", "paper on", "essay on",
        "long form", "long answer", "lengthy explanation", "multi page",
        "full report", "complete analysis", "holistic view", "360 view",
        "end to end analysis", "granular", "nuanced", "multi-faceted",
        "exploratory analysis", "qualitative analysis",
        "quantitative analysis", "feasibility study", "risk assessment",
        "audit", "benchmark", "deep comparison", "long report",
        "extensive report", "all-encompassing", "wide ranging",
        "expansive", "expanded analysis", "thorough investigation",
        "deep look", "closer look", "in-depth look", "review in detail",
        "scrutinize", "scrutiny", "dissect", "dissect this", "pick apart",
        "examine closely", "examine in detail", "rigorous analysis",
        "rigorous study", "academic paper", "research paper",
        "research report", "field study", "longitudinal study",
        "cross sectional study", "qualitative research",
        "quantitative research", "mixed methods", "primary research",
        "secondary research", "desk research", "fact finding",
        "due diligence", "deep audit", "comprehensive audit", "full audit",
        "exhaustive search", "thorough search", "extensive search",
        "wide research", "broad research", "macro analysis",
        "micro analysis", "sector analysis", "industry analysis",
        "company analysis", "financial analysis", "investment analysis",
        "policy analysis", "impact analysis", "cost benefit analysis",
        "gap analysis", "needs analysis", "scenario analysis",
        "what if analysis", "sensitivity analysis", "forecast analysis",
        "predictive analysis", "trend forecasting", "long term outlook",
        "deep understanding", "complete picture", "full picture",
        "everything about", "all about", "everything you know about",
        "tell me everything", "give me all details",
        "exhaustively explain", "phd level", "expert level",
        "in great detail", "very detailed", "extremely detailed",
        "thorough write up", "long write up", "lengthy report",
        "big picture analysis", "context and detail", "history and detail",
        "background and analysis"
    ],
    "code": [
        "code", "write", "build", "fix", "function", "bug", "website",
        "script", "program", "class", "api", "database", "error", "debug",
        "create", "implement", "develop", "deploy", "refactor", "optimize",
        "algorithm", "library", "module", "frontend", "backend",
        "fullstack", "html", "css", "javascript", "python", "typescript",
        "react", "django", "flask", "sql", "mongodb", "node", "nodejs",
        "express", "vue", "angular", "next.js", "nextjs", "java", "c++",
        "c#", "golang", "go lang", "rust", "php", "ruby", "rails",
        "swift", "kotlin", "dart", "flutter", "docker", "kubernetes",
        "git", "github", "gitlab", "ci/cd", "pipeline", "regex", "json",
        "xml", "yaml", "rest api", "graphql", "websocket", "oop",
        "object oriented", "data structure", "linked list", "binary tree",
        "recursion", "compile", "compilation", "syntax error",
        "runtime error", "exception", "traceback", "stack trace",
        "unit test", "test case", "pytest", "jest", "npm", "pip install",
        "virtual environment", "package", "dependency", "framework",
        "import", "variable", "for loop", "while loop", "array",
        "dictionary", "hashmap", "sdk", "cli", "command line",
        "shell script", "bash script", "cron job", "microservice",
        "server", "client side", "server side", "authentication",
        "authorization", "jwt", "oauth", "encryption", "hashing",
        "web scraping", "automation script", "selenium", "beautifulsoup",
        "pandas", "numpy", "tensorflow", "pytorch", "neural network",
        "app development", "mobile app", "ios app", "android app",
        "game dev", "unity", "unreal engine", "sql query", "no sql",
        "firebase", "aws", "azure", "gcp", "lambda function", "endpoint",
        "middleware", "schema", "orm", "sqlalchemy", "css grid",
        "flexbox", "responsive design", "dom", "async", "await",
        "promise", "callback", "event listener", "sql injection",
        "vulnerability", "patch", "hotfix", "merge conflict",
        "pull request", "commit", "branch", "version control",
        "compile error", "stack overflow", "web app", "web application",
        "single page app", "spa", "progressive web app", "pwa",
        "responsive site", "landing page", "portfolio site",
        "ecommerce site", "shopping cart code", "payment gateway",
        "stripe integration", "twilio", "sendgrid", "email api",
        "cron expression", "regex pattern", "string manipulation",
        "array methods", "list comprehension", "lambda function (python)",
        "closures", "generators", "decorators", "context manager",
        "exception handling", "try except", "try catch", "null pointer",
        "undefined error", "type error", "value error", "index error",
        "key error", "memory leak", "race condition", "deadlock",
        "thread", "multithreading", "multiprocessing", "concurrency",
        "parallel processing", "load balancing", "caching", "redis",
        "memcached", "rabbitmq", "kafka", "message queue",
        "event driven", "pub sub", "websocket server", "socket programming",
        "tcp ip", "http request", "http response", "status code",
        "cors error", "cross origin", "csrf", "xss", "input validation",
        "form validation", "data validation", "schema validation",
        "openapi", "swagger", "postman", "curl command", "fetch api",
        "axios", "ajax call", "dom event", "event bubbling",
        "state management", "redux", "context api", "hooks",
        "use effect", "use state", "component lifecycle", "props drilling",
        "jsx syntax", "tailwind css", "bootstrap", "sass", "scss",
        "webpack", "vite", "babel", "eslint", "prettier", "linting",
        "code review", "code optimization", "performance tuning",
        "big o notation", "time complexity", "space complexity",
        "sorting algorithm", "searching algorithm", "dynamic programming",
        "greedy algorithm", "graph algorithm", "tree traversal", "dfs",
        "bfs", "leetcode", "coding interview", "system design",
        "design pattern", "singleton pattern", "factory pattern",
        "mvc architecture", "mvvm", "clean code", "solid principles",
        "tdd", "test driven development", "integration test",
        "end to end test", "mock data", "stub", "fixture", "ide",
        "vscode", "pycharm", "jupyter notebook", "colab", "anaconda",
        "venv", "requirements.txt", "package.json", "dockerfile",
        "docker compose", "yaml config", "env file", ".env", "api key",
        "secret key", "token", "rate limiting", "webhook",
        "cron syntax", "shell command", "linux command", "bash command",
        "powershell script", "batch file", "makefile", "build script",
        "deployment script", "ci pipeline", "github actions", "jenkins",
        "vercel deploy", "netlify deploy", "heroku deploy"
    ],
    "math": [
        "calculate", "solve", "equation", "integral", "derivative",
        "matrix", "algebra", "geometry", "trigonometry", "calculus",
        "formula", "proof", "theorem", "graph", "plot", "statistics",
        "probability", "sum", "multiply", "divide", "subtract", "add",
        "percentage", "percent", "ratio", "fraction", "decimal",
        "square root", "cube root", "exponent", "power of", "logarithm",
        "log", "factorial", "permutation", "combination", "mean",
        "median", "mode", "standard deviation", "variance", "regression",
        "linear equation", "quadratic equation", "polynomial", "vector",
        "scalar", "eigenvalue", "eigenvector", "differential equation",
        "limit", "series", "sequence", "arithmetic", "geometric series",
        "number theory", "prime number", "gcd", "lcm", "modulus",
        "area of", "volume of", "perimeter", "circumference", "angle",
        "radius", "diameter", "sin cos tan", "pythagorean theorem",
        "simplify", "factorize", "expand", "solve for x", "solve for y",
        "word problem", "math problem", "numeric", "numerical",
        "compute", "computation", "convert units", "unit conversion",
        "currency conversion", "interest rate", "compound interest",
        "simple interest", "profit and loss", "speed distance time",
        "age problem", "set theory", "venn diagram", "binomial",
        "normal distribution", "hypothesis testing", "p value",
        "confidence interval", "optimization problem",
        "linear programming", "long division", "long multiplication",
        "carrying numbers", "rounding numbers", "estimation",
        "place value", "prime factorization", "lowest common denominator",
        "highest common factor", "least common multiple",
        "ratio and proportion", "direct proportion", "inverse proportion",
        "unit rate", "rate of change", "slope of a line", "y intercept",
        "x intercept", "domain and range", "function notation",
        "inverse function", "composite function", "even and odd function",
        "asymptote", "discontinuity", "continuity",
        "intermediate value theorem", "mean value theorem",
        "taylor series", "maclaurin series", "fourier series",
        "partial derivative", "double integral", "triple integral",
        "line integral", "surface integral", "vector calculus",
        "gradient", "divergence", "curl", "laplace transform",
        "fourier transform", "complex numbers", "imaginary numbers",
        "polar coordinates", "parametric equations", "conic sections",
        "ellipse equation", "parabola equation", "hyperbola equation",
        "circle equation", "law of sines", "law of cosines", "unit circle",
        "radians to degrees", "degrees to radians", "binomial theorem",
        "pascal's triangle", "combinatorics", "discrete math",
        "boolean algebra", "truth table", "set operations",
        "union and intersection", "matrix multiplication", "matrix inverse",
        "determinant", "transpose matrix", "eigen decomposition",
        "singular value decomposition", "linear transformation",
        "vector space", "basis vectors", "dot product", "cross product",
        "magnitude of vector", "unit vector", "z score", "chi square test",
        "anova test", "t test", "correlation coefficient", "covariance",
        "bayes theorem", "conditional probability", "expected value",
        "random variable", "probability distribution",
        "poisson distribution", "binomial distribution",
        "central limit theorem", "sample size calculation",
        "margin of error", "interest calculation", "loan emi calculation",
        "tax calculation", "gst calculation", "discount calculation",
        "markup calculation", "break even point", "npv calculation",
        "irr calculation", "amortization schedule"
    ],
    "creative": [
        "story", "poem", "write a", "imagine", "creative", "fiction",
        "character", "plot", "screenplay", "dialogue", "lyrics", "song",
        "metaphor", "describe", "paint", "design", "invent", "dream",
        "fantasy", "narrative", "short story", "novel", "chapter",
        "plot twist", "world building", "worldbuilding", "fictional",
        "fairy tale", "fable", "myth", "legend", "haiku", "sonnet",
        "limerick", "rap", "verse", "stanza", "rhyme", "rhyming",
        "freestyle", "script writing", "scene", "monologue", "prose",
        "write me a", "compose a", "draft a story", "draft a poem",
        "imaginative", "surreal", "sci-fi story", "fantasy story",
        "romance story", "horror story", "mystery story", "thriller",
        "plot idea", "backstory", "character development",
        "character profile", "setting description", "descriptive writing",
        "vivid description", "brainstorm story ideas", "creative writing",
        "concept art", "design concept", "name ideas", "name generator",
        "slogan", "tagline", "branding idea", "jingle", "caption",
        "creative caption", "creative title", "alternate ending",
        "what if story", "parody", "satire", "joke", "pun", "riddle",
        "anecdote", "bedtime story", "children's story", "lullaby",
        "make up a story", "make up a character", "create a character",
        "create a world", "invent a creature", "invent a language",
        "invent a gadget", "fictional universe", "fictional world",
        "alternate universe", "alternate history", "dystopian story",
        "utopian story", "post apocalyptic story", "cyberpunk story",
        "steampunk story", "medieval fantasy", "epic fantasy",
        "adventure story", "superhero story", "villain backstory",
        "hero's journey", "origin story", "love story", "tragic story",
        "comedy sketch", "stand up bit", "funny story", "ghost story",
        "supernatural story", "paranormal story", "detective story",
        "spy story", "war story", "western story", "coming of age story",
        "moral of the story", "story prompt", "writing prompt",
        "plot generator", "character name generator", "fantasy name",
        "dragon", "wizard", "magic spell", "magical creature",
        "epic poem", "ode to", "elegy", "ballad", "free verse",
        "concrete poetry", "acrostic poem", "love poem", "sad poem",
        "happy poem", "nature poem", "poem about", "song about",
        "rap verse", "rap battle", "diss track", "love letter",
        "breakup letter", "speech for", "toast for", "wedding speech",
        "best man speech", "eulogy", "vows", "wedding vows",
        "birthday message", "greeting card message", "quote about",
        "inspirational quote", "motivational quote",
        "caption for instagram", "caption for photo", "bio for instagram",
        "twitter bio", "username ideas", "team name ideas",
        "company name ideas", "product name ideas", "brand name",
        "mascot idea", "logo idea", "color palette idea", "mood board idea",
        "art style", "drawing idea", "sketch idea", "painting idea",
        "mural idea", "comic strip idea", "manga plot", "anime plot",
        "video game plot", "game character idea", "quest idea",
        "riddle me this", "tongue twister", "knock knock joke"
    ],
    "translate": [
        "translate", "in hindi", "in french", "in spanish", "in arabic",
        "in german", "in japanese", "in chinese", "convert to",
        "change language", "say in", "meaning in", "hindi mein",
        "english mein", "in english", "in urdu", "in punjabi",
        "in bengali", "in tamil", "in telugu", "in marathi", "in gujarati",
        "in kannada", "in malayalam", "in russian", "in italian",
        "in portuguese", "in korean", "in turkish", "in dutch", "in greek",
        "in hebrew", "in persian", "in swahili", "in vietnamese",
        "in thai", "in indonesian", "translate to", "translate into",
        "translation of", "how to say", "what does it mean in",
        "kaise kahein", "matlab", "ka matlab", "ka arth", "anuvad",
        "tarjuma", "language conversion", "language translation",
        "localize", "localization", "transliterate", "transliteration",
        "romanize", "native language", "mother tongue", "bilingual",
        "multilingual", "switch language to", "write in hindi",
        "write in spanish", "reply in french", "respond in german",
        "answer in japanese", "convert language", "language to language",
        "from english to", "to english from", "in my language",
        "apna matlab", "iska matlab", "iska arth kya hai",
        "english me bolo", "english me samjhao", "hindi me samjhao",
        "hindi me bolo", "translate this sentence", "translate this paragraph",
        "translate this word", "translate this phrase",
        "word for word translation", "literal translation",
        "free translation", "machine translation", "subtitle translation",
        "dub script", "transcreation", "interpret this",
        "interpretation of", "what's this in", "say this in", "phrase in",
        "word in", "term in", "vocabulary in", "how do you say",
        "how would you say", "equivalent word in",
        "synonym in another language", "translate document",
        "translate paragraph", "translate email", "translate message",
        "translate caption", "translate lyrics meaning",
        "translate poem meaning", "regional language",
        "dialect translation", "formal translation", "informal translation",
        "slang translation", "translate idiom", "idiom meaning in",
        "proverb meaning in", "convert script", "devanagari to roman",
        "roman to devanagari", "in nepali", "in sinhala", "in burmese",
        "in filipino", "in tagalog", "in swedish", "in norwegian",
        "in danish", "in finnish", "in polish", "in czech", "in romanian",
        "in ukrainian", "in farsi"
    ],
    "system": [
        "open", "close", "launch", "run", "execute", "file", "folder",
        "delete", "move", "copy", "rename", "search file", "find file",
        "terminal", "command", "shutdown", "restart", "volume",
        "brightness", "screenshot", "clipboard", "lock screen", "unlock",
        "sleep mode", "wake up", "reboot", "power off", "power on",
        "mute", "unmute", "increase volume", "decrease volume",
        "turn on wifi", "turn off wifi", "bluetooth", "airplane mode",
        "battery status", "check battery", "disk space", "storage",
        "system info", "task manager", "process", "kill process",
        "install", "uninstall", "update software", "system update",
        "settings", "control panel", "registry", "env variable",
        "environment variable", "path variable", "permissions", "chmod",
        "sudo", "admin rights", "root access", "create folder",
        "new folder", "new file", "zip file", "unzip", "extract",
        "compress", "backup", "restore", "format drive", "partition",
        "mount drive", "eject", "network settings", "ip address", "dns",
        "vpn", "firewall", "antivirus", "scan virus", "clean cache",
        "clear cache", "temp files", "startup programs", "task scheduler",
        "alarm", "set reminder", "set timer", "notification settings",
        "default app", "default browser", "keyboard shortcut", "hotkey",
        "dark mode", "light mode", "theme change", "wallpaper",
        "lock file", "unlock file", "hide file", "show hidden files",
        "file properties", "file size", "compress file",
        "send to recycle bin", "empty trash", "recycle bin",
        "system tray", "taskbar", "desktop icon", "open app",
        "open application", "open browser", "open chrome", "open settings",
        "close app", "close window", "close tab", "minimize window",
        "maximize window", "switch window", "switch tab", "switch app",
        "alt tab", "force quit", "force close", "end task",
        "terminate process", "process id", "pid", "memory usage",
        "cpu usage", "ram usage", "gpu usage", "device manager",
        "driver update", "update driver", "install driver",
        "system requirements", "check specs", "system specs",
        "hardware info", "software info", "os version", "windows version",
        "mac version", "linux distro", "ubuntu command", "macos command",
        "windows command", "cmd command", "powershell command",
        "registry edit", "regedit", "bios settings", "boot menu",
        "safe mode", "recovery mode", "factory reset", "reset settings",
        "clear data", "clear app data", "app permissions",
        "location permission", "camera permission", "microphone permission",
        "notification permission", "storage permission", "manage apps",
        "app manager", "uninstall app", "disable app", "enable app",
        "background apps", "battery saver", "power saving mode",
        "performance mode", "do not disturb", "silent mode", "ringtone",
        "alarm sound", "screen timeout", "auto lock", "auto rotate",
        "screen orientation", "display settings", "resolution settings",
        "refresh rate", "color profile", "night mode", "blue light filter",
        "accessibility settings", "voice control", "voice assistant",
        "siri command", "alexa command", "google assistant command",
        "automation rule", "shortcut automation", "macro recording",
        "script automation", "scheduled task", "startup app", "login item",
        "user account", "switch user", "create user account",
        "delete user account", "change password", "reset password",
        "two factor authentication", "biometric login", "fingerprint setup",
        "face id setup"
    ],
}

CATEGORY_WEIGHTS = {
    "fast": 1, "smart": 1, "heavy": 2,
    "code": 2, "math": 2, "creative": 2,
    "translate": 3, "system": 2,
}

TIE_BREAK_ORDER = [
    "translate", "math", "code", "creative",
    "system", "heavy", "smart", "fast"
]

_COMPILED_KEYWORDS = {
    category: re.compile(
        r'\b(?:' + '|'.join(
            re.escape(kw) for kw in sorted(keywords, key=len, reverse=True)
        ) + r')\b'
    )
    for category, keywords in KEYWORD_MAP.items()
}

_MATH_EXPRESSION = re.compile(r'\d+\s*[\+\-\*\/\^]\s*\d+')


# ─────────────────────────────────────────────────────────────────────────────
#  GEMINI UTILS
# ─────────────────────────────────────────────────────────────────────────────

def list_gemini_models() -> list:
    """List all Gemini models your API key can actually access."""
    try:
        models = [m.name for m in gemini_native.models.list()]
        return models
    except Exception as e:
        print(f"[GEMINI] Could not list models: {e}")
        return []


def call_gemini_native(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    """Call Gemini using native SDK — more reliable than OpenAI-compat endpoint."""
    response = gemini_native.models.generate_content(
        model=model_name,
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
    )
    return response.text


def list_groq_models() -> list:
    """List all live Groq models on your account."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
            timeout=5
        )
        return [m["id"] for m in r.json().get("data", [])]
    except Exception as e:
        print(f"[GROQ] Could not list models: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_response(raw: str) -> tuple:
    """Extract code and response blocks. Strips <think> leaks from some models."""
    raw  = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    code = None

    if "<code>" in raw and "</code>" in raw:
        extracted = raw.split("<code>")[1].split("</code>")[0].strip()
        if extracted and extracted.lower() not in ("none", ""):
            code = extracted

    if "<response>" in raw and "</response>" in raw:
        text = raw.split("<response>")[1].split("</response>")[0].strip()
    else:
        text = re.sub(r"</?code>|</?response>|</?think>", "", raw).strip()

    text = re.sub(r"</?code>|</?response>", "", text).strip()
    return code, text


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PICKER
# ─────────────────────────────────────────────────────────────────────────────

def pickmodel(prompt: str, override: str | None = None) -> str:
    if override and override in Model:
        return override

    prompt_lower = prompt.lower()

    if _MATH_EXPRESSION.search(prompt_lower):
        return "math"

    scores = {
        category: len(pattern.findall(prompt_lower)) * CATEGORY_WEIGHTS[category]
        for category, pattern in _COMPILED_KEYWORDS.items()
    }

    top_score = max(scores.values())
    if top_score == 0:
        return "smart"

    for category in TIE_BREAK_ORDER:
        if scores[category] == top_score:
            return category

    return "smart"


def client_model(mode: str) -> OpenAI:
    # everything uses groq — gemini kept only as optional native fallback
    return groq_client


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION
# ─────────────────────────────────────────────────────────────────────────────

def load_session(last_n: int = 10) -> list:
    if not os.path.exists(SESSION_FILE):
        return []
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            session = json.load(f)
        if not isinstance(session, list):
            return []
    except (json.JSONDecodeError, OSError):
        return []
    messages = []
    for turn in session[-last_n:]:
        messages.append({"role": "user",      "content": turn.get("prompt", "")})
        messages.append({"role": "assistant",  "content": turn.get("response", "")})
    return messages


def save_to_session(prompt: str, response: dict) -> None:
    session = []
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                session = json.load(f)
            if not isinstance(session, list):
                session = []
        except (json.JSONDecodeError, OSError):
            session = []

    session.append({
        "timestamp": datetime.now().isoformat(),
        "prompt":    prompt,
        "mode":      response["mode"],
        "code":      response["code"],
        "response":  response["response"],
    })

    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=4, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
#  TIMING
# ─────────────────────────────────────────────────────────────────────────────

def save_timing(prompt: str, stage_times: dict, total: float):
    log = []
    if os.path.exists(TIMING_FILE):
        try:
            with open(TIMING_FILE, "r") as f:
                log = json.load(f)
        except (json.JSONDecodeError, OSError):
            log = []
    log.append({
        "timestamp":     datetime.now().isoformat(),
        "prompt":        prompt[:120],
        "stages": {
            "stt_seconds": round(stage_times.get("stt", 0), 3),
            "llm_seconds": round(stage_times.get("llm", 0), 3),
            "tts_seconds": round(stage_times.get("tts", 0), 3),
        },
        "total_seconds": round(total, 3),
    })
    with open(TIMING_FILE, "w") as f:
        json.dump(log, f, indent=4)
    print(f"[TIMER] LLM={stage_times.get('llm',0):.2f}s  TOTAL={total:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def txt_sending(prompt: str, override: str | None = None) -> dict:
    mode       = pickmodel(prompt, override)
    model_name = Model.get(mode)
    history    = load_session(last_n=10)

    if not model_name:
        model_name = "openai/gpt-oss-120b"

    print(f"[MODEL] {mode} → {model_name}")

    try:
        # ── primary: groq ──────────────────────────────────────────────────────
        raw = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history,
                {"role": "user",   "content": f"[mode: {mode}]\n{prompt}"},
            ],
        ).choices[0].message.content or ""

    except Exception as e:
        err = str(e).lower()
        print(f"[GROQ ERROR] {e}")

        if any(x in err for x in ["over capacity", "overloaded", "429", "rate limit"]):
            # ── backup groq model ──────────────────────────────────────────────
            print(f"[FALLBACK] Switching to backup Groq model...")
            try:
                raw = groq_client.chat.completions.create(
                    model=Model["code_backup"],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *history,
                        {"role": "user",   "content": f"[mode: {mode}]\n{prompt}"},
                    ],
                ).choices[0].message.content or ""
            except Exception:
                # ── last resort: native gemini SDK ─────────────────────────────
                print(f"[FALLBACK] Trying native Gemini SDK...")
                try:
                    raw = call_gemini_native(f"[mode: {mode}]\n{prompt}")
                except Exception as e3:
                    result = {
                        "mode": mode, "code": None,
                        "response": f"All models failed. Last error: {e3}",
                    }
                    save_to_session(prompt, result)
                    return result
        else:
            # ── non-capacity error: try native gemini immediately ──────────────
            print(f"[FALLBACK] Trying native Gemini SDK...")
            try:
                raw = call_gemini_native(f"[mode: {mode}]\n{prompt}")
            except Exception as e2:
                result = {
                    "mode": mode, "code": None,
                    "response": f"Couldn't reach any model. Error: {e2}",
                }
                save_to_session(prompt, result)
                return result

    code, response = parse_response(raw)
    result = {"mode": mode, "code": code, "response": response}
    save_to_session(prompt, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── debug flag: python model.py --list ────────────────────────────────────
    if "--list" in sys.argv:
        print("\n[GROQ] Available models:")
        for m in list_groq_models():
            print(f"  {m}")
        print("\n[GEMINI] Available models:")
        for m in list_gemini_models():
            print(f"  {m}")
        sys.exit(0)

    print("Alexandria ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        start  = time.perf_counter()
        result = txt_sending(user_input)
        total  = time.perf_counter() - start

        print(f"\n[{result['mode']}]")
        if result["code"]:
            print("\n--- code ---")
            print(result["code"])
        print("\n--- response ---")
        print(result["response"])
        print()

        save_timing(user_input, {"llm": total}, total)
        save_timing(user_input, {"llm": total}, total)