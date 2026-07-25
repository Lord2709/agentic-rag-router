"""
Shared configuration: environment loading and the Claude client used across
all tools (judge, reformulate, generate) and the orchestrator/router.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Model tiers: cheap/fast for router + judge (simple decisions),
# stronger for reformulation + generation (need more reasoning quality).
ROUTER_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-haiku-4-5-20251001"
REFORMULATE_MODEL = "claude-sonnet-5"
GENERATE_MODEL = "claude-sonnet-5"
