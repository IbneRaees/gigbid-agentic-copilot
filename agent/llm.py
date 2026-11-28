"""Tiered Model Router supporting `open`, `paid`, and deterministic `offline_demo` profiles.

Reads `MODEL_PROFILE` and node-level configurations from `agent/config.yaml`,
enforces step/token/USD budgets from graph state, supports Pydantic v2 structured
outputs, and falls back deterministically to offline heuristics whenever external
LLM providers or local Ollama servers are unreachable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "agent" / "config.yaml"

TModel = TypeVar("TModel", bound=BaseModel)


class BudgetExceededError(RuntimeError):
    """Raised when a LangGraph run exceeds its configured step, token, or USD budget."""


@lru_cache(maxsize=4)
def load_agent_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and cache `agent/config.yaml`."""
    path = Path(config_path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def clear_config_cache() -> None:
    """Clear cached YAML config (useful in tests)."""
    load_agent_config.cache_clear()


def resolve_model_profile(explicit_profile: str | None = None) -> str:
    """Resolve active model profile (`offline_demo`, `open`, or `paid`)."""
    raw = (explicit_profile or os.getenv("MODEL_PROFILE", "open")).strip().lower()
    if raw in {"offline", "offline_demo", "demo", "mock"}:
        return "offline_demo"
    if raw in {"paid", "frontier"}:
        return "paid"
    return "open"


def _estimate_tokens(text: str) -> int:
    """Approximate token count (~4 chars/token) for offline accounting & budget checks."""
    return max(1, len(text) // 4)


def _estimate_cost_usd(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Estimate USD inference cost ($0.0 for local Ollama and offline_demo)."""
    if provider in {"offline", "ollama", "vllm"}:
        return 0.0
    if "gpt-4o-mini" in model or "8b" in model or "7b" in model:
        return round((prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000.0, 6)
    if "claude" in model or "gpt-4o" in model:
        return round((prompt_tokens * 2.50 + completion_tokens * 10.0) / 1_000_000.0, 6)
    return round((prompt_tokens * 0.50 + completion_tokens * 1.50) / 1_000_000.0, 6)


def _ollama_reachable(base_url: str, timeout_s: float = 0.8) -> bool:
    """Quick health probe to check if local Ollama is running before attempting inference."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


class ModelRouter:
    """Routes LLM calls per node and profile with budget enforcement and offline fallback."""

    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        profile: str | None = None,
    ) -> None:
        self.config = load_agent_config(config_path)
        self.profile = resolve_model_profile(profile)
        self.budgets_cfg = self.config.get("budgets", {})
        self.nodes_cfg = self.config.get("nodes", {})
        self.profiles_cfg = self.config.get("model_profiles", {})

    def check_budget(self, state: dict[str, Any] | None) -> None:
        """Enforce step, token, and USD caps stored in graph state."""
        if not state:
            return
        budgets = state.get("budgets") or {}
        steps_used = int(budgets.get("steps_used", 0))
        max_steps = int(budgets.get("max_steps", self.budgets_cfg.get("max_steps", 24)))
        tokens_used = int(budgets.get("tokens_used", 0))
        max_tokens = int(budgets.get("max_tokens", self.budgets_cfg.get("max_tokens", 24000)))
        cost_used = float(budgets.get("estimated_cost_usd", 0.0))
        max_cost = float(
            budgets.get("max_budget_usd", self.budgets_cfg.get("max_daily_budget_usd", 5.0))
        )

        if steps_used >= max_steps:
            raise BudgetExceededError(f"Step budget exhausted ({steps_used} >= {max_steps}).")
        if tokens_used >= max_tokens:
            raise BudgetExceededError(f"Token budget exhausted ({tokens_used} >= {max_tokens}).")
        if cost_used >= max_cost:
            raise BudgetExceededError(
                f"USD budget exhausted (${cost_used:.4f} >= ${max_cost:.2f})."
            )

    def resolve_backend_for_node(self, node_name: str) -> tuple[str, str, float, int, int]:
        """Return `(provider, model_name, temperature, max_tokens, timeout_s)` for a node."""
        node_cfg = self.nodes_cfg.get(node_name, {})
        tier = str(node_cfg.get("tier", "fast"))
        temperature = float(node_cfg.get("temperature", 0.2))
        max_tokens = int(node_cfg.get("max_tokens", 1200))
        timeout_s = int(node_cfg.get("timeout_s", 30))

        if self.profile == "offline_demo":
            p_cfg = self.profiles_cfg.get("offline_demo", {})
            model = str(p_cfg.get(f"{tier}_model", f"gigbid-offline-{tier}-v1"))
            return "offline", model, temperature, max_tokens, timeout_s

        if self.profile == "paid":
            p_cfg = self.profiles_cfg.get("paid", {})
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            openai_key = os.getenv("OPENAI_API_KEY", "").strip()
            if tier == "reasoning" and anthropic_key:
                return (
                    "anthropic",
                    str(p_cfg.get("reasoning_model", "claude-3-5-sonnet-latest")),
                    temperature,
                    max_tokens,
                    timeout_s,
                )
            if openai_key:
                model_key = "openai_reasoning_model" if tier == "reasoning" else "fast_model"
                return (
                    "openai",
                    str(p_cfg.get(model_key, "gpt-4o-mini")),
                    temperature,
                    max_tokens,
                    timeout_s,
                )
            if anthropic_key:
                return (
                    "anthropic",
                    str(p_cfg.get("reasoning_model", "claude-3-5-sonnet-latest")),
                    temperature,
                    max_tokens,
                    timeout_s,
                )
            return "offline", f"gigbid-offline-{tier}-fallback", temperature, max_tokens, timeout_s

        # `open` profile: local Ollama -> OpenRouter -> Groq -> deterministic offline fallback
        p_cfg = self.profiles_cfg.get("open", {})
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        if os.getenv("GIGBID_DISABLE_LIVE_LLM", "").lower() not in {"1", "true", "yes"}:
            if ollama_url and _ollama_reachable(ollama_url):
                return (
                    "ollama",
                    str(
                        p_cfg.get(
                            f"{tier}_model", "qwen3:8b" if tier == "reasoning" else "qwen3:4b"
                        )
                    ),
                    temperature,
                    max_tokens,
                    timeout_s,
                )
            openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if openrouter_key:
                model = str(
                    p_cfg.get(
                        f"hosted_{tier}_model",
                        "qwen/qwen-2.5-72b-instruct"
                        if tier == "reasoning"
                        else "qwen/qwen-2.5-7b-instruct",
                    )
                )
                return "openrouter", model, temperature, max_tokens, timeout_s
            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            if groq_key:
                model = str(
                    p_cfg.get(
                        f"groq_{tier}_model",
                        "llama-3.3-70b-versatile"
                        if tier == "reasoning"
                        else "llama-3.1-8b-instant",
                    )
                )
                return "groq", model, temperature, max_tokens, timeout_s

        return "offline", f"gigbid-offline-{tier}-fallback", temperature, max_tokens, timeout_s

    def _build_chat_model(
        self,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
    ) -> Any:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )

        from langchain_openai import ChatOpenAI

        if provider == "openai":
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
        if provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
            return ChatOpenAI(
                model=model,
                base_url=base_url,
                api_key="ollama",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
        if provider == "openrouter":
            return ChatOpenAI(
                model=model,
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
        if provider == "groq":
            return ChatOpenAI(
                model=model,
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY", ""),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
            )
        raise ValueError(f"Unsupported live provider '{provider}'")

    async def invoke_structured(
        self,
        *,
        node_name: str,
        schema: type[TModel],
        system_prompt: str,
        user_prompt: str,
        fallback_fn: Callable[[], TModel],
        state: dict[str, Any] | None = None,
    ) -> tuple[TModel, dict[str, Any]]:
        """Invoke an LLM for Pydantic structured output, with deterministic offline fallback."""
        self.check_budget(state)
        provider, model, temperature, max_tokens, timeout_s = self.resolve_backend_for_node(
            node_name
        )
        prompt_tokens = _estimate_tokens(system_prompt + "\n" + user_prompt)

        if provider != "offline":
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                llm = self._build_chat_model(provider, model, temperature, max_tokens, timeout_s)
                structured_llm = llm.with_structured_output(schema)
                result = await structured_llm.ainvoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                )
                if isinstance(result, schema):
                    parsed = result
                else:
                    parsed = schema.model_validate(result)
                completion_tokens = _estimate_tokens(parsed.model_dump_json())
                cost = _estimate_cost_usd(provider, model, prompt_tokens, completion_tokens)
                usage = {
                    "provider": provider,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "estimated_cost_usd": cost,
                    "used_fallback": False,
                }
                return parsed, usage
            except Exception as exc:
                logger.warning(
                    "Live LLM call failed for node '%s' (%s/%s): %s. Falling back deterministically.",
                    node_name,
                    provider,
                    model,
                    exc,
                )

        parsed = fallback_fn()
        completion_tokens = _estimate_tokens(parsed.model_dump_json())
        prompt_digest = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:8]
        usage = {
            "provider": "offline",
            "model": f"{model}#{prompt_digest}",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost_usd": 0.0,
            "used_fallback": True,
        }
        return parsed, usage

    async def invoke_text(
        self,
        *,
        node_name: str,
        system_prompt: str,
        user_prompt: str,
        fallback_fn: Callable[[], str],
        state: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Invoke an LLM for free-form text output with deterministic offline fallback."""
        self.check_budget(state)
        provider, model, temperature, max_tokens, timeout_s = self.resolve_backend_for_node(
            node_name
        )
        prompt_tokens = _estimate_tokens(system_prompt + "\n" + user_prompt)

        if provider != "offline":
            try:
                from langchain_core.messages import HumanMessage, SystemMessage

                llm = self._build_chat_model(provider, model, temperature, max_tokens, timeout_s)
                resp = await llm.ainvoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                )
                text = str(getattr(resp, "content", resp)).strip()
                if text:
                    completion_tokens = _estimate_tokens(text)
                    cost = _estimate_cost_usd(provider, model, prompt_tokens, completion_tokens)
                    usage = {
                        "provider": provider,
                        "model": model,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "estimated_cost_usd": cost,
                        "used_fallback": False,
                    }
                    return text, usage
            except Exception as exc:
                logger.warning(
                    "Live LLM text call failed for node '%s' (%s/%s): %s. Using fallback.",
                    node_name,
                    provider,
                    model,
                    exc,
                )

        text = fallback_fn()
        completion_tokens = _estimate_tokens(text)
        usage = {
            "provider": "offline",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost_usd": 0.0,
            "used_fallback": True,
        }
        return text, usage


__all__ = [
    "BudgetExceededError",
    "ModelRouter",
    "clear_config_cache",
    "load_agent_config",
    "resolve_model_profile",
]
