"""One LLM wrapper with four run modes plus a $0 extractive fallback.

    mode="local"        -> local HF model (Qwen) or an Ollama server        ($0)
    mode="free"         -> a free-tier hosted API (Groq / OpenRouter)       ($0)
    mode="paid"         -> the $5 credit on an OpenAI-compatible gateway    (tracked)
    mode="hosted_auto"  -> auto-detect whichever hosted key is present in the
                            environment (.env), trying config.HOSTED_PROVIDER_ORDER
    mode="extractive"   -> no model at all; generate() returns None so callers
                           fall back to their deterministic evidence rendering.

Every hosted call goes through BudgetTracker, which tracks *attempted*, successful
and failed calls, refuses to spend past the $5 cap on billed providers, and
distinguishes tracked cost (counts against the cap) from estimated list cost
(always reported, even for a free-tier provider that costs $0) so Phase 4 can
report accurate API accounting.

Hosted calls retry transient failures (429/5xx) and fall back from the normal
"environment proxy" network path to a direct, no-proxy path -- some Windows/VPN
setups have an environment proxy that hangs while direct works fine.
"""
from __future__ import annotations

import json
import logging
import time

from .. import config

log = logging.getLogger("digikala.llm")

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class BudgetTracker:
    """Tracks hosted-API attempts/successes/failures, tokens, and $ spend."""

    def __init__(self, cap_usd: float = config.BUDGET_USD, log_path=config.BUDGET_LOG):
        self.cap = cap_usd
        self.log_path = log_path
        self.attempted_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self.spent = 0.0
        self.estimated_list_cost = 0.0

    @property
    def calls(self) -> int:                            # back-compat alias
        return self.successful_calls

    def can_spend(self) -> bool:
        return self.spent < self.cap

    def _write(self, row: dict):
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def record_attempt(self, provider: str, model: str):
        self.attempted_calls += 1
        self._write({"t": time.time(), "event": "attempt", "provider": provider, "model": model})

    def record_failure(self, provider: str, model: str, status=None, error: str = ""):
        self.failed_calls += 1
        self._write({"t": time.time(), "event": "failure", "provider": provider, "model": model,
                     "status": status, "error": str(error)[:300]})

    def record_success(self, in_tok: int, out_tok: int, *, provider: str, model: str,
                        price_per_m=(0.0, 0.0), billed: bool = False) -> float:
        self.successful_calls += 1
        self.in_tokens += int(in_tok or 0)
        self.out_tokens += int(out_tok or 0)
        estimated = (int(in_tok or 0) * float(price_per_m[0])
                    + int(out_tok or 0) * float(price_per_m[1])) / 1e6
        actual = estimated if billed else 0.0
        self.estimated_list_cost += estimated
        self.spent += actual
        self._write({"t": time.time(), "event": "success", "provider": provider, "model": model,
                     "in": int(in_tok or 0), "out": int(out_tok or 0),
                     "estimated_list_cost": round(estimated, 6), "tracked_cost": round(actual, 6),
                     "billed": bool(billed)})
        return actual

    def summary(self) -> dict:
        resolved = self.successful_calls + self.failed_calls
        return {"api_attempts": self.attempted_calls, "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls, "resolved_attempts": resolved,
                "unresolved_attempts": max(0, self.attempted_calls - resolved),
                "input_tokens": self.in_tokens, "output_tokens": self.out_tokens,
                "total_cost_usd": round(self.spent, 6),
                "estimated_list_cost_usd": round(self.estimated_list_cost, 6),
                "budget_usd": self.cap, "remaining_usd": round(self.cap - self.spent, 6)}


class LLM:
    """Backend-agnostic chat model. Resolves its provider from `mode` + config."""

    def __init__(self, mode: str | None = None, budget: BudgetTracker | None = None,
                 temperature: float = config.LLM_TEMPERATURE,
                 max_tokens: int = config.LLM_MAX_NEW_TOKENS):
        self.mode = mode or config.RUN_MODE
        self.budget = budget or BudgetTracker()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_cost_usd = 0.0
        self._cache: dict = {}
        self._hf = None
        self.hosted_disabled_reason = None
        self.preferred_network_path = None
        self.provider = self._resolve_provider()

    def _provider_with_key(self, provider_name: str) -> dict:
        import os
        p = dict(config.PROVIDERS[provider_name])
        p["provider_name"] = provider_name
        p["api_key"] = os.environ.get(p["api_key_env"], "")
        return p

    def _resolve_provider(self) -> dict | None:
        if self.mode == "hosted_auto":
            for name in config.HOSTED_PROVIDER_ORDER:
                p = self._provider_with_key(name)
                if p["api_key"]:
                    p["auto_selected"] = True
                    return p
            return {"provider_name": None, "model": None, "base_url": None, "api_key": "",
                    "auto_selected": False, "price_per_m": (0.0, 0.0), "billed": False}
        if self.mode == "free":
            return self._provider_with_key(config.FREE_PROVIDER)
        if self.mode == "paid":
            return self._provider_with_key("paid")
        return None

    @property
    def backend(self) -> str:
        """Human label of what will actually run (for logging/eval)."""
        if self.mode == "local":
            return config.LOCAL_BACKEND
        if self.mode in ("hosted_auto", "free", "paid") and self.provider:
            return f"{self.mode}:{self.provider.get('model')}"
        return self.mode

    def available(self) -> bool:
        if self.mode == "extractive":
            return False
        if self.mode in ("hosted_auto", "free", "paid"):
            return bool(self.provider and self.provider.get("api_key"))
        return True

    # ---- diagnostics (no key ever printed) -------------------------------
    def diagnose(self) -> dict:
        """Check the hosted provider without exposing the API key: try listing
        models, and if that's inconclusive, run one tiny, fully-accounted Chat
        Completions probe. Used before a real run so a dead key / wrong model /
        network issue is caught with a bounded number of requests."""
        if self.mode not in ("hosted_auto", "free", "paid"):
            return {"mode": self.mode, "hosted": False, "available": self.available()}
        p = self.provider or {}
        result = {"mode": self.mode, "provider": p.get("provider_name"), "model": p.get("model"),
                  "key_present": bool(p.get("api_key")), "models_status": None,
                  "model_listed": None, "chat_probe_attempted": False,
                  "chat_probe_success": False, "network_path": None, "error": None}
        if not p.get("api_key"):
            result["error"] = "No recognized hosted API key found in the environment (.env)."
            return result
        import requests
        for path_name, trust_env in (("environment", True), ("direct_no_env_proxy", False)):
            session = requests.Session()
            session.trust_env = trust_env
            try:
                r = session.get(f"{p['base_url']}/models",
                                headers={"Authorization": f"Bearer {p['api_key']}"},
                                timeout=(config.API_CONNECT_TIMEOUT_S, 30))
                result["models_status"] = int(r.status_code)
                result["network_path"] = path_name
                if r.ok:
                    ids = {x.get("id") for x in r.json().get("data", []) if isinstance(x, dict)}
                    result["model_listed"] = p.get("model") in ids
                    self.preferred_network_path = path_name
                    if result["model_listed"]:
                        return result
                    result["error"] = "Configured model not in the provider's model list."
                    break
                result["error"] = r.text[:300]
                if r.status_code == 401:
                    return result
            except requests.RequestException as e:
                result["error"] = str(e)[:300]
            finally:
                session.close()
        # bounded tiny chat probe (fully accounted for in BudgetTracker)
        result["chat_probe_attempted"] = True
        text = self.generate("You are a diagnostic probe.", "Reply with exactly: OK")
        result["chat_probe_success"] = bool(text)
        if text:
            result["error"] = None
        return result

    # ---- generation -------------------------------------------------------
    def generate(self, system: str, user: str) -> str | None:
        """Return the model's reply, or None to signal the extractive fallback."""
        self.last_cost_usd = 0.0
        if not self.available() or self.hosted_disabled_reason:
            return None
        key = (self.mode, system, user)
        if key in self._cache:
            return self._cache[key]
        try:
            if self.mode == "local" and config.LOCAL_BACKEND == "ollama":
                text = self._ollama(system, user)
            elif self.mode == "local":
                text = self._hf_generate(system, user)
            else:
                text = self._openai_compatible(system, user)
        except Exception as e:                       # never crash the pipeline on an LLM error
            log.warning("LLM error (%s), falling back to extractive: %s", self.backend, str(e)[:200])
            return None
        self._cache[key] = text
        return text

    # -- local transformers --
    def _load_hf(self):
        if self._hf is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            log.info("loading %s", config.HF_LLM_MODEL)
            tok = AutoTokenizer.from_pretrained(config.HF_LLM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                config.HF_LLM_MODEL, dtype=torch.float16,
                device_map="cuda" if torch.cuda.is_available() else "cpu")
            self._hf = (tok, model)
        return self._hf

    def _hf_generate(self, system: str, user: str) -> str:
        import torch
        tok, model = self._load_hf()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False,
                                 temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _ollama(self, system: str, user: str) -> str:
        import requests
        r = requests.post(f"{config.OLLAMA_BASE_URL}/api/chat", timeout=300,
                          json={"model": config.OLLAMA_MODEL, "stream": False,
                                "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    # -- hosted OpenAI-compatible (free / paid / hosted_auto) --
    def _openai_compatible(self, system: str, user: str) -> str:
        import requests
        p = self.provider
        provider_name = p.get("provider_name", "")
        chargeable = bool(p.get("billed", False))
        if chargeable and not self.budget.can_spend():
            raise RuntimeError("budget cap reached; refusing a chargeable API call")

        payload = {"model": p["model"], "temperature": self.temperature,
                  "max_tokens": self.max_tokens,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]}

        last_error: Exception | None = None
        max_rounds = max(1, int(config.API_MAX_ATTEMPTS))
        for round_idx in range(1, max_rounds + 1):
            if self.preferred_network_path == "direct_no_env_proxy":
                paths = [("direct_no_env_proxy", False)]
            elif self.preferred_network_path == "environment":
                paths = [("environment", True)]
            else:
                paths = [("environment", True), ("direct_no_env_proxy", False)]

            for path_name, trust_env in paths:
                if chargeable and not self.budget.can_spend():
                    raise RuntimeError("budget cap reached during retry loop")
                session = requests.Session()
                session.trust_env = trust_env
                self.budget.record_attempt(provider_name, p["model"])
                try:
                    r = session.post(f"{p['base_url']}/chat/completions",
                                     timeout=(config.API_CONNECT_TIMEOUT_S, config.API_READ_TIMEOUT_S),
                                     headers={"Authorization": f"Bearer {p['api_key']}",
                                              "Content-Type": "application/json"},
                                     json=payload)
                except requests.RequestException as e:
                    self.budget.record_failure(provider_name, p["model"], status=None,
                                               error=f"{path_name}: {str(e)[:250]}")
                    last_error = RuntimeError(f"network error via {path_name}: {str(e)[:250]}")
                    session.close()
                    continue                          # try the next network path immediately
                finally:
                    session.close()

                if r.ok:
                    data = r.json()
                    usage = data.get("usage", {})
                    self.preferred_network_path = path_name
                    self.last_cost_usd = self.budget.record_success(
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                        provider=provider_name, model=p["model"],
                        price_per_m=p.get("price_per_m", config.PAID_PRICE_PER_M),
                        billed=chargeable)
                    return str(data["choices"][0]["message"].get("content", "") or "").strip()

                self.budget.record_failure(provider_name, p["model"], status=int(r.status_code),
                                           error=f"{path_name}: {r.text[:250]}")
                last_error = RuntimeError(f"HTTP {r.status_code} via {path_name}: {r.text[:250]}")
                if r.status_code in (400, 401, 404, 422):
                    raise last_error               # not retryable: bad request/auth/model
                if r.status_code == 403 and path_name == "environment":
                    continue                        # 403 can be network/IP-dependent; try direct
                if r.status_code == 403 and path_name == "direct_no_env_proxy":
                    self.hosted_disabled_reason = "HTTP 403 on the direct path; using extractive fallback."
                    raise last_error
                if r.status_code not in _RETRYABLE_STATUSES:
                    raise last_error
                break                                # retryable: end this round, retry next round

            if round_idx < max_rounds:
                time.sleep(float(config.API_RETRY_BASE_S) * round_idx)

        raise last_error or RuntimeError("hosted API call failed")


def judge_llm(budget: BudgetTracker | None = None) -> "LLM":
    """The model Phase 4 uses as LLM-as-judge, chosen by config.JUDGE_MODE."""
    mode = "extractive" if config.JUDGE_MODE == "none" else config.JUDGE_MODE
    return LLM(mode=mode, budget=budget)
