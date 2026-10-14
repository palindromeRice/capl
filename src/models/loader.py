"""Model loader supporting 4-bit NF4 quantized inference, sequential offloading, and mock execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.memory import detect_hardware, flush_memory

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_id: str
    fallback_id: Optional[str] = None
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: str = "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0


class MockLLM:
    """Mock LLM for fast unit tests, smoke tests, and CPU execution without downloading full weights."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _mock_preference_extraction(self, prompt: str, prompt_low: str) -> str:
        """Synthesize mock preference agent structured JSON rules."""
        import json

        rules = []
        is_contextual = "context-aware" in prompt_low or "b4" in prompt_low or "context" in prompt_low
        is_static = "static" in prompt_low or "b1" in prompt_low

        # Determine target context from prompt
        ctx = "general"
        if "technical_engineering" in prompt_low or "locking" in prompt_low or "database" in prompt_low or "memory" in prompt_low:
            ctx = "technical_engineering"
        elif "executive_briefing" in prompt_low or "cost" in prompt_low or "board" in prompt_low or "strategic" in prompt_low:
            ctx = "executive_briefing"
        elif "educational_tutoring" in prompt_low or "binary search" in prompt_low:
            ctx = "educational_tutoring"
        elif "casual_chat" in prompt_low:
            ctx = "casual_chat"

        conflict = is_contextual and ctx in ["technical_engineering", "executive_briefing"]

        if ctx == "executive_briefing":
            rules.append({
                "rule_id": "rule_pref_1",
                "preference": "Under executive_briefing, provide comprehensive high-level context and strategic impact.",
                "target_aspect": "verbosity",
                "context_condition": ctx if is_contextual else "general",
                "supporting_evidence_ids": ["int_exec_01"],
                "confidence": 0.92,
                "conflict_detected": conflict,
                "conflict_details": "Opposing terse verbosity observed in technical engineering context.",
                "recency_weight": 1.0,
            })
            rules.append({
                "rule_id": "rule_pref_2",
                "preference": "Under executive_briefing, organize recommendations in strictly numbered actionable items.",
                "target_aspect": "structure",
                "context_condition": ctx if is_contextual else "general",
                "supporting_evidence_ids": ["int_exec_02"],
                "confidence": 0.90,
                "conflict_detected": conflict,
                "recency_weight": 1.0,
            })
            rules.append({
                "rule_id": "rule_pref_3",
                "preference": "Under executive_briefing, include formal executive greeting and professional sign-off.",
                "target_aspect": "salutation",
                "context_condition": ctx if is_contextual else "general",
                "supporting_evidence_ids": ["int_exec_03"],
                "confidence": 0.88,
                "conflict_detected": conflict,
                "recency_weight": 1.0,
            })
            rules.append({
                "rule_id": "rule_pref_4",
                "preference": "Under executive_briefing, avoid raw code; focus exclusively on architecture and business trade-offs.",
                "target_aspect": "code_style",
                "context_condition": ctx if is_contextual else "general",
                "supporting_evidence_ids": [],
                "confidence": 0.95,
                "conflict_detected": conflict,
                "recency_weight": 1.0,
            })
        else:
            # technical_engineering or general
            # Check for temporal drift towards elaborate
            is_drift_elaborate = "drift" in prompt_low or "provide thorough" in prompt_low
            verb_pref = "Provide thorough architectural explanations with extensive rationale." if is_drift_elaborate else "Keep responses concise and under 100 words with zero unnecessary preamble."
            rules.append({
                "rule_id": "rule_pref_1",
                "preference": verb_pref if is_static else f"Under {ctx}, {verb_pref}",
                "target_aspect": "verbosity",
                "context_condition": ctx if is_contextual else "general",
                "supporting_evidence_ids": ["int_tech_01"],
                "confidence": 0.89,
                "conflict_detected": conflict,
                "conflict_details": "Opposing elaborate verbosity observed in executive briefing context." if conflict else None,
                "recency_weight": 0.95,
            })
            rules.append({
                "rule_id": "rule_pref_2",
                "preference": "Format all explanations in crisp bullet points." if is_static else f"Under {ctx}, present all explanations in crisp bullet points.",
                "target_aspect": "structure",
                "context_condition": ctx if is_contextual else "general",
                "supporting_evidence_ids": ["int_tech_02"],
                "confidence": 0.86,
                "conflict_detected": conflict,
                "recency_weight": 0.95,
            })
            rules.append({
                "rule_id": "rule_pref_3",
                "preference": "Never use greetings like 'Hello' or sign-offs; jump immediately into the answer." if is_static else f"Under {ctx}, never use greetings like 'Hello' or sign-offs; jump immediately into the answer.",
                "target_aspect": "salutation",
                "context_condition": ctx if is_contextual else "general",
                "supporting_evidence_ids": ["int_tech_03"],
                "confidence": 0.91,
                "conflict_detected": conflict,
                "recency_weight": 1.0,
            })
            rules.append({
                "rule_id": "rule_pref_4",
                "preference": "Provide executable Python code implementations." if is_static else f"Under {ctx}, provide minimal code snippets without conversational preamble.",
                "target_aspect": "code_style",
                "context_condition": ctx if is_contextual else "general",
                "supporting_evidence_ids": ["int_tech_04"],
                "confidence": 0.85,
                "conflict_detected": conflict,
                "recency_weight": 1.0,
            })

        return json.dumps(rules, indent=2)

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs: Any) -> str:
        prompt_low = prompt.lower()

        # Handle preference extraction requests
        if "preference" in prompt_low and ("json" in prompt_low or "extract" in prompt_low or "rules" in prompt_low):
            return self._mock_preference_extraction(prompt, prompt_low)

        has_rules = ("user personalization preferences" in prompt_low) or ("behavioral profile" in prompt_low) or ("past examples" in prompt_low)

        # Baseline B0 (Zero-Shot unpersonalized): emits default population average
        if not has_rules:
            return "Handling this architecture requires balancing scalability against operational complexity. In production environments, isolating resources and maintaining bounded queues prevents cascading failures while preserving predictable performance."

        # Parse steering cues from the prompt
        has_terse = ("under 100 words" in prompt_low) or ("under 60 words" in prompt_low) or ("concise" in prompt_low)
        has_elaborate = ("comprehensive" in prompt_low) or ("thorough" in prompt_low) or ("architectural depth" in prompt_low)
        has_bullets = "bullet" in prompt_low
        has_numbered = "numbered" in prompt_low
        has_greeting = ("formal executive greeting" in prompt_low) or ("polite salutations" in prompt_low) or ("greetings,\n" in prompt_low)
        has_no_greeting = ("never include greetings" in prompt_low) or ("omit" in prompt_low) or ("no_greeting" in prompt_low)
        wants_code = ("executable python code" in prompt_low) or ("code_only" in prompt_low) or ("snippets" in prompt_low)
        forbid_code = ("avoid raw code" in prompt_low) or ("no_code" in prompt_low)

        response_lines = []
        if has_greeting and not has_no_greeting:
            response_lines.append("Greetings,\n")

        if has_bullets:
            if has_terse:
                response_lines.append("- Optimal throughput: Isolate connection pool.\n- Reliability: Enforce bounded queues.\n- Telemetry: Monitor p99 latency.")
            else:
                response_lines.append("- Primary consideration: Scalability bottlenecks emerge at high concurrency, demanding robust connection pooling and backpressure mitigation.\n- Reliability: Implementing circuit breakers alongside exponential backoff prevents cascading network partitions.\n- Observability: Instrumenting distributed tracing across all RPC boundaries ensures real-time anomaly detection and precise latency diagnostics.")
        elif has_numbered:
            if has_terse:
                response_lines.append("1. Audit allocations.\n2. Staged rollout.\n3. Align SLOs.")
            else:
                response_lines.append("1. Conduct thorough enterprise-wide infrastructure audit to isolate recurring latency spikes and cost drivers.\n2. Establish staged phased migration cohorts with automated canary validation and zero-downtime rollback guarantees.\n3. Formulate rigorous cross-functional SLO agreements to guarantee operational resilience and audit compliance across regions.")
        else:
            if has_terse:
                response_lines.append("Isolate shared state and bound worker pools to preserve throughput under peak load.")
            else:
                response_lines.append("Analyzing distributed systems thoroughly requires evaluating the interplay between distributed consistency, operational cost, and developer ergonomics. In high-throughput architectures, relying solely on naive retries introduces thundering herd hazards that can collapse downstream databases. By implementing adaptive concurrency limits and dynamic rate limiters, engineering teams can preserve tail latency guarantees while sustaining continuous availability during unforeseen traffic spikes.")

        if wants_code and not forbid_code:
            response_lines.append("\n```python\n# Implementation\ndef execute_task():\n    return True\n```")

        if has_greeting and not has_no_greeting:
            response_lines.append("\nSincerely,\nExecutive Engineering Lead")

        return "\n".join(response_lines)


class ModelManager:
    """Manages model loading, 4-bit NF4 quantization, sequential mode, and generation."""

    def __init__(
        self,
        pref_config: ModelConfig,
        main_config: ModelConfig,
        sequential_mode: bool = False,
        mock_mode: bool = False,
    ) -> None:
        self.pref_config = pref_config
        self.main_config = main_config
        self.sequential_mode = sequential_mode
        self.mock_mode = mock_mode

        self.pref_model: Any = None
        self.pref_tokenizer: Any = None
        self.main_model: Any = None
        self.main_tokenizer: Any = None

        self.hw_info = detect_hardware()
        logger.info(
            "Hardware detected: %s (%s, %.1f MB VRAM)",
            self.hw_info.device_type,
            self.hw_info.device_name,
            self.hw_info.total_memory_mb,
        )

        # Auto-enable sequential mode if GPU VRAM is tight (<16 GB)
        if self.hw_info.device_type == "cuda" and self.hw_info.total_memory_mb < 16000:
            logger.info("Total VRAM < 16GB. Automatically enabling sequential loading mode.")
            self.sequential_mode = True

    def _get_quantization_config(self, cfg: ModelConfig) -> Optional[Any]:
        if not (self.hw_info.device_type == "cuda" and cfg.load_in_4bit and self.hw_info.supports_4bit_bnb):
            return None

        try:
            from transformers import BitsAndBytesConfig

            compute_dtype = torch.bfloat16 if (cfg.compute_dtype == "bfloat16" and self.hw_info.supports_bf16) else torch.float16
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        except ImportError:
            logger.warning("BitsAndBytes not available. Loading in standard precision.")
            return None

    def load_preference_agent(self) -> None:
        """Load the lightweight preference agent model."""
        if self.pref_model is not None:
            return

        if self.sequential_mode:
            self.unload_main_generator()

        if self.mock_mode:
            logger.info("Initializing Mock Preference Agent: %s", self.pref_config.model_id)
            self.pref_model = MockLLM(self.pref_config.model_id)
            return

        bnb_config = self._get_quantization_config(self.pref_config)
        model_id = self.pref_config.model_id

        logger.info("Loading Preference Agent: %s (4-bit NF4=%s)", model_id, bnb_config is not None)
        try:
            self.pref_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.pref_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto" if self.hw_info.device_type == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.hw_info.supports_bf16 else torch.float32,
                trust_remote_code=True,
            )
        except Exception as e:
            if self.pref_config.fallback_id:
                logger.warning("Failed loading %s (%s). Attempting fallback %s...", model_id, e, self.pref_config.fallback_id)
                model_id = self.pref_config.fallback_id
                self.pref_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self.pref_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto" if self.hw_info.device_type == "cuda" else None,
                    trust_remote_code=True,
                )
            else:
                raise e

    def unload_preference_agent(self) -> None:
        """Unload preference agent and flush VRAM."""
        if self.pref_model is not None:
            logger.info("Unloading Preference Agent to free GPU memory...")
            del self.pref_model
            del self.pref_tokenizer
            self.pref_model = None
            self.pref_tokenizer = None
            flush_memory()

    def unload_main_generator(self) -> None:
        """Unload main generator and flush VRAM."""
        if self.main_model is not None:
            logger.info("Unloading Main Generator to free GPU memory...")
            del self.main_model
            del self.main_tokenizer
            self.main_model = None
            self.main_tokenizer = None
            flush_memory()

    def generate_with_pref_agent(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """Generate structured preference rules from the small quantized preference agent."""
        if self.pref_model is None:
            self.load_preference_agent()

        if self.mock_mode or isinstance(self.pref_model, MockLLM):
            return self.pref_model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        inputs = self.pref_tokenizer(prompt, return_tensors="pt")
        if self.hw_info.device_type == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.pref_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.pref_tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.pref_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def load_main_generator(self) -> None:
        """Load the frozen main foundation LLM."""
        if self.main_model is not None:
            return

        if self.sequential_mode:
            self.unload_preference_agent()

        if self.mock_mode:
            logger.info("Initializing Mock Main Generator: %s", self.main_config.model_id)
            self.main_model = MockLLM(self.main_config.model_id)
            return

        bnb_config = self._get_quantization_config(self.main_config)
        model_id = self.main_config.model_id

        logger.info("Loading Main Response Generator: %s (4-bit NF4=%s)", model_id, bnb_config is not None)
        try:
            self.main_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.main_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto" if self.hw_info.device_type == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.hw_info.supports_bf16 else torch.float32,
                trust_remote_code=True,
            )
        except Exception as e:
            if self.main_config.fallback_id:
                logger.warning("Failed loading %s (%s). Attempting fallback %s...", model_id, e, self.main_config.fallback_id)
                model_id = self.main_config.fallback_id
                self.main_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self.main_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto" if self.hw_info.device_type == "cuda" else None,
                    trust_remote_code=True,
                )
            else:
                raise e

    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate response from the main LLM."""
        if self.main_model is None:
            self.load_main_generator()

        if self.mock_mode or isinstance(self.main_model, MockLLM):
            return self.main_model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        inputs = self.main_tokenizer(prompt, return_tensors="pt")
        if self.hw_info.device_type == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.main_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.main_tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.main_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
