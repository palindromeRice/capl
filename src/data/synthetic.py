"""Synthetic dataset generator for context-dependent user preference learning."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AspectCategory(str, Enum):
    VERBOSITY = "verbosity"  # "terse" vs "elaborate"
    STRUCTURE = "structure"  # "bullet_points" vs "prose" vs "numbered_steps"
    TONE = "tone"  # "formal_business" vs "casual_direct" vs "academic"
    CODE_STYLE = "code_style"  # "code_only" vs "commented_code" vs "no_code"
    SALUTATION = "salutation"  # "no_greeting" vs "warm_greeting"


@dataclass
class GroundTruthRule:
    rule_id: str
    target_aspect: str
    context_condition: str  # e.g., "technical_engineering" or "executive_briefing"
    preference_statement: str
    rule_value: str = ""
    active_from_step: int = 0
    deprecated_at_step: Optional[int] = None
    confidence: float = 1.0


@dataclass
class Interaction:
    interaction_id: str
    user_id: str
    step: int
    context: str
    query: str
    user_response: str
    applied_rules: List[str] = field(default_factory=list)
    aspect_manifestations: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationSample:
    sample_id: str
    user_id: str
    context: str
    query: str
    target_ground_truth_rules: List[GroundTruthRule]
    deterministic_criteria: Dict[str, Any]  # Criteria for automated verification


@dataclass
class UserHistory:
    user_id: str
    persona_name: str
    all_rules: List[GroundTruthRule]
    history: List[Interaction]
    test_samples: List[EvaluationSample]


# Context Templates and Archetypes
CONTEXT_DOMAINS = [
    "technical_engineering",
    "executive_briefing",
    "educational_tutoring",
    "casual_chat",
]

PREFERENCE_CONFIGURATIONS = [
    {
        "technical_engineering": {
            AspectCategory.VERBOSITY: ("terse", "Responses must be under 100 words with zero fluff."),
            AspectCategory.STRUCTURE: ("bullet_points", "Present all explanations in crisp bullet points."),
            AspectCategory.TONE: ("casual_direct", "Use a direct, highly technical engineering tone."),
            AspectCategory.CODE_STYLE: ("code_only", "Provide minimal code snippets without conversational preamble."),
            AspectCategory.SALUTATION: ("no_greeting", "Never use greetings like 'Hello' or sign-offs."),
        },
        "executive_briefing": {
            AspectCategory.VERBOSITY: ("elaborate", "Provide comprehensive high-level context and strategic impact."),
            AspectCategory.STRUCTURE: ("numbered_steps", "Organize recommendations in strictly numbered actionable items."),
            AspectCategory.TONE: ("formal_business", "Maintain an elevated, formal executive business tone."),
            AspectCategory.CODE_STYLE: ("no_code", "Avoid raw code; focus exclusively on architecture and business trade-offs."),
            AspectCategory.SALUTATION: ("warm_greeting", "Begin with a formal salutation and conclude with a professional sign-off."),
        },
    },
    {
        "technical_engineering": {
            AspectCategory.VERBOSITY: ("elaborate", "Explain architectural trade-offs thoroughly with in-depth reasoning."),
            AspectCategory.STRUCTURE: ("prose", "Use continuous analytical paragraphs rather than fragmented bullets."),
            AspectCategory.TONE: ("academic", "Employ precise academic and systems nomenclature."),
            AspectCategory.CODE_STYLE: ("commented_code", "Include well-commented modular code implementations."),
            AspectCategory.SALUTATION: ("no_greeting", "Omit conversational pleasantries."),
        },
        "casual_chat": {
            AspectCategory.VERBOSITY: ("terse", "Keep responses short, witty, and under 50 words."),
            AspectCategory.STRUCTURE: ("bullet_points", "Use quick bullet points with emojis."),
            AspectCategory.TONE: ("casual_direct", "Very friendly, relaxed, and colloquial."),
            AspectCategory.CODE_STYLE: ("no_code", "Explain concepts using analogies instead of code."),
            AspectCategory.SALUTATION: ("warm_greeting", "Start with an enthusiastic greeting!"),
        },
    },
]

TASK_QUERIES = {
    "technical_engineering": [
        ("How should we handle database connection pooling in a high-throughput microservice?", "pooling"),
        ("What is the difference between optimistic and pessimistic locking in distributed systems?", "locking"),
        ("How can we optimize memory allocation in a latency-critical Python service?", "memory"),
        ("Explain the trade-offs between Raft and Paxos consensus algorithms.", "consensus"),
        ("How do we implement circuit breaking in our RPC clients?", "circuit_breaker"),
    ],
    "executive_briefing": [
        ("Summarize the infrastructure cost projections for our cloud migration next quarter.", "cost"),
        ("What are the primary operational risks of moving from monolithic to service architecture?", "risks"),
        ("Provide a strategic roadmap for adopting generative AI in customer workflows.", "roadmap"),
        ("How do we present our multi-region disaster recovery readiness to the board?", "disaster_recovery"),
        ("Assess vendor lock-in risks between AWS and GCP for our data lake.", "vendor_lockin"),
    ],
    "educational_tutoring": [
        ("Explain how binary search trees maintain their balance property.", "bst"),
        ("What is the intuition behind dynamic programming memoization?", "dp"),
        ("How do transformers utilize self-attention mechanisms?", "attention"),
        ("Why does gradient descent require learning rate scheduling?", "lr_schedule"),
        ("What does time complexity Big-O notation actually signify?", "big_o"),
    ],
    "casual_chat": [
        ("What's a good way to stay focused while debugging complex code all day?", "focus"),
        ("Can you recommend some interesting podcasts for software engineers?", "podcasts"),
        ("Why is naming variables consistently so hard in large projects?", "naming"),
        ("What are your favorite keyboard shortcuts for fast code navigation?", "shortcuts"),
        ("How do you take a quick break when stuck on an algorithmic bug?", "breaks"),
    ],
}


def _render_synthetic_response(context: str, topic: str, rules_dict: Dict[AspectCategory, tuple[str, str]]) -> str:
    """Deterministically render historical response matching given style rules."""
    verb_val, _ = rules_dict.get(AspectCategory.VERBOSITY, ("terse", ""))
    struct_val, _ = rules_dict.get(AspectCategory.STRUCTURE, ("bullet_points", ""))
    salut_val, _ = rules_dict.get(AspectCategory.SALUTATION, ("no_greeting", ""))
    tone_val, _ = rules_dict.get(AspectCategory.TONE, ("casual_direct", ""))
    code_val, _ = rules_dict.get(AspectCategory.CODE_STYLE, ("code_only", ""))

    parts = []
    if salut_val == "warm_greeting":
        parts.append("Greetings,\n" if "formal" in tone_val else "Hey there! 👋\n")

    if verb_val == "elaborate":
        if struct_val == "bullet_points":
            parts.append(f"Comprehensive analysis and architectural trade-offs for {topic}:\n- Primary consideration: Scalability bottlenecks emerge at high concurrency, demanding robust connection pooling and backpressure mitigation.\n- Reliability: Implementing circuit breakers alongside exponential backoff prevents cascading network partitions.\n- Observability: Instrumenting distributed tracing across all RPC boundaries ensures real-time anomaly detection and precise latency diagnostics.\n- Strategic evolution: Consider adopting asynchronous event-driven messaging to decouple producer and consumer throughput curves.")
        elif struct_val == "numbered_steps":
            parts.append(f"Comprehensive strategic roadmap and executive alignment for {topic}:\n1. Conduct thorough enterprise-wide infrastructure audit to isolate recurring latency spikes and cost drivers.\n2. Establish staged phased migration cohorts with automated canary validation and zero-downtime rollback guarantees.\n3. Formulate rigorous cross-functional SLO agreements to guarantee operational resilience and audit compliance across regions.\n4. Deliver executive milestone reviews to quantify cost avoidance and team velocity multipliers.")
        else:
            parts.append(f"Analyzing {topic} thoroughly requires evaluating the interplay between distributed consistency, operational cost, and developer ergonomics. In high-throughput architectures, relying solely on naive retries introduces thundering herd hazards that can collapse downstream databases. By implementing adaptive concurrency limits and dynamic rate limiters, engineering teams can preserve tail latency guarantees while sustaining continuous availability during unforeseen traffic spikes.")
    else:  # terse
        if struct_val == "bullet_points":
            parts.append(f"Key points regarding {topic}:\n- Optimal throughput: Isolate pool.\n- Reliability: Enforce bounded queues.\n- Telemetry: Track p99.")
        elif struct_val == "numbered_steps":
            parts.append(f"Strategic steps for {topic}:\n1. Audit allocations.\n2. Staged rollout.\n3. Align SLOs.")
        else:
            parts.append(f"Isolate shared state and bound worker pools to preserve throughput for {topic}.")

    if code_val == "code_only":
        parts.append(f"\n```python\n# {topic} pattern\ndef manage_{topic}():\n    return True\n```")
    elif code_val == "commented_code":
        parts.append(f"\n```python\n# Implementation for {topic}\n# Ensures defensive resource bounds\nclass {topic.capitalize()}Handler:\n    def execute(self):\n        return 'OK'\n```")

    if salut_val == "warm_greeting":
        parts.append("\nSincerely,\nEngineering Lead" if "formal" in tone_val else "\nCheers!")

    return "\n".join(parts)


def generate_synthetic_dataset(
    num_users: int = 50,
    interactions_per_user: int = 20,
    temporal_drift_ratio: float = 0.25,
    seed: int = 42,
) -> List[UserHistory]:
    """Generate a controlled synthetic dataset of users with multi-context conflicting preferences and temporal drift."""
    rng = random.Random(seed)
    users: List[UserHistory] = []

    for u_idx in range(num_users):
        user_id = f"user_{u_idx:03d}"
        persona_name = f"Persona_{u_idx}"
        
        # Assign contextual preferences
        config_template = PREFERENCE_CONFIGURATIONS[u_idx % len(PREFERENCE_CONFIGURATIONS)]
        has_temporal_drift = rng.random() < temporal_drift_ratio
        drift_step = interactions_per_user // 2 if has_temporal_drift else 9999

        ground_truth_rules: List[GroundTruthRule] = []
        rule_counter = 0

        # Build ground truth rules
        for ctx, aspects in config_template.items():
            for aspect_cat, (val, stmt) in aspects.items():
                rule_counter += 1
                r_id = f"rule_{user_id}_{rule_counter}"
                gt_rule = GroundTruthRule(
                    rule_id=r_id,
                    target_aspect=aspect_cat.value,
                    context_condition=ctx,
                    preference_statement=stmt,
                    rule_value=val,
                    active_from_step=0,
                    deprecated_at_step=drift_step if (has_temporal_drift and aspect_cat == AspectCategory.VERBOSITY and ctx == "technical_engineering") else None,
                )
                ground_truth_rules.append(gt_rule)

                # If temporal drift applies, insert the replacement rule active from drift_step
                if has_temporal_drift and aspect_cat == AspectCategory.VERBOSITY and ctx == "technical_engineering":
                    rule_counter += 1
                    r_id_new = f"rule_{user_id}_{rule_counter}_drift"
                    new_val = "elaborate" if val == "terse" else "terse"
                    new_stmt = "Provide thorough architectural explanations with extensive rationale." if new_val == "elaborate" else "Keep responses under 60 words, strictly bulleted code."
                    ground_truth_rules.append(
                        GroundTruthRule(
                            rule_id=r_id_new,
                            target_aspect=aspect_cat.value,
                            context_condition=ctx,
                            preference_statement=new_stmt,
                            rule_value=new_val,
                            active_from_step=drift_step,
                            deprecated_at_step=None,
                        )
                    )

        # Generate historical interactions
        history: List[Interaction] = []
        contexts_pool = list(config_template.keys())

        for step in range(interactions_per_user):
            ctx = contexts_pool[step % len(contexts_pool)]
            query_tuple = rng.choice(TASK_QUERIES.get(ctx, TASK_QUERIES["technical_engineering"]))
            query_text, topic = query_tuple

            # Determine active aspects for this step and context
            active_aspects: Dict[AspectCategory, tuple[str, str]] = {}
            for r in ground_truth_rules:
                if r.context_condition == ctx:
                    is_active = (step >= r.active_from_step) and (r.deprecated_at_step is None or step < r.deprecated_at_step)
                    if is_active:
                        cat = AspectCategory(r.target_aspect)
                        rule_val = r.rule_value or ("elaborate" if "elaborate" in r.preference_statement.lower() else "terse")
                        active_aspects[cat] = (rule_val, r.preference_statement)

            response_text = _render_synthetic_response(ctx, topic, active_aspects)
            applied_ids = [
                r.rule_id for r in ground_truth_rules
                if r.context_condition == ctx and (step >= r.active_from_step) and (r.deprecated_at_step is None or step < r.deprecated_at_step)
            ]

            interaction = Interaction(
                interaction_id=f"int_{user_id}_{step:03d}",
                user_id=user_id,
                step=step,
                context=ctx,
                query=query_text,
                user_response=response_text,
                applied_rules=applied_ids,
                aspect_manifestations={k.value: v[0] for k, v in active_aspects.items()},
            )
            history.append(interaction)

        # Generate held-out evaluation samples (strictly separated)
        test_samples: List[EvaluationSample] = []
        for ctx in contexts_pool:
            eval_query, eval_topic = rng.choice(TASK_QUERIES.get(ctx, TASK_QUERIES["technical_engineering"]))
            active_gt = [
                r for r in ground_truth_rules
                if r.context_condition == ctx and (r.deprecated_at_step is None)
            ]
            
            # Formulate deterministic verification criteria
            criteria: Dict[str, Any] = {"context": ctx, "forbidden_phrases": [], "required_elements": []}
            for r in active_gt:
                if r.target_aspect == AspectCategory.SALUTATION.value:
                    if "never" in r.preference_statement.lower() or "omit" in r.preference_statement.lower():
                        criteria["forbidden_phrases"].extend(["hello", "greetings", "hey there", "sincerely", "cheers"])
                    else:
                        criteria["required_elements"].append("salutation_present")
                elif r.target_aspect == AspectCategory.STRUCTURE.value:
                    if "bullet" in r.preference_statement.lower():
                        criteria["required_structure"] = "bullets"
                    elif "numbered" in r.preference_statement.lower():
                        criteria["required_structure"] = "numbered"
                elif r.target_aspect == AspectCategory.CODE_STYLE.value:
                    if "code_only" in r.preference_statement.lower() or "snippets" in r.preference_statement.lower():
                        criteria["requires_code"] = True
                    elif "avoid raw code" in r.preference_statement.lower():
                        criteria["forbid_code"] = True
                elif r.target_aspect == AspectCategory.VERBOSITY.value:
                    if "under 100" in r.preference_statement.lower() or "under 60" in r.preference_statement.lower():
                        criteria["max_word_count"] = 120
                    elif "comprehensive" in r.preference_statement.lower() or "thorough" in r.preference_statement.lower():
                        criteria["min_word_count"] = 80

            sample = EvaluationSample(
                sample_id=f"eval_{user_id}_{ctx}",
                user_id=user_id,
                context=ctx,
                query=f"Held-out test: {eval_query}",
                target_ground_truth_rules=active_gt,
                deterministic_criteria=criteria,
            )
            test_samples.append(sample)

        users.append(
            UserHistory(
                user_id=user_id,
                persona_name=persona_name,
                all_rules=ground_truth_rules,
                history=history,
                test_samples=test_samples,
            )
        )

    return users


def save_dataset_to_disk(dataset: List[UserHistory], output_path: str) -> None:
    """Serialize synthetic dataset to JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    serialized = []
    for u in dataset:
        serialized.append({
            "user_id": u.user_id,
            "persona_name": u.persona_name,
            "all_rules": [asdict(r) for r in u.all_rules],
            "history": [asdict(h) for h in u.history],
            "test_samples": [asdict(s) for s in u.test_samples],
        })

    with open(p, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)


def load_dataset_from_disk(input_path: str) -> List[UserHistory]:
    """Load serialized dataset from JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    users: List[UserHistory] = []
    for item in data:
        rules = [GroundTruthRule(**r) for r in item["all_rules"]]
        history = [Interaction(**h) for h in item["history"]]
        samples = []
        for s in item["test_samples"]:
            gt_rules = [GroundTruthRule(**r) for r in s["target_ground_truth_rules"]]
            samples.append(EvaluationSample(
                sample_id=s["sample_id"],
                user_id=s["user_id"],
                context=s["context"],
                query=s["query"],
                target_ground_truth_rules=gt_rules,
                deterministic_criteria=s["deterministic_criteria"],
            ))
        users.append(UserHistory(
            user_id=item["user_id"],
            persona_name=item["persona_name"],
            all_rules=rules,
            history=history,
            test_samples=samples,
        ))
    return users
