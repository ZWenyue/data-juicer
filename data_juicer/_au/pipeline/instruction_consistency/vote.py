"""Multi-expert voting aggregation strategies."""
from __future__ import annotations


def aggregate_votes(votes: list[dict], strategy: str) -> dict:
    """Aggregate expert VLM votes into a final verdict.

    Args:
        votes: List of {"model": str, "verdict": str, "confidence": float}.
        strategy: "majority" | "weighted" | "unanimous_override".

    Returns:
        {"final_verdict": str, "final_score": float,
         "voting_strategy": str, "expert_votes": list,
         "num_experts_responded": int}
    """
    if not votes:
        return {
            "final_verdict": "inconsistent",
            "final_score": 0.0,
            "voting_strategy": strategy,
            "expert_votes": [],
            "num_experts_responded": 0,
        }

    if strategy == "majority":
        return _majority(votes)
    elif strategy == "weighted":
        return _weighted(votes)
    elif strategy == "unanimous_override":
        return _unanimous(votes)
    else:
        raise ValueError(f"Unknown voting strategy: {strategy}")


def _majority(votes):
    c = sum(1 for v in votes if v["verdict"] == "consistent")
    n = len(votes)
    final = "consistent" if c > n / 2 else "inconsistent"
    return _result(final, c / n, "majority", votes)


def _weighted(votes):
    w_c = sum(v["confidence"] for v in votes
              if v["verdict"] == "consistent")
    w_total = sum(v["confidence"] for v in votes)
    score = w_c / w_total if w_total > 0 else 0
    final = "consistent" if score > 0.5 else "inconsistent"
    return _result(final, score, "weighted", votes)


def _unanimous(votes):
    verdicts = set(v["verdict"] for v in votes)
    if len(verdicts) == 1:
        return _result(verdicts.pop(), 1.0, "unanimous_override", votes)
    return _result("ambiguous", 0.5, "unanimous_override", votes)


def _result(verdict, score, strategy, votes):
    return {
        "final_verdict": verdict,
        "final_score": round(score, 4),
        "voting_strategy": strategy,
        "expert_votes": votes,
        "num_experts_responded": len(votes),
    }
