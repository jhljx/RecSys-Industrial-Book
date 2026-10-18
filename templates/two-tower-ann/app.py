#!/usr/bin/env python3
"""Train a deterministic two-tower toy model and query a versioned ANN service."""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass

VECTORS = {"i1": [1.0, 0.0], "i2": [0.8, 0.2], "i3": [0.0, 1.0], "i4": [-0.8, 0.2]}
POSITIVES = {"u1": "i1", "u2": "i3"}

@dataclass(frozen=True)
class Artifact:
    version: str
    vectors: dict[str, list[float]]

class VectorService:
    def __init__(self, artifact: Artifact) -> None: self._artifact = artifact
    def switch(self, model: Artifact, index: Artifact) -> None:
        if model.version != index.version: raise ValueError("model/index version mismatch")
        self._artifact = index
    def query(self, vector: list[float], version: str, limit: int) -> list[tuple[str, float]]:
        if version != self._artifact.version: raise ValueError("request/index version mismatch")
        return sorted(((item, cosine(vector, value)) for item, value in self._artifact.vectors.items()), key=lambda row: (-row[1], row[0]))[:limit]

def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / denominator

def negatives(user: str) -> list[str]:
    positive = POSITIVES[user]
    return [item for item in sorted(VECTORS) if item != positive]

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--user", default="u1"); parser.add_argument("--limit", type=int, default=3); parser.add_argument("--version", default="v1"); parser.add_argument("--inject-version-mismatch", action="store_true"); args=parser.parse_args()
    if args.user not in POSITIVES or args.limit < 1: raise SystemExit("invalid user or limit")
    model=Artifact("v1", VECTORS); index=Artifact("v0" if args.inject_version_mismatch else "v1", VECTORS); service=VectorService(model)
    try: service.switch(model,index); result=service.query(VECTORS[POSITIVES[args.user]],args.version,args.limit)
    except ValueError as error: print(json.dumps({"status":"rejected","error":str(error)})); raise SystemExit(2)
    print(json.dumps({"status":"ok","version":args.version,"positive":POSITIVES[args.user],"negatives":negatives(args.user),"neighbors":result}))
if __name__ == "__main__": main()
