#!/usr/bin/env python3
"""Demonstrate point-in-time feature joins and leakage detection."""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass

@dataclass(frozen=True)
class Event: user: str; timestamp: int; value: float

def point_in_time_join(labels: list[Event], features: list[Event]) -> list[dict[str, float]]:
    rows=[]
    for label in labels:
        eligible=[feature for feature in features if feature.user == label.user and feature.timestamp <= label.timestamp]
        if not eligible: continue
        feature=max(eligible,key=lambda item:item.timestamp)
        rows.append({"label_time":label.timestamp,"feature_time":feature.timestamp,"feature":feature.value,"label":label.value})
    return rows

def validate(rows: list[dict[str,float]]) -> None:
    leaked=[row for row in rows if row["feature_time"] > row["label_time"]]
    if leaked: raise ValueError(f"future features detected: {len(leaked)}")

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--inject-leakage",action="store_true"); args=parser.parse_args()
    labels=[Event("u1",100,1),Event("u1",200,0)]; features=[Event("u1",80,0.2),Event("u1",150,0.8),Event("u1",220,1.0)]
    rows=point_in_time_join(labels,features)
    if args.inject_leakage: rows[0]["feature_time"]=220
    try: validate(rows)
    except ValueError as error: print(json.dumps({"status":"rejected","error":str(error)})); raise SystemExit(2)
    print(json.dumps({"status":"ok","rows":rows,"max_feature_age":max(row["label_time"]-row["feature_time"] for row in rows)}))
if __name__ == "__main__": main()
