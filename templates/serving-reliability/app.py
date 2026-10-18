#!/usr/bin/env python3
"""Deterministic load/fault simulation for recommendation serving."""
from __future__ import annotations
import argparse, json

def simulate(requests:int, capacity:int, dependency_latency_ms:int, timeout_ms:int) -> dict[str,float|int]:
    served=min(requests,capacity)
    overloaded=requests-served
    timed_out=served if dependency_latency_ms > timeout_ms else 0
    personalized=served-timed_out
    fallback=overloaded+timed_out
    latencies=[dependency_latency_ms if index < personalized else min(timeout_ms,5) for index in range(requests)]
    p99=sorted(latencies)[max(0,math_index(len(latencies),0.99))] if latencies else 0
    return {"requests":requests,"personalized":personalized,"fallback":fallback,"non_empty":requests,"fallback_rate":round(fallback/max(requests,1),4),"p99_ms":p99}

def math_index(size:int,quantile:float)->int: return min(size-1,max(0,int(size*quantile)-1))

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--requests",type=int,default=100); parser.add_argument("--capacity",type=int,default=80); parser.add_argument("--dependency-latency-ms",type=int,default=20); parser.add_argument("--timeout-ms",type=int,default=50); args=parser.parse_args()
    if min(args.requests,args.capacity,args.timeout_ms)<1 or args.dependency_latency_ms<0: raise SystemExit("invalid simulation parameters")
    print(json.dumps(simulate(args.requests,args.capacity,args.dependency_latency_ms,args.timeout_ms)))
if __name__ == "__main__": main()
