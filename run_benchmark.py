#!/usr/bin/env python3
"""
AI-Native HIS — Reproducible simulation benchmark
=================================================
Companion script for the manuscript "From Digital Health to Intelligent Health:
Redefining Hospital Information Architecture in the Era of Large Language Models".

What it does
------------
1. Generates synthetic FHIR R4 resources (Patient, Observation, Encounter,
   Condition) with configurable field distributions. NO real patient data.
2. Measures ingestion latency at a sustained target rate (default 100 res/s).
3. Measures a vector-search substitute (brute-force cosine similarity over a
   local embedding index — see manuscript Table 10, "Simulation fidelity").
4. Measures a knowledge-graph substitute (SQLite relational query).
5. Repeats each scenario N times and reports p50 / p95 / p99 and run-to-run
   variability, writing results to CSV + JSON.

Requirements: Python >= 3.9, standard library only.

Usage
-----
    python run_benchmark.py --runs 30 --rate 100 --out results/

The generated numbers are the ones to transcribe into manuscript Section 5.1.3
("Repetition and variability") and Tables 5-7, replacing the placeholders.
"""

import argparse
import csv
import json
import math
import os
import random
import sqlite3
import statistics
import time
import uuid

# ---------------------------------------------------------------- workloads

FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
         "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez"]
LOINC = [("8867-4", "Heart rate", "/min", 60, 100),
         ("9279-1", "Respiratory rate", "/min", 12, 20),
         ("2708-6", "Oxygen saturation", "%", 94, 100),
         ("8310-5", "Body temperature", "Cel", 36.2, 37.5),
         ("8462-4", "Diastolic blood pressure", "mm[Hg]", 60, 90),
         ("8480-6", "Systolic blood pressure", "mm[Hg]", 100, 140)]
CONDITIONS = [("E11.9", "Type 2 diabetes mellitus"),
              ("I10", "Essential hypertension"),
              ("J45.909", "Asthma, unspecified"),
              ("I25.10", "Coronary artery disease"),
              ("N18.30", "Chronic kidney disease, stage 3")]


def rand_patient() -> dict:
    return {
        "resourceType": "Patient",
        "id": str(uuid.uuid4()),
        "name": [{"family": random.choice(LAST),
                  "given": [random.choice(FIRST)]}],
        "gender": random.choice(["male", "female"]),
        "birthDate": f"{random.randint(1940, 2000)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
    }


def rand_observation(patient_id: str) -> dict:
    code, display, unit, lo, hi = random.choice(LOINC)
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": code,
                             "display": display}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "valueQuantity": {"value": round(random.uniform(lo, hi), 1),
                          "unit": unit, "system": "http://unitsofmeasure.org"},
    }


def rand_encounter(patient_id: str) -> dict:
    return {
        "resourceType": "Encounter",
        "id": str(uuid.uuid4()),
        "status": "finished",
        "class": {"code": random.choice(["AMB", "IMP", "EMER"])},
        "subject": {"reference": f"Patient/{patient_id}"},
        "period": {"start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }


def rand_condition(patient_id: str) -> dict:
    code, display = random.choice(CONDITIONS)
    return {
        "resourceType": "Condition",
        "id": str(uuid.uuid4()),
        "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10",
                             "code": code, "display": display}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "clinicalStatus": {"text": random.choice(["active", "resolved"])},
    }


def generate_resources(n: int) -> list:
    """Generate n synthetic FHIR resources across the four types."""
    out = []
    for _ in range(n):
        p = rand_patient()
        out.append(p)
        out.append(rand_observation(p["id"]))
        out.append(rand_encounter(p["id"]))
        out.append(rand_condition(p["id"]))
    random.shuffle(out)
    return out[:n]

# ------------------------------------------------------------- benchmarks

def bench_ingestion(resources: list, rate: int) -> list:
    """Simulate sustained ingestion at `rate` resources/second.

    Latency = serialize + validate + store (in-memory list append), measured
    per resource with time.perf_counter_ns. Pacing keeps the target rate so
    queueing behaviour approximates a streamed workload.
    """
    lat_ms = []
    interval = 1.0 / rate
    start = time.perf_counter()
    store = []
    for i, res in enumerate(resources):
        t0 = time.perf_counter_ns()
        payload = json.dumps(res).encode("utf-8")          # serialize
        assert res.get("resourceType") and res.get("id")   # validate
        store.append(payload)                              # store
        lat_ms.append((time.perf_counter_ns() - t0) / 1e6)
        # pace to target rate
        target = start + (i + 1) * interval
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
    return lat_ms


def _cosine(a: list, b: list) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12)


def bench_vector_search(dim: int, index_size: int, queries: int) -> list:
    """Vector-search SUBSTITUTE (fidelity table, Table 10): brute-force cosine
    over a random local index. NOT a production vector DB measurement."""
    try:
        import numpy as np
        rng = np.random.default_rng(42)
        index = rng.standard_normal((index_size, dim))
        index /= np.linalg.norm(index, axis=1, keepdims=True) + 1e-12
        lat_ms = []
        for _ in range(queries):
            q = rng.standard_normal(dim)
            q = q / (np.linalg.norm(q) + 1e-12)
            t0 = time.perf_counter_ns()
            best = float((index @ q).max())
            assert isinstance(best, float)
            lat_ms.append((time.perf_counter_ns() - t0) / 1e6)
        return lat_ms
    except ImportError:
        rng = random.Random(42)
        index = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(index_size)]
        lat_ms = []
        for _ in range(queries):
            q = [rng.gauss(0, 1) for _ in range(dim)]
            t0 = time.perf_counter_ns()
            best = max((_cosine(q, v) for v in index), default=0.0)
            assert isinstance(best, float)
            lat_ms.append((time.perf_counter_ns() - t0) / 1e6)
        return lat_ms


def bench_kg_query(db_path: str, nodes: int, queries: int) -> list:
    """Knowledge-graph SUBSTITUTE (Table 10): relational two-hop query over
    concept/edge tables in SQLite."""
    rng = random.Random(7)
    if os.path.exists(db_path):
        for _ in range(5):
            try:
                os.remove(db_path)
                break
            except PermissionError:
                time.sleep(0.5)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("CREATE TABLE node(id INTEGER PRIMARY KEY, label TEXT)")
        cur.execute("CREATE TABLE edge(src INT, dst INT, rel TEXT)")
        cur.executemany("INSERT INTO node VALUES(?, ?)",
                        [(i, f"Concept{i}") for i in range(nodes)])
        cur.executemany("INSERT INTO edge VALUES(?, ?, ?)",
                        [(rng.randrange(nodes), rng.randrange(nodes), "RELATED_TO")
                         for _ in range(nodes * 4)])
        con.commit()
        lat_ms = []
        for _ in range(queries):
            seed = rng.randrange(nodes)
            t0 = time.perf_counter_ns()
            cur.execute("""SELECT e2.dst FROM edge e1 JOIN edge e2 ON e1.dst = e2.src
                           WHERE e1.src = ? LIMIT 50""", (seed,))
            cur.fetchall()
            lat_ms.append((time.perf_counter_ns() - t0) / 1e6)
    finally:
        con.close()
    return lat_ms

# ---------------------------------------------------------------- reporting

def pct(values: list, p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return s[k]


def summarize(name: str, lat_ms: list) -> dict:
    return {
        "scenario": name,
        "n": len(lat_ms),
        "mean_ms": round(statistics.fmean(lat_ms), 3),
        "std_ms": round(statistics.stdev(lat_ms), 3) if len(lat_ms) > 1 else 0.0,
        "p50_ms": round(pct(lat_ms, 50), 3),
        "p95_ms": round(pct(lat_ms, 95), 3),
        "p99_ms": round(pct(lat_ms, 99), 3),
        "min_ms": round(min(lat_ms), 3),
        "max_ms": round(max(lat_ms), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30,
                    help="independent runs per scenario (default 30)")
    ap.add_argument("--rate", type=int, default=100,
                    help="ingestion rate, resources/second (default 100)")
    ap.add_argument("--resources", type=int, default=1000,
                    help="resources per ingestion run (default 1000)")
    ap.add_argument("--index-size", type=int, default=10000,
                    help="local vector index size (default 10000)")
    ap.add_argument("--dim", type=int, default=768,
                    help="embedding dimension (default 768)")
    ap.add_argument("--kg-nodes", type=int, default=5000)
    ap.add_argument("--queries", type=int, default=200,
                    help="queries per vector/KG run (default 200)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summaries = {k: [] for k in ("ingestion", "vector_search", "kg_query")}

    # Warm-up pass (results discarded) so caches/JIT stabilize before measurement
    print("warm-up pass (discarded)...", flush=True)
    warm_resources = generate_resources(200)
    bench_ingestion(warm_resources, args.rate)
    bench_vector_search(args.dim, min(args.index_size, 2000), 50)
    warm_db = os.path.join(args.out, "kg_warmup.db")
    bench_kg_query(warm_db, min(args.kg_nodes, 1000), 50)
    try:
        os.remove(warm_db)
    except OSError:
        pass

    for run in range(1, args.runs + 1):
        resources = generate_resources(args.resources)
        summaries["ingestion"] += bench_ingestion(resources, args.rate)
        summaries["vector_search"] += bench_vector_search(args.dim, args.index_size, args.queries)
        db_path = os.path.join(args.out, f"kg_run{run}.db")
        summaries["kg_query"] += bench_kg_query(db_path, args.kg_nodes, args.queries)
        try:
            os.remove(db_path)
        except OSError:
            pass
        print(f"run {run}/{args.runs} done", flush=True)

    rows = [summarize(k, v) for k, v in summaries.items()]

    csv_path = os.path.join(args.out, "benchmark_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    json_path = os.path.join(args.out, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": args.runs, "warmup": "1 discarded pass before measurement",
                   "parameters": vars(args),
                   "note": "vector_search and kg_query are substitutes per "
                           "manuscript Table 10 (simulation fidelity)",
                   "results": rows}, f, indent=2)

    print("\n=== Summary (transcribe these into Section 5.1.3 / Tables 5-7) ===")
    for r in rows:
        print(f"{r['scenario']:>14}: p50={r['p50_ms']}ms  p95={r['p95_ms']}ms  "
              f"p99={r['p99_ms']}ms  mean±std={r['mean_ms']}±{r['std_ms']}ms  n={r['n']}")
    print(f"\nWrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
