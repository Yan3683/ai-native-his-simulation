# AI-Native HIS — Simulation & Reproducibility Package

Companion repository for the manuscript:

> **From Digital Health to Intelligent Health: Redefining Hospital Information Architecture in the Era of Large Language Models**
> Bing Yan, Information Center, Beijing Hospital, National Center for Geriatrics
> *BMC Medical Informatics and Decision Making* (revised manuscript, 2026)

This repository contains the simulation code, OpenClaw configuration files, and
synthetic-data tooling used for the simulation-based feasibility validation
reported in Section 5 of the manuscript. **No real patient data are used or
included anywhere in this repository.**

## Contents

```
├── scripts/
│   └── run_benchmark.py        # Workload generator + latency benchmark (Section 5.1.3)
├── openclaw-config/
│   ├── agents/                 # Six AI-service agent configurations (Section 5.1.2, Table 6)
│   │   ├── clinical_reasoning.yaml
│   │   ├── predictive_analytics.yaml
│   │   ├── clinical_nlp.yaml
│   │   ├── documentation_coding.yaml
│   │   ├── alert_manager.yaml
│   │   └── resource_scheduler.yaml
│   ├── workflows/
│   │   └── documentation.yaml  # Clinical documentation workflow (Section 5.2.3)
│   └── simulation/
│       ├── data_flow_test.yaml         # Ingestion/latency scenario (Section 5.2.1)
│       └── clinical_workflow_test.yaml # Workflow scenario (Section 5.2.3)
├── data/
│   └── README.md               # How to regenerate the synthetic FHIR datasets
└── results/                    # Benchmark outputs (CSV/JSON), generated locally
```

## Quick start

Requirements: Python ≥ 3.9 (standard library only; NumPy optional, used to
accelerate the vector-search substitute if installed).

```bash
# Reproduce the latency benchmark reported in Section 5 (30 independent runs)
python scripts/run_benchmark.py --runs 30 --out results
```

The script prints p50 / p95 / p99 and between-run variability for the
ingestion, vector-search-substitute, and knowledge-graph-substitute scenarios,
and writes `results/benchmark_results.csv` and `results/benchmark_summary.json`.

## Scope and fidelity

As detailed in Table 10 of the manuscript ("Simulation fidelity relative to the
proposed architecture"), the simulation implements the clinical reasoning
engine directly as OpenClaw agents, while the vector database cluster,
knowledge graph, and temporal/streaming layers are represented by lightweight
local substitutes due to single-host scale constraints. The vector-search and
knowledge-graph benchmarks in `run_benchmark.py` are explicitly these
substitutes, not production database measurements.

## OpenClaw configuration

The `openclaw-config/` directory contains the agent, workflow, and simulation
configurations describing the six AI services and the validation scenarios.
Copy the directory into an OpenClaw workspace to inspect or adapt them.

## Data availability

All synthetic datasets are generated deterministically by
`scripts/run_benchmark.py` (seeded generators); see `data/README.md`. No real
patient data were used in this study.

## Citation

```bibtex
@article{yan2026ainative,
  title   = {From Digital Health to Intelligent Health: Redefining Hospital
             Information Architecture in the Era of Large Language Models},
  author  = {Yan, Bing},
  journal = {BMC Medical Informatics and Decision Making},
  year    = {2026},
  note    = {Revised manuscript}
}
```

## License

Code and configuration files are released under the MIT License (see `LICENSE`).
