# tau2-observe

LLM Observability for tau2-bench using Datadog.

## Overview

This project demonstrates end-to-end LLM observability for the Google Cloud x Datadog hackathon:
- Gemini LLM traces via ddtrace + LiteLLM
- Custom tau2 evaluation metrics
- Detection rules with Case/Incident management
- Health dashboards

The instrumented LLM application is [tau2-bench-agent](https://github.com/wuTims/tau2-bench-agent), an agentified deployment of tau2 benchmark capabilities.

## Architecture

```
┌───────────────────┐      A2A       ┌─────────────────┐      A2A       ┌─────────────────┐
│ traffic_generator │───────────────▶│   tau2_agent    │───────────────▶│   mock_agent    │
└───────────────────┘                │   (Cloud Run)   │                │   (Cloud Run)   │
                                     └────────┬────────┘                └─────────────────┘
                                              │
                                              │ ddtrace (llmobs + patch)
                                              ▼
                                     ┌─────────────────┐
                                     │    Datadog      │
                                     │ (LLM Obs + APM) │
                                     └─────────────────┘
```

## Prerequisites

1. **API Keys**:
   - `DD_API_KEY` - Datadog API key
   - `DD_APP_KEY` - Datadog Application key
   - `GEMINI_API_KEY` - For A2A requests

2. **Python 3.10+** with uv package manager

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/wuTims/tau2-observe.git
cd tau2-observe
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## One-Time Setup

Create Datadog monitors, SLOs, and dashboards:

```bash
export DD_API_KEY=your_datadog_api_key
export DD_APP_KEY=your_datadog_app_key

uv run python scripts/setup_datadog.py --all
```

## Deployed Endpoints

| Service | URL |
|---------|-----|
| tau2_agent | https://tau2-agent-676371821546.us-west2.run.app |
| simple_gemini_agent | https://simple-gemini-agent-4twyiz3sqq-wl.a.run.app |
| kimi_litellm_agent | https://kimi-litellm-agent-4twyiz3sqq-wl.a.run.app |

## Sample A2A Queries

**Health check:**
```bash
curl https://tau2-agent-676371821546.us-west2.run.app/a2a/tau2_agent/.well-known/agent-card.json
```

**Run evaluation:**
```bash
curl -X POST https://tau2-agent-676371821546.us-west2.run.app/a2a/tau2_agent \
  -H "Content-Type: application/json" \
  -H "X-User-LLM-Model: gemini/gemini-3-flash-preview" \
  -H "X-User-LLM-API-Key: $GEMINI_API_KEY" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/stream",
    "params": {
      "message": {
        "messageId": "demo-001",
        "role": "user",
        "parts": [{
          "text": "Run an evaluation on the airline domain for agent at https://simple-gemini-agent-4twyiz3sqq-wl.a.run.app/a2a/simple_gemini_agent. Use 1 tasks and 1 trial(s)."
        }]
      }
    },
    "id": "1"
  }'
```

## Traffic Generation

Generate evaluation traffic across different domains:

```bash
# Airline domain - 2 tasks, 1 trial, 2 evaluations
uv run python scripts/traffic_generator.py \
  --domain airline --num-tasks 2 --num-trials 1 --count 2

# Retail domain - 1 task, 3 trials, 2 evaluation
uv run python scripts/traffic_generator.py \
  --domain retail --num-tasks 1 --num-trials 3 --count 2

# Telecom domain - 3 tasks, 1 trial, 2 evaluations
uv run python scripts/traffic_generator.py \
  --domain telecom --num-tasks 3 --num-trials 1 --count 2
```

**Trigger failure monitors (DR-002, DR-006):**
```bash
uv run python scripts/traffic_generator.py \
  --mode failure --domain airline --count 3
```

**Use multiple mock agents:**
```bash
uv run python scripts/traffic_generator.py \
  --mock-urls https://simple-gemini-agent-4twyiz3sqq-wl.a.run.app \
              https://kimi-litellm-agent-4twyiz3sqq-wl.a.run.app \
  --domain airline --count 2
```

## Datadog URLs

After running traffic, view results at:

| Resource | URL |
|----------|-----|
| Dashboard | https://app.datadoghq.com/dashboard/tau2-bench-health |
| APM Traces | https://app.datadoghq.com/apm/traces?query=service:tau2-bench-agent |
| Metrics | https://app.datadoghq.com/metric/explorer?query=tau2.task.reward |
| Monitors | https://app.datadoghq.com/monitors/manage |

## Detection Rules

| ID | Name | Trigger Condition | Action |
|----|------|-------------------|--------|
| DR-001 | High Error Rate | error_count / total > 0.2 | Create Case |
| DR-002 | Task Quality Degradation | avg:tau2.task.reward < 0.5 | Create Case |
| DR-003 | Token Cost Anomaly | token_cost > 2x baseline | Alert |
| DR-004 | Premature Termination | termination:max_errors > 10/hr | Create Incident |
| DR-005 | Latency SLO Breach | p99:duration > 60s | SLO Alert |
| DR-006 | Low Task Efficiency | reward_per_turn < 0.03 | Create Case |

## Directory Structure

```
tau2-observe/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── tau2_agent/              # Instrumented LLM application
├── simple_gemini_agent/     # Mock agent (Gemini)
├── kimi_litellm_agent/      # Mock agent (Kimi)
├── configs/
│   ├── monitors.json
│   ├── slos.json
│   ├── dashboards_agents.json
│   └── dashboards_operations.json
├── scripts/
│   ├── traffic_generator.py
│   ├── emit_metrics.py
│   └── setup_datadog.py
└── data/tau2/               # Domain data
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DD_API_KEY` | Yes | Datadog API key |
| `DD_APP_KEY` | Yes | Datadog Application key |
| `DD_SITE` | No | Datadog site (default: datadoghq.com) |
| `GEMINI_API_KEY` | Yes | Gemini API key for A2A requests |
| `TAU2_AGENT_URL` | No | Custom tau2_agent base URL |
| `MOCK_AGENT_URL` | No | Custom mock agent base URL |

## Datadog Organization

Account: tim.wulin@gmail.com

## License

Apache-2.0
