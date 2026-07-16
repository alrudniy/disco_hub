"""
The multi-agent layer.

`base` defines the contract (AgentResult / Agent / the trace row) every agent
codes against. `llm` is the optional glm-4.6 client -- optional in the strong
sense: with no API key, agents must fall back to a deterministic path and say so.
`graph_index` is the compact CSR view of the merged knowledge graph that backs
the expertise-gap agent.

Nothing here imports the retriever or any large artifact at module import time: a
real Retriever load is ~20 GB of RAM, and importing this package must stay cheap
enough for a unit test.
"""
from agents.base import (
    Agent,
    AgentResult,
    make_evidence,
    run_agent,
    timed,
    to_trace_row,
)

__all__ = ["Agent", "AgentResult", "make_evidence", "run_agent", "timed",
           "to_trace_row"]
