# Agent Toolkit

`OptimizationAgentToolkit` exposes optimization as a bounded in-process tool surface. It does not connect to an LLM service, send tensors over a network, or evaluate code from tool arguments.

## Register Workloads

```python
from custom_dl_optimizer import OptimizationAgentToolkit, Optimizer

toolkit = OptimizationAgentToolkit(Optimizer(device="cuda"))
toolkit.register_workload(
    "encoder-b16",
    encoder.eval(),
    tokens,
    attention_mask=mask,
    description="Encoder serving signature, batch 16",
)
```

Only objects registered by the host application can be optimized.

## Declared Tools

- `custom_dl_inspect_runtime`
- `custom_dl_list_workloads`
- `custom_dl_optimize`
- `custom_dl_get_report`

```python
schemas = toolkit.tool_schemas()
result = toolkit.invoke(
    "custom_dl_optimize",
    {"workload": "encoder-b16"},
)
```

The returned values are JSON-serializable dictionaries. A host can adapt the schemas to its agent framework and route approved calls to `invoke`.

## Security Boundary

The toolkit accepts a tool name and a small JSON argument object. Workload names resolve against an in-memory registry. There is no file path argument, module import argument, shell command, model download, arbitrary Python expression, or credential handling surface.

Agent approval and authorization remain the host application's responsibility. Register only workloads the agent is allowed to inspect and optimize.
