"""
Tracing integrations for Ragas evaluation framework.

This module provides integrations with popular tracing and observability platforms
to track and monitor Ragas evaluation runs.

Supported Platforms:
- Langfuse: Open-source LLM engineering platform
- MLflow: Machine learning lifecycle management platform
- OpenTelemetry: Vendor-neutral observability framework

Example:
    Basic usage with Langfuse:
    ```python
    from ragas.integrations.tracing.langfuse import observe, sync_trace
    from ragas import evaluate

    @observe()
    def run_evaluation():
        result = evaluate(dataset, metrics)
        return result

    # Get trace after evaluation
    trace = await sync_trace()
    print(trace.get_url())
    ```

    Basic usage with MLflow:
    ```python
    from ragas.integrations.tracing.mlflow import sync_trace
    from ragas import evaluate
    import mlflow

    with mlflow.start_run():
        result = evaluate(dataset, metrics)
        trace = await sync_trace()
        print(trace.get_url())
    ```
"""

# Type stubs for pyright - these won't execute but provide type information
if False:
    from .langfuse import (  # noqa: F401
        LangfuseTrace,
        add_query_param,
        logger,
        observe,
        sync_trace,
    )
    from .mlflow import MLflowTrace  # noqa: F401
    from .opentelemetry import OpenTelemetryTracer, RagasTracer  # noqa: F401


# Lazy imports to handle optional dependencies gracefully
def __getattr__(name: str):
    if name in ["observe", "logger", "LangfuseTrace", "sync_trace", "add_query_param"]:
        from .langfuse import (
            LangfuseTrace,
            add_query_param,
            logger,
            observe,
            sync_trace,
        )

        if name == "observe":
            return observe
        elif name == "logger":
            return logger
        elif name == "LangfuseTrace":
            return LangfuseTrace
        elif name == "sync_trace":
            return sync_trace
        elif name == "add_query_param":
            return add_query_param
    elif name == "MLflowTrace":
        from .mlflow import MLflowTrace

        return MLflowTrace
    elif name in ["OpenTelemetryTracer", "RagasTracer"]:
        from .opentelemetry import OpenTelemetryTracer, RagasTracer

        if name == "OpenTelemetryTracer":
            return OpenTelemetryTracer
        elif name == "RagasTracer":
            return RagasTracer
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
