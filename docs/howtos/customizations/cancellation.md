# Cancelling Long-Running Tasks

When working with large datasets or complex evaluations, some Ragas operations can take significant time to complete. The cancellation feature allows you to gracefully terminate these long-running tasks when needed, which is especially important in production environments.

## Overview

Ragas provides cancellation support for:
- **`evaluate()`** - Evaluation of datasets with metrics
- **`generate_with_langchain_docs()`** - Test set generation from documents

The cancellation mechanism is thread-safe and allows for graceful termination with partial results when possible.

## Basic Usage

### Cancellable Evaluation

Instead of running evaluation directly, you can get an executor that allows cancellation:

```py
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset

# Your dataset and metrics
dataset = EvaluationDataset(...)
metrics = [...]

# Get executor instead of running evaluation immediately
executor = evaluate(
    dataset=dataset,
    metrics=metrics,
    return_executor=True,  # Key parameter
)

# Now you can:
# - Cancel: executor.cancel()
# - Check status: executor.is_cancelled()
# - Get results: executor.results()  # This blocks until completion
```

### Cancellable Test Set Generation

Similar approach for test set generation:

```py
from ragas.testset.synthesizers.generate import TestsetGenerator

generator = TestsetGenerator(...)

# Get executor for cancellable generation
executor = generator.generate_with_langchain_docs(
    documents=documents,
    testset_size=100,
    return_executor=True,  # Allow access to Executor to cancel
)

# Use the same cancellation interface
executor.cancel()
```

## Production Patterns

### 1. Timeout Pattern

Automatically cancel operations that exceed a time limit:

```py
import threading
import time


def evaluate_with_timeout(dataset, metrics, timeout_seconds=300):
    """Run evaluation with automatic timeout."""
    # Get cancellable executor
    executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)

    results = None
    exception = None

    def run_evaluation():
        nonlocal results, exception
        try:
            results = executor.results()
        except Exception as e:
            exception = e

    # Start evaluation in background thread
    thread = threading.Thread(target=run_evaluation)
    thread.start()

    # Wait for completion or timeout
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(f"Evaluation exceeded {timeout_seconds}s timeout, cancelling...")
        executor.cancel()
        thread.join(timeout=10)  # Custom timeout as per need
        return None, "timeout"

    return results, exception


# Usage
results, error = evaluate_with_timeout(dataset, metrics, timeout_seconds=600)
if error == "timeout":
    print("Evaluation was cancelled due to timeout")
else:
    print(f"Evaluation completed: {results}")
```

### 2. Signal Handler Pattern (Ctrl+C)

Allow users to cancel with keyboard interrupt:

```py
import signal
import sys


def setup_cancellation_handler():
    """Set up graceful cancellation on Ctrl+C."""
    executor = None

    def signal_handler(signum, frame):
        if executor and not executor.is_cancelled():
            print("\nReceived interrupt signal, cancelling evaluation...")
            executor.cancel()
            print("Cancellation requested. Waiting for graceful shutdown...")
        sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    return lambda exec: setattr(signal_handler, "executor", exec)


# Usage
set_executor = setup_cancellation_handler()

executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)
set_executor(executor)

print("Running evaluation... Press Ctrl+C to cancel gracefully")
try:
    results = executor.results()
    print("Evaluation completed successfully")
except KeyboardInterrupt:
    print("Evaluation was cancelled")
```

### 3. Web Application Pattern

For web applications, cancel operations when requests are aborted:

```py
from flask import Flask, request
import threading
import uuid

app = Flask(__name__)
active_evaluations = {}


@app.route("/evaluate", methods=["POST"])
def start_evaluation():
    # Create unique evaluation ID
    eval_id = str(uuid.uuid4())

    # Get dataset and metrics from request
    dataset = get_dataset_from_request(request)
    metrics = get_metrics_from_request(request)

    # Start cancellable evaluation
    executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)
    active_evaluations[eval_id] = executor

    # Start evaluation in background
    def run_eval():
        try:
            results = executor.results()
            # Store results somewhere
            store_results(eval_id, results)
        except Exception as e:
            store_error(eval_id, str(e))
        finally:
            active_evaluations.pop(eval_id, None)

    threading.Thread(target=run_eval).start()

    return {"evaluation_id": eval_id, "status": "started"}


@app.route("/evaluate/<eval_id>/cancel", methods=["POST"])
def cancel_evaluation(eval_id):
    executor = active_evaluations.get(eval_id)
    if executor:
        executor.cancel()
        return {"status": "cancelled"}
    return {"error": "Evaluation not found"}, 404
```

## Advanced Usage

### Checking Cancellation Status

```py
executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)


# Start in background
def monitor_evaluation():
    while not executor.is_cancelled():
        print("Evaluation still running...")
        time.sleep(5)
    print("Evaluation was cancelled")


threading.Thread(target=monitor_evaluation).start()

# Cancel after some condition
if some_condition():
    executor.cancel()
```

### Partial Results

When cancellation occurs during execution, you may get partial results:

```py
executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)

try:
    results = executor.results()
    print(f"Completed {len(results)} evaluations")
except Exception as e:
    if executor.is_cancelled():
        print("Evaluation was cancelled - may have partial results")
    else:
        print(f"Evaluation failed: {e}")
```

### Custom Cancellation Logic

```py
class EvaluationManager:
    def __init__(self):
        self.executors = []

    def start_evaluation(self, dataset, metrics):
        executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)
        self.executors.append(executor)
        return executor

    def cancel_all(self):
        """Cancel all running evaluations."""
        for executor in self.executors:
            if not executor.is_cancelled():
                executor.cancel()
        print(f"Cancelled {len(self.executors)} evaluations")

    def cleanup_completed(self):
        """Remove completed executors."""
        self.executors = [ex for ex in self.executors if not ex.is_cancelled()]


# Usage
manager = EvaluationManager()

# Start multiple evaluations
exec1 = manager.start_evaluation(dataset1, metrics)
exec2 = manager.start_evaluation(dataset2, metrics)

# Cancel all if needed
manager.cancel_all()
```

## Best Practices

### 1. Always Use Timeouts in Production
```py
# Good: Always set reasonable timeouts
results, error = evaluate_with_timeout(
    dataset, metrics, timeout_seconds=1800
)  # 30 minutes

# Avoid: Indefinite blocking
results = executor.results()  # Could block forever
```

### 2. Handle Cancellation Gracefully
```py
try:
    results = executor.results()
    process_results(results)
except Exception as e:
    if executor.is_cancelled():
        log_cancellation()
        cleanup_partial_work()
    else:
        log_error(e)
        handle_failure()
```

### 3. Provide User Feedback
```py
def run_with_progress_and_cancellation(executor):
    print("Starting evaluation... Press Ctrl+C to cancel")
    
    # Monitor progress in background
    def show_progress():
        while not executor.is_cancelled():
            # Show some progress indication
            print(".", end="", flush=True)
            time.sleep(1)
    
    progress_thread = threading.Thread(target=show_progress)
    progress_thread.daemon = True
    progress_thread.start()
    
    try:
        return executor.results()
    except KeyboardInterrupt:
        print("\nCancelling...")
        executor.cancel()
        return None
```

### 4. Clean Up Resources
```py
def managed_evaluation(dataset, metrics):
    executor = None
    try:
        executor = evaluate(dataset=dataset, metrics=metrics, return_executor=True)
        return executor.results()
    except Exception as e:
        if executor:
            executor.cancel()
        raise
    finally:
        # Clean up any temporary resources
        cleanup_temp_files()
```

## Limitations

- **Async Operations**: Cancellation works at the task level, not within individual LLM calls
- **Partial State**: Cancelled operations may leave partial results or temporary files
- **Timing**: Cancellation is cooperative - tasks need to check for cancellation periodically
- **Dependencies**: Some external services may not respect cancellation immediately

## Troubleshooting

### Cancellation Not Working
```py
# Check if cancellation is set
if executor.is_cancelled():
    print("Cancellation was requested")
else:
    print("Cancellation not requested yet")

# Ensure you're calling cancel()
executor.cancel()
assert executor.is_cancelled()
```

### Tasks Still Running After Cancellation
```py
# Give time for graceful shutdown
executor.cancel()
time.sleep(2)  # Allow tasks to detect cancellation

# Force cleanup if needed
import asyncio

try:
    loop = asyncio.get_running_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()
except RuntimeError:
    pass  # No event loop running
```

The cancellation feature provides robust control over long-running Ragas operations, enabling production-ready deployments with proper resource management and user experience.