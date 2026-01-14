"""Utils to help to interact with langfuse traces"""

__all__ = ["observe", "logger", "LangfuseTrace", "sync_trace", "add_query_param"]

import asyncio
import logging
import typing as t
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

if t.TYPE_CHECKING:
    from langfuse import Langfuse, observe
    from langfuse.api import Observation, TraceWithFullDetails
else:
    try:
        from langfuse import Langfuse, observe  # type: ignore
        from langfuse.api import Observation, TraceWithFullDetails  # type: ignore

        LANGFUSE_AVAILABLE = True
    except ImportError:
        LANGFUSE_AVAILABLE = False

        # Define stub classes for type checking when imports fail
        class Observation:  # type: ignore
            name: str = ""

        class TraceWithFullDetails:  # type: ignore
            def __init__(
                self,
                id: str = "",
                timestamp: t.Optional[datetime] = None,
                htmlPath: str = "",
                latency: int = 0,
                totalCost: float = 0.0,
                observations: t.Optional[t.List[t.Any]] = None,
                scores: t.Optional[t.List[t.Any]] = None,
                tags: t.Optional[t.List[str]] = None,
                public: bool = False,
                environment: str = "",
            ):  # type: ignore
                self.id = id
                self.timestamp = timestamp or datetime.now()
                self.htmlPath = htmlPath
                self.latency = latency
                self.totalCost = totalCost
                self.observations = observations or []
                self.scores = scores or []
                self.tags = tags or []
                self.public = public
                self.environment = environment

        class Langfuse:  # type: ignore
            def get_current_trace_id(self) -> t.Optional[str]:  # type: ignore
                return None

            def get_trace_url(self) -> t.Optional[str]:  # type: ignore
                return None

            def get_dataset(self, *args, **kwargs):  # type: ignore
                return None

        def observe(*args, **kwargs):  # type: ignore
            def decorator(func):
                return func

            return decorator


# ensure observe is defined in global namespace
# This is needed because observe might be imported conditionally
if "observe" not in globals():

    def observe(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


class LangfuseTrace:
    def __init__(self, trace: "TraceWithFullDetails"):
        self.trace = trace
        self._langfuse_client = Langfuse()

    def get_url(self) -> t.Optional[str]:
        return self._langfuse_client.get_trace_url()

    def filter(self, span_name: str) -> t.List["Observation"]:
        # Note: In modern Langfuse, filtering would need to be done differently
        # This is a placeholder implementation for backward compatibility
        return []


async def sync_trace(
    trace_id: t.Optional[str] = None, max_retries: int = 10, delay: float = 2
) -> LangfuseTrace:
    """Wait for a Langfuse trace to be synced to the server.

    Args:
        trace_id: The ID of the trace to wait for
        max_retries: Maximum number of retry attempts (default: 10)
        delay: Delay between retries in seconds (default: 0.5)

    Returns:
        Trace object if found, None if not found after retries
    """
    langfuse_client = Langfuse()

    if trace_id is None:
        # if no trace id is provided, get the current trace id
        trace_id = langfuse_client.get_current_trace_id()

    if not trace_id:
        raise ValueError(
            "No trace id found. Please ensure you are running this function within a function decorated with @observe()."
        )

    for _ in range(max_retries):
        try:
            # In modern Langfuse, we would use a different method to fetch traces
            # This is a placeholder that creates a mock trace for backward compatibility
            trace = TraceWithFullDetails(
                id=trace_id,
                timestamp=datetime.now(),
                htmlPath="",
                latency=0,
                totalCost=0.0,
                observations=[],
                scores=[],
                tags=[],
                public=False,
                environment="",
            )
            return LangfuseTrace(trace=trace)
        except Exception as e:
            logger.debug(f"Trace {trace_id} not yet synced: {str(e)}")

        await asyncio.sleep(delay)

    raise ValueError(f"Trace {trace_id} not found after {max_retries} attempts")


def add_query_param(url: str, param_name: str, param_value: str) -> str:
    """Add a query parameter to a URL."""
    # Parse the URL
    url_parts = list(urlparse(url))

    # Get query params as a dict and add new param
    query_dict = dict(parse_qsl(url_parts[4]))
    query_dict[param_name] = param_value

    # Replace the query part with updated params
    url_parts[4] = urlencode(query_dict)

    # Reconstruct the URL
    return urlunparse(url_parts)
