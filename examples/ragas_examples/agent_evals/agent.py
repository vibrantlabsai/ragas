import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import openai

SYSTEM_MESSAGE = """You are a mathematical problem-solving agent. You can only use these four atomic tools to solve problems:
- add(a, b): Add two numbers
- sub(a, b): Subtract b from a
- mul(a, b): Multiply two numbers
- div(a, b): Divide a by b

Your task is to break down complex mathematical expressions into a sequence of these atomic operations, following proper order of operations (parentheses, multiplication/division, addition/subtraction).

For each step, call the appropriate tool with the correct arguments. Work step by step, showing your reasoning.

When you have the final answer, respond with just the number."""


@dataclass
class TraceEvent:
    """Single event in the application trace"""

    event_type: (
        str  # "llm_call", "tool_execution", "error", "init", "result_extraction"
    )
    component: str  # "openai_api", "math_tools", "agent", "parser"
    data: Dict[str, Any]


@dataclass
class ToolResult:
    tool_name: str
    args: Dict[str, float]
    result: float
    step_number: int


class MathToolsAgent:
    def __init__(
        self,
        client,
        model_name: str = "gpt-4o",
        system_message: str = SYSTEM_MESSAGE,
        logdir: str = "logs",
    ):
        """
        Initialize the LLM agent with OpenAI API

        Args:
            client: OpenAI client instance
            model_name: Name of the model to use
            system_message: System message for the agent
            logdir: Directory to save trace logs
        """

        self.client = client
        self.system_message = system_message
        self.model_name = model_name
        self.step_counter = 0
        self.traces = []
        self.logdir = logdir

        # Create log directory if it doesn't exist
        os.makedirs(self.logdir, exist_ok=True)

        # Define available tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers together",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sub",
                    "description": "Subtract second number from first number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "Number to subtract from",
                            },
                            "b": {
                                "type": "number",
                                "description": "Number to subtract",
                            },
                        },
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mul",
                    "description": "Multiply two numbers together",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "div",
                    "description": "Divide first number by second number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {
                                "type": "number",
                                "description": "Number to divide (numerator)",
                            },
                            "b": {
                                "type": "number",
                                "description": "Number to divide by (denominator)",
                            },
                        },
                        "required": ["a", "b"],
                    },
                },
            },
        ]

    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        result = a + b

        return result

    def sub(self, a: float, b: float) -> float:
        """Subtract b from a"""
        result = a - b
        return result

    def mul(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        result = a * b
        return result

    def div(self, a: float, b: float) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b
        return result

    def _execute_tool_call(self, tool_call) -> str:
        """Execute a tool call and return the result"""

        self.traces.append(
            TraceEvent(
                event_type="tool_execution",
                component="math_tools",
                data={
                    "tool_name": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments),
                },
            )
        )

        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # Execute the appropriate function
        if function_name == "add":
            result = self.add(arguments["a"], arguments["b"])
        elif function_name == "sub":
            result = self.sub(arguments["a"], arguments["b"])
        elif function_name == "mul":
            result = self.mul(arguments["a"], arguments["b"])
        elif function_name == "div":
            result = self.div(arguments["a"], arguments["b"])
        else:
            raise ValueError(f"Unknown function: {function_name}")

        self.traces.append(
            TraceEvent(
                event_type="tool_result",
                component="math_tools",
                data={
                    "result": result,
                },
            )
        )

        return str(result)

    def export_traces_to_log(
        self, run_id: str, problem: str, final_result: Optional[float] = None
    ):
        """
        Export traces to a log file with run_id

        Args:
            run_id: Unique identifier for this run
            problem: The problem that was solved
            final_result: The final result of the computation
        """
        timestamp = datetime.now().isoformat()
        log_filename = (
            f"run_{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        )
        log_filepath = os.path.join(self.logdir, log_filename)

        log_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "problem": problem,
            "final_result": final_result,
            "model_name": self.model_name,
            "traces": [asdict(trace) for trace in self.traces],
        }

        with open(log_filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        logging.info(f"Traces exported to: {log_filepath}")
        return log_filepath

    def solve(
        self, problem: str, max_iterations: int = 10, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a math problem using iterative planning with LLM and atomic tools

        Args:
            problem: Mathematical expression or problem to solve
            max_iterations: Maximum number of LLM iterations to prevent infinite loops
            run_id: Optional run identifier. If None, generates one automatically

        Returns:
            Final numerical result
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(problem) % 10000:04d}"

        # Reset traces for each new problem
        self.traces = []

        logging.info(f"Solving: {problem} (Run ID: {run_id})")
        logging.info("=" * 60)

        # Reset state
        self.execution_history = []
        self.step_counter = 0

        messages = [
            {"role": "system", "content": self.system_message},
            {
                "role": "user",
                "content": f"Solve this mathematical expression step by step: {problem}",
            },
        ]

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logging.info(f"\n--- LLM Iteration {iteration} ---")

            try:
                self.traces.append(
                    TraceEvent(
                        event_type="llm_call",
                        component="openai_api",
                        data={
                            "model": self.model_name,
                            "messages": messages,
                            # "tools": [tool["function"] for tool in self.tools]
                        },
                    )
                )

                # Call OpenAI API with function calling
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    # temperature=0
                )

                message = response.choices[0].message
                messages.append(message.model_dump())

                self.traces.append(
                    TraceEvent(
                        event_type="llm_response",
                        component="openai_api",
                        data={
                            "content": message.content,
                            "tool_calls": (
                                [tool.model_dump() for tool in message.tool_calls]
                                if message.tool_calls
                                else []
                            ),
                        },
                    )
                )

                # Check if the model wants to call functions
                if message.tool_calls:
                    logging.info(
                        f"LLM planning: {message.content or 'Executing tools...'}"
                    )

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        result = self._execute_tool_call(tool_call)

                        # Add tool result to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )
                else:
                    # No more tool calls - this should be the final answer
                    logging.info(f"LLM final response: {message.content}")

                    # Try to extract the numerical result
                    try:
                        # Look for a number in the response
                        import re

                        numbers = re.findall(r"-?\d+\.?\d*", message.content)
                        if numbers:
                            final_result = float(
                                numbers[-1]
                            )  # Take the last number found
                            logging.info("=" * 60)
                            logging.info(f"Final result: {final_result}")
                            self.traces.append(
                                TraceEvent(
                                    event_type="result_extraction",
                                    component="math_tools",
                                    data={"final_result": final_result},
                                )
                            )

                            # Export traces to log file
                            log_filename = self.export_traces_to_log(
                                run_id, problem, final_result
                            )
                            return {"result": final_result, "log_file": log_filename}

                        else:
                            logging.info(
                                "Could not extract numerical result from LLM response"
                            )
                            break
                    except ValueError:
                        logging.info("Could not parse final result as number")
                        break

            except Exception as e:
                logging.info(f"Error in iteration {iteration}: {e}")
                break

        logging.info("Max iterations reached or error occurred")
        # Export traces even if solve failed
        return {
            "result": 0,
            "log_file": self.export_traces_to_log(run_id, problem, 0.0),
        }


def get_default_agent(
    model_name: str = "gpt-4o", logdir: str = "logs"
) -> MathToolsAgent:
    """Get a default instance of the MathToolsAgent with OpenAI client"""
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return MathToolsAgent(client=openai_client, model_name=model_name, logdir=logdir)


if __name__ == "__main__":
    # Example usage
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    agent = MathToolsAgent(client, logdir="agent_logs")

    problem = "((2 + 3) * 4) - (6 / 2)"
    print(f"Problem: {problem}")

    result = agent.solve(problem)
    print(f"Result: {result}")
