"""Base prompt class for metrics with structured input/output models."""

import json
import typing as t
from abc import ABC

from pydantic import BaseModel

# Type variables for generics
InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


class BasePrompt(ABC, t.Generic[InputModel, OutputModel]):
    """
    Base class for structured prompts with type-safe input/output models.

    Attributes:
        input_model: Pydantic model class for input validation
        output_model: Pydantic model class for output schema generation
        instruction: Task description for the LLM
        examples: List of (input, output) example pairs for few-shot learning
        language: Language for the prompt (default: "english")
    """

    # Must be set by subclasses
    input_model: t.Type[InputModel]
    output_model: t.Type[OutputModel]
    instruction: str
    examples: t.List[t.Tuple[InputModel, OutputModel]]
    language: str = "english"

    def to_string(self, data: InputModel) -> str:
        """
        Convert prompt with input data to complete prompt string for LLM.

        Args:
            data: Input data instance (validated by input_model)

        Returns:
            Complete prompt string ready for LLM
        """
        # Generate JSON schema for output
        output_schema = json.dumps(self.output_model.model_json_schema())

        # Generate examples section
        examples_str = self._generate_examples()

        # Convert input data to JSON
        input_json = data.model_dump_json(indent=4, exclude_none=True)

        # Build complete prompt (matches existing function format)
        return f"""{self.instruction}
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{output_schema}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

{examples_str}
-----------------------------

Now perform the same with the following input
input: {input_json}
Output: """

    def _generate_examples(self) -> str:
        """
        Generate examples section of the prompt.

        Returns:
            Formatted examples string or empty string if no examples
        """
        if not self.examples:
            return ""

        example_strings = []
        for idx, (input_data, output_data) in enumerate(self.examples):
            example_strings.append(
                f"Example {idx + 1}\n"
                f"Input: {input_data.model_dump_json(indent=4)}\n"
                f"Output: {output_data.model_dump_json(indent=4)}"
            )

        return "--------EXAMPLES-----------\n" + "\n\n".join(example_strings)

    async def adapt(
        self,
        target_language: str,
        llm,
        adapt_instruction: bool = False,
    ) -> "BasePrompt[InputModel, OutputModel]":
        """
        Adapt the prompt to a new language using minimal translation.

        Args:
            target_language: Target language (e.g., "spanish", "french")
            llm: LLM instance for translation
            adapt_instruction: Whether to adapt instruction text (default: False)

        Returns:
            New prompt instance adapted to the target language
        """
        import copy

        # Create adapted prompt
        new_prompt = copy.deepcopy(self)
        new_prompt.language = target_language

        # Translate instruction if requested
        if adapt_instruction:
            instruction_prompt = f"Translate this to {target_language}, keep technical terms: {self.instruction}"
            try:
                response = await llm.agenerate(instruction_prompt)
                new_prompt.instruction = str(response).strip()
            except Exception:
                # Keep original if translation fails
                pass

        # Translate examples (simplified approach)
        translated_examples = []
        for input_ex, output_ex in self.examples:
            try:
                # Simple per-example translation
                example_prompt = f"""Translate this example to {target_language}, keep the same structure:

Input: {input_ex.model_dump_json()}
Output: {output_ex.model_dump_json()}

Return as: Input: {{translated_input_json}} Output: {{translated_output_json}}"""

                response = await llm.agenerate(example_prompt)

                # Try to extract translated JSON (basic parsing)
                response_str = str(response)
                if "Input:" in response_str and "Output:" in response_str:
                    parts = response_str.split("Output:")
                    input_part = parts[0].replace("Input:", "").strip()
                    output_part = parts[1].strip()

                    translated_input = self.input_model.model_validate_json(input_part)
                    translated_output = self.output_model.model_validate_json(
                        output_part
                    )
                    translated_examples.append((translated_input, translated_output))
                else:
                    # Fallback to original
                    translated_examples.append((input_ex, output_ex))

            except Exception:
                # Fallback to original example if translation fails
                translated_examples.append((input_ex, output_ex))

        new_prompt.examples = translated_examples
        return new_prompt
