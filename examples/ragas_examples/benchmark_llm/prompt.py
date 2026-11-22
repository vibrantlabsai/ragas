import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv(".env")

DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"


def get_client() -> AsyncOpenAI:
    """Lazily create an AsyncOpenAI client, requiring the API key only when used.

    This avoids raising errors during module import (e.g., when running --help).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please export it before running prompts."
        )
    return AsyncOpenAI(api_key=api_key)


SYSTEM_PROMPT = """You are a discount calculation assistant. I will provide a customer profile and you must calculate their discount percentage and explain your reasoning.

Discount rules:
- Age 65+ OR student status: 15% discount
- Annual income < $30,000: 20% discount
- Premium member for 2+ years: 10% discount
- New customer (< 6 months): 5% discount

Rules can stack up to a maximum of 35% discount.

Respond in JSON format only:
{
  "discount_percentage": number,
  "reason": "clear explanation of which rules apply and calculations",
  "applied_rules": ["list", "of", "applied", "rule", "names"]
}"""


async def run_prompt(prompt: str, model: str = DEFAULT_MODEL):
    """Run the discount calculation prompt with the specified model."""
    client = get_client()
    response = await client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    response = response.choices[0].message.content.strip()
    return response


if __name__ == "__main__":
    import asyncio
    
    async def main():
        customer_profile = """
        Customer Profile:
        - Name: Sarah Johnson
        - Age: 67
        - Student: No
        - Annual Income: $45,000
        - Premium Member: Yes, for 3 years
        - Account Age: 3 years
        """
        print("=== System Prompt ===")
        print(SYSTEM_PROMPT)
        print("\n=== Customer Profile ===")
        print(customer_profile)
        print(f"\n=== Running Prompt with default model {DEFAULT_MODEL} ===")
        print(await run_prompt(customer_profile, model=DEFAULT_MODEL))
    
    asyncio.run(main())
