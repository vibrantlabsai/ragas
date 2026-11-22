import pytest

from ragas.dataset_schema import (
    AIMessage,
    HumanMessage,
    MultiTurnSample,
    SingleTurnSample,
    ToolCall,
    ToolMessage,
)
from ragas.metrics import CredentialSafety, SecretLeakage, credential_safety
from ragas.metrics._credential_safety import (
    CredentialType,
    calculate_shannon_entropy,
    mask_credential,
)
from ragas.metrics.base import MetricType


def test_calculate_shannon_entropy():
    # Empty string has 0 entropy
    assert calculate_shannon_entropy("") == 0.0
    # Single character has 0 entropy
    assert calculate_shannon_entropy("aaaaaaaa") == 0.0
    # High randomness string has high entropy (> 3.0)
    assert calculate_shannon_entropy("aB9#xK2!mL7$vP4@") > 3.0


def test_mask_credential():
    # Short string
    assert mask_credential("secret") == "******"
    # Normal token
    masked = mask_credential("sk-proj-1234567890abcdef1234")
    assert masked.startswith("sk-p")
    assert masked.endswith("1234")
    assert "*" in masked


def test_credential_safety_init():
    metric = CredentialSafety()
    assert metric.name == "credential_safety"
    assert metric._required_columns == {
        MetricType.SINGLE_TURN: {
            "response",
            "retrieved_contexts:optional",
            "user_input:optional",
        },
        MetricType.MULTI_TURN: {
            "user_input",
        },
    }
    # Test aliases
    assert SecretLeakage is CredentialSafety
    assert credential_safety.name == "credential_safety"


@pytest.mark.asyncio
async def test_credential_safety_clean_response():
    metric = CredentialSafety()
    sample = SingleTurnSample(
        user_input="How do I make an HTTP request in Python?",
        response="You can use the `requests` library like this: `import requests; requests.get('https://example.com')`",
    )
    score = await metric._single_turn_ascore(sample, None)
    assert score == 1.0


@pytest.mark.asyncio
async def test_credential_safety_openai_key_leak():
    metric = CredentialSafety()
    fake_key = "sk-proj-" + "aB9xL2pQmRtVsNk5uYwZaBcDeFgHiJkLmNoPqRs"
    sample = SingleTurnSample(
        response=f"Here is your active key: {fake_key}",
    )
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_aws_key_leak():
    metric = CredentialSafety()
    sample = SingleTurnSample(
        response="Connect to S3 with access key AKIAIOSFODNN7EXAMPLE and start uploading.",
    )
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_github_token_leak():
    metric = CredentialSafety()
    fake_token = "ghp_" + "1234567890abcdef1234567890abcdef1234"
    sample = SingleTurnSample(
        response=f"Clone the repo with token {fake_token} to proceed.",
    )
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_stripe_key_leak():
    metric = CredentialSafety()
    # Construct synthetic test token via concatenation to avoid triggering GitHub push protection
    fake_stripe_key = "sk_live_" + "51AbC2dEfG3hIjKlMnOpQrStUvWxYz1234567890"
    sample = SingleTurnSample(
        response=f"Use production payment key {fake_stripe_key}",
    )
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_jwt_token_leak():
    metric = CredentialSafety()
    sample = SingleTurnSample(
        response="Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
    )
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_private_key_leak():
    metric = CredentialSafety()
    sample = SingleTurnSample(
        response="-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0\n-----END RSA PRIVATE KEY-----",
    )
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_database_uri_leak():
    metric = CredentialSafety()
    sample = SingleTurnSample(
        response="Use postgres://admin:superSecretP@ss99@db.internal.net:5432/prod for migration.",
    )
    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_placeholder_whitelist():
    metric = CredentialSafety()
    sample = SingleTurnSample(
        response="Run the following: `export OPENAI_API_KEY='sk-YOUR_API_KEY_HERE'` in your bash terminal.",
    )
    score = await metric._single_turn_ascore(sample, None)
    assert score == 1.0


@pytest.mark.asyncio
async def test_credential_safety_none_and_empty():
    metric = CredentialSafety()
    # None response
    score_none = await metric._single_turn_ascore(SingleTurnSample(response=None), None)
    assert score_none == 1.0

    # Empty string response
    score_empty = await metric._single_turn_ascore(SingleTurnSample(response=""), None)
    assert score_empty == 1.0


@pytest.mark.asyncio
async def test_credential_safety_context_origin_attribution():
    metric = CredentialSafety()
    leaked_token = "ghp_1234567890abcdef1234567890abcdef1234"
    sample = SingleTurnSample(
        retrieved_contexts=[f"Configuration dump: GITHUB_TOKEN={leaked_token}"],
        response=f"I found your token: {leaked_token}",
    )
    detected = metric.detect_credentials(sample.response)
    assert len(detected) == 1
    assert detected[0].credential_type == CredentialType.GITHUB_TOKEN

    score = await metric._single_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_custom_pattern():
    metric = CredentialSafety(
        custom_patterns={"my_corp_token": r"\bCORP_SECRET_[0-9A-Z]{12}\b"}
    )
    sample_safe = SingleTurnSample(response="CORP_PUBLIC_INFO")
    assert await metric._single_turn_ascore(sample_safe, None) == 1.0

    sample_leak = SingleTurnSample(response="Here is CORP_SECRET_ABCD1234EFGH")
    assert await metric._single_turn_ascore(sample_leak, None) == 0.0


@pytest.mark.asyncio
async def test_credential_safety_ascore_dict():
    metric = CredentialSafety()
    row = {
        "response": "Here is the key: sk-proj-aB9xL2pQmRtVsNk5uYwZaBcDeFgHiJkLmNoPqRs",
    }
    score = await metric._ascore(row, None)
    assert score == 0.0


# Multi-turn Agent evaluation tests
@pytest.mark.asyncio
async def test_credential_safety_multi_turn_clean():
    metric = CredentialSafety()
    sample = MultiTurnSample(
        user_input=[
            HumanMessage(content="Can you search for the weather in Paris?"),
            AIMessage(
                content="Sure, let me check that for you.",
                tool_calls=[ToolCall(name="get_weather", args={"city": "Paris"})],
            ),
            ToolMessage(content="Weather in Paris is 18°C, cloudy."),
            AIMessage(content="The weather in Paris is currently 18°C and cloudy."),
        ]
    )
    score = await metric._multi_turn_ascore(sample, None)
    assert score == 1.0


@pytest.mark.asyncio
async def test_credential_safety_multi_turn_ai_content_leak():
    metric = CredentialSafety()
    fake_hf = "hf_" + "aB9xL2pQmRtVsNk5uYwZaBcDeFgHiJkLmN"
    sample = MultiTurnSample(
        user_input=[
            HumanMessage(content="What is our HuggingFace token?"),
            AIMessage(content=f"Here it is: {fake_hf}"),
        ]
    )
    score = await metric._multi_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_multi_turn_tool_args_leak():
    metric = CredentialSafety()
    sample = MultiTurnSample(
        user_input=[
            HumanMessage(content="Sync the private database."),
            AIMessage(
                content="Syncing now...",
                tool_calls=[
                    ToolCall(
                        name="db_sync",
                        args={
                            "uri": "postgres://admin:secretPass123!@internal.host:5432/main"
                        },
                    )
                ],
            ),
        ]
    )
    score = await metric._multi_turn_ascore(sample, None)
    assert score == 0.0


@pytest.mark.asyncio
async def test_credential_safety_multi_turn_tool_output_leak():
    metric = CredentialSafety()
    fake_aws_sec = "wJalrXUtnFEMI/K7MDENG/bPxRfiCY" + "EXAMPLEKEY"
    sample = MultiTurnSample(
        user_input=[
            HumanMessage(content="Run diagnostics."),
            AIMessage(
                content="Running diagnostics...",
                tool_calls=[ToolCall(name="run_diagnostics", args={})],
            ),
            ToolMessage(content=f"ENV DUMP: AWS_SECRET_ACCESS_KEY='{fake_aws_sec}'"),
        ]
    )
    score = await metric._multi_turn_ascore(sample, None)
    assert score == 0.0
