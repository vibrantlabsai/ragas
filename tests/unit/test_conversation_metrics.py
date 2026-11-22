"""
Unit tests for multi-turn conversation evaluation metrics.
All tests are deterministic — no API calls, no LLM required.
"""

import pytest
from ragas.metrics._conversation_metrics import (
    ConversationFaithfulness,
    ConversationRelevancy,
    MultiTurnConversation,
    ConversationTurn,
    ClaimVerdict,
    ClaimVerdicts,
    RelevancyVerdict,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_conversation():
    """Two-turn conversation where turn 2 references turn 1 content."""
    return MultiTurnConversation(
        conversation_id="test-001",
        turns=[
            ConversationTurn(
                user_input="Who founded OpenAI?",
                retrieved_contexts=[
                    "OpenAI was founded in 2015 by Sam Altman, Greg Brockman, "
                    "Elon Musk, and others."
                ],
                response="OpenAI was founded by Sam Altman, Greg Brockman, "
                         "and Elon Musk in 2015."
            ),
            ConversationTurn(
                user_input="When did he leave the board?",
                retrieved_contexts=[
                    "OpenAI experienced significant leadership changes in 2023."
                ],
                response="Elon Musk left the OpenAI board in 2018."
            ),
        ]
    )


@pytest.fixture
def three_turn_conversation():
    """Three-turn conversation for sliding window tests."""
    return MultiTurnConversation(
        turns=[
            ConversationTurn(
                user_input="What is the capital of France?",
                retrieved_contexts=["Paris is the capital of France."],
                response="The capital of France is Paris."
            ),
            ConversationTurn(
                user_input="What is its population?",
                retrieved_contexts=["Paris has a population of about 2.1 million."],
                response="Paris has a population of about 2.1 million people."
            ),
            ConversationTurn(
                user_input="What famous tower is there?",
                retrieved_contexts=["The Eiffel Tower is in Paris, built in 1889."],
                response="The Eiffel Tower is in Paris, built in 1889."
            ),
        ]
    )


@pytest.fixture
def single_turn_conversation():
    """Single-turn conversation — edge case."""
    return MultiTurnConversation(
        turns=[
            ConversationTurn(
                user_input="What is 2 + 2?",
                retrieved_contexts=["Basic arithmetic: 2 + 2 = 4."],
                response="2 + 2 equals 4."
            )
        ]
    )


# ── ConversationTurn tests ────────────────────────────────────────────────────

class TestConversationTurn:

    def test_format_for_history_contains_user_input(self):
        turn = ConversationTurn(
            user_input="What is AI?",
            retrieved_contexts=["AI is artificial intelligence."],
            response="AI stands for artificial intelligence.",
            turn_index=0
        )
        formatted = turn.format_for_history()
        assert "What is AI?" in formatted

    def test_format_for_history_contains_response(self):
        turn = ConversationTurn(
            user_input="What is AI?",
            retrieved_contexts=[],
            response="AI stands for artificial intelligence.",
            turn_index=0
        )
        assert "AI stands for artificial intelligence." in turn.format_for_history()

    def test_format_for_history_correct_turn_number(self):
        turn = ConversationTurn(
            user_input="test", retrieved_contexts=[], response="test", turn_index=2
        )
        assert "Turn 3" in turn.format_for_history()

    def test_default_turn_index_is_zero(self):
        turn = ConversationTurn(user_input="q", retrieved_contexts=[], response="a")
        assert turn.turn_index == 0

    def test_empty_contexts_allowed(self):
        turn = ConversationTurn(user_input="q", retrieved_contexts=[], response="a")
        assert turn.retrieved_contexts == []


# ── MultiTurnConversation tests ───────────────────────────────────────────────

class TestMultiTurnConversation:

    def test_turn_indices_assigned_on_init(self, three_turn_conversation):
        for i, turn in enumerate(three_turn_conversation.turns):
            assert turn.turn_index == i

    def test_history_empty_for_first_turn(self, simple_conversation):
        history = simple_conversation.get_history_for_turn(0)
        assert history == []

    def test_history_one_turn_for_second_turn(self, simple_conversation):
        history = simple_conversation.get_history_for_turn(1)
        assert len(history) == 1
        assert history[0].user_input == "Who founded OpenAI?"

    def test_history_respects_window_size(self, three_turn_conversation):
        history = three_turn_conversation.get_history_for_turn(2, window_size=1)
        assert len(history) == 1
        assert history[0].turn_index == 1

    def test_window_size_zero_returns_empty(self, three_turn_conversation):
        history = three_turn_conversation.get_history_for_turn(2, window_size=0)
        assert history == []

    def test_window_size_larger_than_history(self, three_turn_conversation):
        history = three_turn_conversation.get_history_for_turn(2, window_size=10)
        assert len(history) == 2

    def test_format_history_no_prior(self, simple_conversation):
        formatted = simple_conversation.format_history(0)
        assert "No prior conversation history" in formatted

    def test_format_history_contains_prior_user_input(self, simple_conversation):
        formatted = simple_conversation.format_history(1)
        assert "Who founded OpenAI?" in formatted

    def test_format_history_contains_prior_response(self, simple_conversation):
        formatted = simple_conversation.format_history(1)
        assert "Sam Altman" in formatted

    def test_conversation_id_default_empty(self, three_turn_conversation):
        assert three_turn_conversation.conversation_id == ""

    def test_conversation_id_set(self, simple_conversation):
        assert simple_conversation.conversation_id == "test-001"

    def test_single_turn_has_no_history(self, single_turn_conversation):
        assert single_turn_conversation.get_history_for_turn(0) == []


# ── ConversationFaithfulness score computation tests ─────────────────────────

class TestConversationFaithfulness:

    def test_all_faithful_scores_one(self):
        metric = ConversationFaithfulness()
        verdicts = ClaimVerdicts(verdicts=[
            ClaimVerdict(claim="c1", supported_by_context=True,
                         supported_by_history=False, verdict=1, reason="ok"),
            ClaimVerdict(claim="c2", supported_by_context=False,
                         supported_by_history=True, verdict=1, reason="ok"),
        ])
        assert metric._compute_turn_score(verdicts) == 1.0

    def test_no_faithful_scores_zero(self):
        metric = ConversationFaithfulness()
        verdicts = ClaimVerdicts(verdicts=[
            ClaimVerdict(claim="c1", supported_by_context=False,
                         supported_by_history=False, verdict=0, reason="not found"),
        ])
        assert metric._compute_turn_score(verdicts) == 0.0

    def test_half_faithful_scores_half(self):
        metric = ConversationFaithfulness()
        verdicts = ClaimVerdicts(verdicts=[
            ClaimVerdict(claim="c1", supported_by_context=True,
                         supported_by_history=False, verdict=1, reason="ok"),
            ClaimVerdict(claim="c2", supported_by_context=False,
                         supported_by_history=False, verdict=0, reason="not found"),
        ])
        assert metric._compute_turn_score(verdicts) == 0.5

    def test_empty_verdicts_scores_one(self):
        """No claims = vacuously faithful."""
        metric = ConversationFaithfulness()
        assert metric._compute_turn_score(ClaimVerdicts(verdicts=[])) == 1.0

    def test_default_window_size_is_three(self):
        assert ConversationFaithfulness().window_size == 3

    def test_custom_window_size(self):
        assert ConversationFaithfulness(window_size=5).window_size == 5

    def test_default_name(self):
        assert ConversationFaithfulness().name == "conversation_faithfulness"

    def test_three_of_four_faithful(self):
        metric = ConversationFaithfulness()
        verdicts = ClaimVerdicts(verdicts=[
            ClaimVerdict(claim="c1", supported_by_context=True,
                         supported_by_history=False, verdict=1, reason="ok"),
            ClaimVerdict(claim="c2", supported_by_context=True,
                         supported_by_history=False, verdict=1, reason="ok"),
            ClaimVerdict(claim="c3", supported_by_context=False,
                         supported_by_history=True, verdict=1, reason="ok"),
            ClaimVerdict(claim="c4", supported_by_context=False,
                         supported_by_history=False, verdict=0, reason="not found"),
        ])
        assert metric._compute_turn_score(verdicts) == 0.75


# ── ConversationRelevancy tests ───────────────────────────────────────────────

class TestConversationRelevancy:

    def test_default_window_size_is_three(self):
        assert ConversationRelevancy().window_size == 3

    def test_custom_window_size(self):
        assert ConversationRelevancy(window_size=2).window_size == 2

    def test_default_name(self):
        assert ConversationRelevancy().name == "conversation_relevancy"


# ── Import tests ──────────────────────────────────────────────────────────────

class TestImports:

    def test_import_from_metrics_init(self):
        from ragas.metrics import (
            ConversationFaithfulness,
            ConversationRelevancy,
            MultiTurnConversation,
            ConversationTurn,
        )
        assert ConversationFaithfulness is not None
        assert ConversationRelevancy is not None
        assert MultiTurnConversation is not None
        assert ConversationTurn is not None

    def test_conversation_faithfulness_instantiates(self):
        from ragas.metrics import ConversationFaithfulness
        m = ConversationFaithfulness()
        assert m.name == "conversation_faithfulness"

    def test_conversation_relevancy_instantiates(self):
        from ragas.metrics import ConversationRelevancy
        m = ConversationRelevancy()
        assert m.name == "conversation_relevancy"

    def test_multi_turn_conversation_instantiates(self):
        from ragas.metrics import MultiTurnConversation, ConversationTurn
        c = MultiTurnConversation(turns=[
            ConversationTurn(user_input="q", retrieved_contexts=[], response="a")
        ])
        assert len(c.turns) == 1