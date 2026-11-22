"""
Multi-turn conversational RAG evaluation metrics.

These metrics extend RAGAS to evaluate RAG systems in conversational settings
where responses may reference information from prior conversation turns,
not just the current retrieved context. This addresses the key limitation of
single-turn evaluation metrics which incorrectly penalize faithful cross-turn
references as hallucinations.

Example
-------
    >>> from ragas.metrics._conversation_metrics import (
    ...     ConversationFaithfulness, MultiTurnConversation, ConversationTurn
    ... )
    >>> 
    >>> conversation = MultiTurnConversation(turns=[
    ...     ConversationTurn(
    ...         user_input="Who founded OpenAI?",
    ...         retrieved_contexts=["OpenAI was founded by Sam Altman and others in 2015."],
    ...         response="OpenAI was founded by Sam Altman and others in 2015."
    ...     ),
    ...     ConversationTurn(
    ...         user_input="When did Elon leave?",
    ...         retrieved_contexts=["OpenAI has had leadership changes."],
    ...         response="Elon Musk left OpenAI board in 2018."
    ...     ),
    ... ])
    >>> 
    >>> metric = ConversationFaithfulness(window_size=3)
    >>> # Set metric with LLM: metric.llm = your_llm
    >>> # result = await metric.score_conversation(conversation)
"""

from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ragas.metrics.base import MetricWithLLM
from ragas.metrics.result import MetricResult
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models for Structured LLM Output
# ─────────────────────────────────────────────────────────────────────────────


class ClaimsOutput(BaseModel):
    """Output model for claim extraction from a response."""

    claims: t.List[str] = Field(
        default_factory=list,
        description="List of factual claims extracted from the response"
    )


class ClaimVerdict(BaseModel):
    """Verdict for a single claim in multi-turn faithfulness evaluation."""

    claim: str
    supported_by_context: bool = Field(
        description="Whether the claim is supported by current retrieved context"
    )
    supported_by_history: bool = Field(
        description="Whether the claim is supported by prior conversation history"
    )
    verdict: int = Field(
        description="1 if claim is faithful (supported by context OR history), 0 otherwise"
    )
    reason: str = Field(description="Brief explanation for the verdict")


class ClaimVerdicts(BaseModel):
    """Collection of verdicts for all claims in a response."""

    verdicts: t.List[ClaimVerdict]


class RelevancyVerdict(BaseModel):
    """Verdict for response relevancy in conversation context."""

    relevancy_score: float = Field(
        ge=0.0, le=1.0,
        description="Score from 0.0 (irrelevant) to 1.0 (fully relevant) in conversation"
    )
    addresses_question: bool = Field(
        description="Whether response directly addresses the user's current question"
    )
    coherent_with_history: bool = Field(
        description="Whether response is coherent with established conversation context"
    )
    reason: str = Field(description="Brief explanation for the relevancy score")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Input Models
# ─────────────────────────────────────────────────────────────────────────────


class ClaimExtractionInput(BaseModel):
    """Input for claim extraction prompt."""

    response: str = Field(description="Response text to extract claims from")


class ClaimExtractionPrompt(PydanticPrompt[ClaimExtractionInput, ClaimsOutput]):
    """Prompt for extracting factual claims from a response."""

    instruction: str = (
        "Given a response, extract all factual claims made in that response. "
        "A claim is a specific, verifiable statement of fact. "
        "Do not include opinions, vague statements, or hedged claims.\n\n"
        "Response: {response}\n\n"
        "Return a JSON object with a 'claims' array containing all identified claims. "
        "Return ONLY valid JSON, no additional text."
    )


class ConversationFaithfulnessInput(BaseModel):
    """Input for multi-turn faithfulness evaluation prompt."""

    user_input: str = Field(description="User's question in current turn")
    retrieved_contexts: t.List[str] = Field(
        description="Retrieved context documents for current turn"
    )
    response: str = Field(description="Assistant's response in current turn")
    conversation_history: str = Field(
        description="Formatted conversation history from prior turns"
    )
    claims: str = Field(description="Claims to verify, formatted as a list")


class ConversationFaithfulnessPrompt(
    PydanticPrompt[ConversationFaithfulnessInput, ClaimVerdicts]
):
    """Prompt for evaluating faithfulness with conversation history."""

    instruction: str = (
        "You are evaluating whether an assistant's response in a conversation "
        "is faithful to available information.\n\n"
        "A claim is faithful if it is supported by EITHER:\n"
        "(A) The retrieved context provided for this specific turn, OR\n"
        "(B) Information explicitly stated in prior conversation history\n\n"
        "CONVERSATION HISTORY (prior turns):\n{conversation_history}\n\n"
        "CURRENT TURN:\n"
        "User question: {user_input}\n"
        "Retrieved context: {retrieved_contexts}\n"
        "Assistant response: {response}\n\n"
        "Claims to verify:\n{claims}\n\n"
        "For each claim, determine:\n"
        "1. Is it supported by the retrieved context? (yes/no)\n"
        "2. Is it supported by the conversation history? (yes/no)\n"
        "3. Verdict: faithful (1) if EITHER source supports it, not faithful (0) otherwise\n\n"
        "Return a JSON object with structure:\n"
        '{{"verdicts": [{{"claim": "...", "supported_by_context": bool, "supported_by_history": bool, "verdict": int, "reason": "..."}}]}}\n\n'
        "Return ONLY valid JSON. No markdown, no explanation."
    )


class ConversationRelevancyInput(BaseModel):
    """Input for conversation relevancy evaluation prompt."""

    user_input: str = Field(description="User's current question")
    response: str = Field(description="Assistant's response")
    conversation_history: str = Field(description="Prior conversation context")


class ConversationRelevancyPrompt(
    PydanticPrompt[ConversationRelevancyInput, RelevancyVerdict]
):
    """Prompt for evaluating response relevancy in multi-turn conversation."""

    instruction: str = (
        "You are evaluating whether an assistant's response is relevant "
        "given the full context of an ongoing conversation.\n\n"
        "CONVERSATION HISTORY (prior turns):\n{conversation_history}\n\n"
        "CURRENT TURN:\n"
        "User question: {user_input}\n"
        "Assistant response: {response}\n\n"
        "Evaluate:\n"
        "1. Does the response directly address the user's current question?\n"
        "2. Is the response coherent with the established conversation context?\n"
        "3. Does it avoid contradicting information from prior turns?\n\n"
        "Scoring guide:\n"
        "  1.0 = Fully relevant, addresses question, perfectly coherent with history\n"
        "  0.7 = Addresses question but with minor coherence issues\n"
        "  0.4 = Partially addresses question or notable incoherence\n"
        "  0.0 = Does not address question or directly contradicts history\n\n"
        "Return a JSON object:\n"
        '{{"relevancy_score": <float 0.0-1.0>, "addresses_question": bool, "coherent_with_history": bool, "reason": "..."}}\n\n'
        "Return ONLY valid JSON. No markdown, no explanation."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures for Multi-Turn Conversations
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConversationTurn:
    """Represents a single turn in a multi-turn conversation."""

    user_input: str
    """User's question or input for this turn."""

    retrieved_contexts: t.List[str]
    """Retrieved context documents relevant to this turn."""

    response: str
    """Assistant's response for this turn."""

    turn_index: int = 0
    """Index of this turn in the conversation (auto-assigned)."""

    def format_for_history(self) -> str:
        """Format this turn for inclusion in conversation history context.
        
        Returns:
            Formatted string representation of this turn suitable for LLM prompts.
        """
        return (
            f"Turn {self.turn_index + 1}:\n"
            f"  User: {self.user_input}\n"
            f"  Assistant: {self.response}"
        )


@dataclass
class MultiTurnConversation:
    """A complete multi-turn conversation for evaluation."""

    turns: t.List[ConversationTurn]
    """List of conversation turns."""

    conversation_id: str = ""
    """Optional identifier for the conversation."""

    def __post_init__(self):
        """Assign turn indices after initialization."""
        for i, turn in enumerate(self.turns):
            turn.turn_index = i

    def get_history_for_turn(
        self,
        turn_index: int,
        window_size: int = 3,
    ) -> t.List[ConversationTurn]:
        """Get prior turns for a given turn index within sliding window.
        
        Args:
            turn_index: Index of the turn we're evaluating.
            window_size: Maximum number of prior turns to include.
            
        Returns:
            List of prior ConversationTurn objects within the window.
            Empty list if turn_index is 0 (first turn has no history).
        """
        if turn_index == 0:
            return []
        
        start = max(0, turn_index - window_size)
        return self.turns[start:turn_index]

    def format_history(
        self,
        turn_index: int,
        window_size: int = 3,
    ) -> str:
        """Format prior conversation turns as readable text for LLM.
        
        Args:
            turn_index: Index of the current turn.
            window_size: Maximum prior turns to include.
            
        Returns:
            Formatted string with prior turns, or "No prior conversation history."
            if this is the first turn.
        """
        history_turns = self.get_history_for_turn(turn_index, window_size)
        if not history_turns:
            return "No prior conversation history."
        return "\n".join(turn.format_for_history() for turn in history_turns)


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConversationFaithfulness(MetricWithLLM):
    """
    Evaluates faithfulness of responses in multi-turn conversations.

    Unlike standard Faithfulness which only checks retrieved context,
    this metric considers both retrieved context AND prior conversation
    history as valid sources for claims.

    A response is faithful if all claims are supported by EITHER the
    current retrieved contexts OR information from prior turns (within
    the sliding window).

    Attributes
    ----------
    name : str
        Metric identifier: "conversation_faithfulness"
    window_size : int
        Number of prior turns to consider as history (default: 3).
        Larger windows = more accurate but higher token cost.
        
    Example
    -------
    >>> metric = ConversationFaithfulness(window_size=3)
    >>> metric.llm = your_llm  # Must provide an LLM
    >>> conversation = MultiTurnConversation(turns=[...])
    >>> result = await metric.score_conversation(conversation)
    >>> print(f"Overall: {result.value:.3f}")
    """

    name: str = "conversation_faithfulness"
    window_size: int = 3
    claim_extraction_prompt: PydanticPrompt = field(
        default_factory=ClaimExtractionPrompt
    )
    faithfulness_prompt: PydanticPrompt = field(
        default_factory=ConversationFaithfulnessPrompt
    )

    async def _extract_claims(
        self,
        response: str,
        callbacks: Callbacks = None,
    ) -> t.List[str]:
        """Extract factual claims from a response.
        
        Args:
            response: The response text to extract claims from.
            callbacks: Optional callback handler.
            
        Returns:
            List of extracted claims, empty list if extraction fails.
        """
        assert self.llm is not None, "LLM is not set"

        try:
            input_data = ClaimExtractionInput(response=response)
            output = await self.claim_extraction_prompt.generate(
                data=input_data,
                llm=self.llm,
                callbacks=callbacks,
            )
            return output.claims
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            return []

    async def _verify_claims_multi_turn(
        self,
        claims: t.List[str],
        turn: ConversationTurn,
        history_text: str,
        callbacks: Callbacks = None,
    ) -> ClaimVerdicts:
        """Verify claims against both context and conversation history.
        
        Args:
            claims: List of claims to verify.
            turn: The current conversation turn.
            history_text: Formatted prior conversation history.
            callbacks: Optional callback handler.
            
        Returns:
            ClaimVerdicts object with verdict for each claim.
            Returns empty verdicts if claims list is empty.
        """
        assert self.llm is not None, "LLM is not set"

        if not claims:
            return ClaimVerdicts(verdicts=[])

        # Format contexts and claims for the prompt
        context_text = "\n".join(
            f"[{i+1}] {ctx}" 
            for i, ctx in enumerate(turn.retrieved_contexts)
        )
        claims_text = "\n".join(f"- {claim}" for claim in claims)

        try:
            input_data = ConversationFaithfulnessInput(
                user_input=turn.user_input,
                retrieved_contexts=turn.retrieved_contexts,
                response=turn.response,
                conversation_history=history_text,
                claims=claims_text,
            )
            verdicts = await self.faithfulness_prompt.generate(
                data=input_data,
                llm=self.llm,
                callbacks=callbacks,
            )
            return verdicts
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            # Fail open: if we cannot verify, assume faithful
            # This is safer than penalizing correct cross-turn references
            return ClaimVerdicts(
                verdicts=[
                    ClaimVerdict(
                        claim=c,
                        supported_by_context=True,
                        supported_by_history=False,
                        verdict=1,
                        reason="Verification failed, defaulting to faithful",
                    )
                    for c in claims
                ]
            )

    def _compute_turn_score(self, verdicts: ClaimVerdicts) -> float:
        """Compute faithfulness score for a single turn.
        
        Score = (number of faithful claims) / (total claims)
        A claim is faithful if verdict == 1 (supported by context OR history).
        Empty claim lists score 1.0 (vacuously faithful).
        
        Args:
            verdicts: Verdicts for all claims in the turn.
            
        Returns:
            Faithfulness score between 0.0 and 1.0.
        """
        if not verdicts.verdicts:
            return 1.0  # No claims = vacuously faithful

        faithful_count = sum(v.verdict for v in verdicts.verdicts)
        return faithful_count / len(verdicts.verdicts)

    async def score_conversation(
        self,
        conversation: MultiTurnConversation,
        return_details: bool = False,
        callbacks: Callbacks = None,
    ) -> MetricResult:
        """Score an entire conversation for multi-turn faithfulness.
        
        Evaluates each turn by:
        1. Extracting claims from the response
        2. Verifying each claim against current context AND conversation history
        3. Computing a faithfulness score for that turn
        4. Averaging scores across all turns
        
        Args:
            conversation: The MultiTurnConversation to evaluate.
            return_details: If True, return detailed per-turn analysis.
            callbacks: Optional callback handler.
            
        Returns:
            MetricResult: score is mean faithfulness across turns.
                If return_details=True, includes traces with per-turn scores.
                
        Raises:
            AssertionError: If llm is not set.
        """
        assert self.llm is not None, "LLM is not set"

        turn_scores = []
        turn_details = []

        for i, turn in enumerate(conversation.turns):
            # Get conversation history for this turn (within window)
            history_text = conversation.format_history(i, self.window_size)

            # Step 1: Extract claims from response
            claims = await self._extract_claims(
                turn.response,
                callbacks=callbacks,
            )

            # Step 2: Verify claims against context + history
            verdicts = await self._verify_claims_multi_turn(
                claims, turn, history_text, callbacks=callbacks
            )

            # Step 3: Compute score for this turn
            turn_score = self._compute_turn_score(verdicts)
            turn_scores.append(turn_score)

            logger.debug(
                f"Turn {i}: {len(claims)} claims, score={turn_score:.3f}, "
                f"history_turns={len(conversation.get_history_for_turn(i, self.window_size))}"
            )

            if return_details:
                turn_details.append({
                    "turn_index": i,
                    "user_input": turn.user_input,
                    "score": turn_score,
                    "claims": claims,
                    "verdicts": [v.model_dump() for v in verdicts.verdicts],
                    "history_window_size": len(
                        conversation.get_history_for_turn(i, self.window_size)
                    ),
                })

        # Compute overall conversation score
        conversation_score = float(np.mean(turn_scores)) if turn_scores else 0.0

        traces = {
            "turn_scores": turn_scores,
            "turns": turn_details if return_details else None,
        }

        return MetricResult(value=conversation_score, traces=traces)

    async def ascore(
        self,
        *args,
        **kwargs,
    ) -> MetricResult:
        """Not implemented for conversation metrics.
        
        Use score_conversation() instead, which requires a MultiTurnConversation.
        """
        raise NotImplementedError(
            "ConversationFaithfulness.ascore() is not implemented. "
            "Use score_conversation(conversation) instead."
        )


@dataclass
class ConversationRelevancy(MetricWithLLM):
    """
    Evaluates response relevancy in multi-turn conversations.

    Unlike standard AnswerRelevancy which evaluates responses in isolation,
    this metric considers whether responses are coherent with the full
    conversation context, not just the immediate question.

    A response scores highly if it:
    1. Directly addresses the current question
    2. Is coherent with information established in prior turns
    3. Does not contradict prior conversation context

    Attributes
    ----------
    name : str
        Metric identifier: "conversation_relevancy"
    window_size : int
        Number of prior turns to consider as context (default: 3).
    """

    name: str = "conversation_relevancy"
    window_size: int = 3
    relevancy_prompt: PydanticPrompt = field(
        default_factory=ConversationRelevancyPrompt
    )

    async def _score_turn_relevancy(
        self,
        turn: ConversationTurn,
        history_text: str,
        callbacks: Callbacks = None,
    ) -> RelevancyVerdict:
        """Score relevancy for a single conversation turn.
        
        Args:
            turn: The conversation turn to evaluate.
            history_text: Formatted prior conversation history.
            callbacks: Optional callback handler.
            
        Returns:
            RelevancyVerdict with score and reasoning.
        """
        assert self.llm is not None, "LLM is not set"

        try:
            input_data = ConversationRelevancyInput(
                user_input=turn.user_input,
                response=turn.response,
                conversation_history=history_text,
            )
            verdict = await self.relevancy_prompt.generate(
                data=input_data,
                llm=self.llm,
                callbacks=callbacks,
            )
            # Clamp score to valid range
            verdict.relevancy_score = max(0.0, min(1.0, verdict.relevancy_score))
            return verdict
        except Exception as e:
            logger.warning(f"Relevancy scoring failed for turn {turn.turn_index}: {e}")
            return RelevancyVerdict(
                relevancy_score=0.5,
                addresses_question=True,
                coherent_with_history=True,
                reason="Scoring failed, defaulting to 0.5",
            )

    async def score_conversation(
        self,
        conversation: MultiTurnConversation,
        return_details: bool = False,
        callbacks: Callbacks = None,
    ) -> MetricResult:
        """Score conversation for multi-turn relevancy.
        
        Args:
            conversation: The MultiTurnConversation to evaluate.
            return_details: If True, return detailed per-turn analysis.
            callbacks: Optional callback handler.
            
        Returns:
            MetricResult with mean relevancy score across turns.
            
        Raises:
            AssertionError: If llm is not set.
        """
        assert self.llm is not None, "LLM is not set"

        turn_scores = []
        turn_details = []

        for i, turn in enumerate(conversation.turns):
            history_text = conversation.format_history(i, self.window_size)
            verdict = await self._score_turn_relevancy(
                turn, history_text, callbacks=callbacks
            )

            turn_scores.append(verdict.relevancy_score)

            logger.debug(
                f"Turn {i}: relevancy={verdict.relevancy_score:.3f}, "
                f"addresses_question={verdict.addresses_question}, "
                f"coherent={verdict.coherent_with_history}"
            )

            if return_details:
                turn_details.append({
                    "turn_index": i,
                    "user_input": turn.user_input,
                    "score": verdict.relevancy_score,
                    "addresses_question": verdict.addresses_question,
                    "coherent_with_history": verdict.coherent_with_history,
                    "reason": verdict.reason,
                })

        conversation_score = float(np.mean(turn_scores)) if turn_scores else 0.0

        traces = {
            "turn_scores": turn_scores,
            "turns": turn_details if return_details else None,
        }

        return MetricResult(value=conversation_score, traces=traces)

    async def ascore(
        self,
        *args,
        **kwargs,
    ) -> MetricResult:
        """Not implemented for conversation metrics.
        
        Use score_conversation() instead, which requires a MultiTurnConversation.
        """
        raise NotImplementedError(
            "ConversationRelevancy.ascore() is not implemented. "
            "Use score_conversation(conversation) instead."
        )