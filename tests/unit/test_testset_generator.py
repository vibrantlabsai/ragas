import asyncio

from ragas.prompt import PromptMixin
from ragas.testset.synthesizers.generate import (
    TestsetGenerator as RagasTestsetGenerator,
)
from ragas.testset.transforms import Parallel


class RecordingPromptMixin(PromptMixin):
    def __init__(self):
        self.adapt_calls = []
        self.adapted_prompts = {}

    async def adapt_prompts(self, language, llm, adapt_instruction=False):
        self.adapt_calls.append((language, llm))
        return {"recording_prompt": f"{language}-prompt"}

    def set_prompts(self, **prompts):
        self.adapted_prompts = prompts


def test_adapt_all_prompts_updates_synthesizers_and_nested_transforms():
    llm = object()
    synthesizer = RecordingPromptMixin()
    extractor = RecordingPromptMixin()
    nested_extractor = RecordingPromptMixin()
    transforms = [extractor, Parallel(Parallel(nested_extractor))]
    generator = RagasTestsetGenerator(llm=llm, embedding_model=object())

    asyncio.run(
        generator.adapt_all_prompts(
            language="uzbek",
            llm=llm,
            query_distribution=[(synthesizer, 1.0)],
            transforms=transforms,
        )
    )

    for component in (synthesizer, extractor, nested_extractor):
        assert component.adapt_calls == [("uzbek", llm)]
        assert component.adapted_prompts == {"recording_prompt": "uzbek-prompt"}


def test_adapt_all_prompts_deduplicates_reused_components():
    llm = object()
    component = RecordingPromptMixin()
    generator = RagasTestsetGenerator(llm=llm, embedding_model=object())

    asyncio.run(
        generator.adapt_all_prompts(
            language="chinese",
            llm=llm,
            query_distribution=[(component, 1.0)],
            transforms=[component],
        )
    )

    assert component.adapt_calls == [("chinese", llm)]
