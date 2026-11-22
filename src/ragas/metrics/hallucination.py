from ragas.metrics.base import Metric


class HallucinationMetric(Metric):
    name = "hallucination_rate"

    async def _ascore(self, llm, question, context, response):

        prompt = f"""
        Context:
        {context}

        Response:
        {response}

        Identify parts of the response that are not supported by the context.
        Return a hallucination score between 0 and 1.
        """

        result = await llm.generate(prompt)

        return float(result)