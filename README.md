<h1 align="center">
  <img style="vertical-align:middle" height="200"
  src="https://raw.githubusercontent.com/vibrantlabsai/ragas/main/docs/_static/imgs/logo.png">
</h1>
<p align="center">
  <i>Supercharge Your LLM Application Evaluations 🚀</i>
</p>

<p align="center">
    <a href="https://github.com/vibrantlabsai/ragas/releases">
        <img alt="Latest release" src="https://img.shields.io/github/release/vibrantlabsai/ragas.svg">
    </a>
    <a href="https://www.python.org/">
        <img alt="Made with Python" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/vibrantlabsai/ragas/blob/master/LICENSE">
        <img alt="License Apache-2.0" src="https://img.shields.io/github/license/vibrantlabsai/ragas.svg?color=green">
    </a>
    <a href="https://pypi.org/project/ragas/">
        <img alt="Ragas Downloads per month" src="https://static.pepy.tech/badge/ragas/month">
    </a>
    <a href="https://discord.gg/5djav8GGNZ">
        <img alt="Join Ragas community on Discord" src="https://img.shields.io/discord/1119637219561451644">
    </a>
    <a target="_blank" href="https://deepwiki.com/vibrantlabsai/ragas">
      <img 
        src="https://devin.ai/assets/deepwiki-badge.png" 
        alt="Ask DeepWiki.com" 
        height="20" 
      />
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://docs.ragas.io/">Documentation</a> |
        <a href="#fire-quickstart">Quick start</a> |
        <a href="https://discord.gg/5djav8GGNZ">Join Discord</a> |
        <a href="https://blog.ragas.io/">Blog</a> |
        <a href="https://newsletter.ragas.io/">NewsLetter</a> |
        <a href="https://www.ragas.io/careers">Careers</a>
    <p>
</h4>

Objective metrics, intelligent test generation, and data-driven insights for LLM apps

Ragas is your ultimate toolkit for evaluating and optimizing Large Language Model (LLM) applications. Say goodbye to time-consuming, subjective assessments and hello to data-driven, efficient evaluation workflows.
Don't have a test dataset ready? We also do production-aligned test set generation.

> [!NOTE]
> Need help setting up Evals for your AI application? We'd love to help! We are conducting Office Hours every week. You can sign up [here](https://cal.com/team/vibrantlabs/office-hours).

## Key Features

- 🎯 Objective Metrics: Evaluate your LLM applications with precision using both LLM-based and traditional metrics.
- 🧪 Test Data Generation: Automatically create comprehensive test datasets covering a wide range of scenarios.
- 🔗 Seamless Integrations: Works flawlessly with popular LLM frameworks like LangChain and major observability tools.
- 📊 Build feedback loops: Leverage production data to continually improve your LLM applications.

## :shield: Installation

Pypi:

```bash
pip install ragas
```

Alternatively, from source:

```bash
pip install git+https://github.com/vibrantlabsai/ragas
```

## :fire: Quickstart

### Clone a Complete Example Project

The fastest way to get started is to use the `ragas quickstart` command:

```bash
# List available templates
ragas quickstart

# Create a RAG evaluation project
ragas quickstart rag_eval

# Specify where you want to create it.
ragas quickstart rag_eval -o ./my-project
```

Available templates:
- `rag_eval` - Evaluate RAG systems

Coming Soon:
- `agent_evals` - Evaluate AI agents
- `benchmark_llm` - Benchmark and compare LLMs
- `prompt_evals` - Evaluate prompt variations
- `workflow_eval` - Evaluate complex workflows

### Evaluate your LLM App

This is a simple example evaluating a summary for accuracy:

```python
import asyncio
from ragas.metrics.collections import AspectCritic
from ragas.llms import llm_factory

# Setup your LLM
llm = llm_factory("gpt-4o")

# Create a metric
metric = AspectCritic(
    name="summary_accuracy",
    definition="Verify if the summary is accurate and captures key information.",
    llm=llm
)

# Evaluate
test_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
}

score = await metric.ascore(
    user_input=test_data["user_input"],
    response=test_data["response"]
)
print(f"Score: {score.value}")
print(f"Reason: {score.reason}")
```

> **Note**: Make sure your `OPENAI_API_KEY` environment variable is set.

Find the complete [Quickstart Guide](https://docs.ragas.io/en/latest/getstarted/evals)

## Want help in improving your AI application using evals?

In the past 2 years, we have seen and helped improve many AI applications using evals. If you want help with improving and scaling up your AI application using evals.

🔗 Book a [slot](https://cal.com/team/vibrantlabs/app) or drop us a line: [founders@vibrantlabs.com](mailto:founders@vibrantlabs.com).

## 🫂 Community

If you want to get more involved with Ragas, check out our [discord server](https://discord.gg/5qGUJ6mh7C). It's a fun community where we geek out about LLM, Retrieval, Production issues, and more.

## Contributors

```yml
+----------------------------------------------------------------------------+
|     +----------------------------------------------------------------+     |
|     | Developers: Those who built with `ragas`.                      |     |
|     | (You have `import ragas` somewhere in your project)            |     |
|     |     +----------------------------------------------------+     |     |
|     |     | Contributors: Those who make `ragas` better.       |     |     |
|     |     | (You make PR to this repo)                         |     |     |
|     |     +----------------------------------------------------+     |     |
|     +----------------------------------------------------------------+     |
+----------------------------------------------------------------------------+
```

We welcome contributions from the community! Whether it's bug fixes, feature additions, or documentation improvements, your input is valuable.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 🔍 Open Analytics

At Ragas, we believe in transparency. We collect minimal, anonymized usage data to improve our product and guide our development efforts.

✅ No personal or company-identifying information

✅ Open-source data collection [code](./src/ragas/_analytics.py)

✅ Publicly available aggregated [data](https://github.com/vibrantlabsai/ragas/issues/49)

To opt-out, set the `RAGAS_DO_NOT_TRACK` environment variable to `true`.

### Cite Us

```
@misc{ragas2024,
  author       = {VibrantLabs},
  title        = {Ragas: Supercharge Your LLM Application Evaluations},
  year         = {2024},
  howpublished = {\url{https://github.com/vibrantlabsai/ragas}},
}
```
