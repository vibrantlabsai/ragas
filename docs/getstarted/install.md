# Installation

To get started, install Ragas using `pip` with the following command:

```bash
pip install ragas
```

If you'd like to experiment with the latest features, install the most recent version from the main branch:

```bash
pip install git+https://github.com/vibrantlabsai/ragas.git
```

If you're planning to contribute and make modifications to the code, ensure that you clone the repository and set it up as an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).

```bash
git clone https://github.com/vibrantlabsai/ragas.git 
pip install -e .
```

!!! note on "LangChain OpenAI dependency versions"
    If you use `langchain_openai` (e.g., `ChatOpenAI`), install `langchain-core` and `langchain-openai` explicitly to avoid version mismatches. You can adjust bounds to match your environment, but installing both explicitly helps prevent strict dependency conflicts.
    ```bash
    pip install -U "langchain-core>=0.2,<0.3" "langchain-openai>=0.1,<0.2" openai
    ```
