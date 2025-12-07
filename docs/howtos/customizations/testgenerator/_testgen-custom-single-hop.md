# Create custom single-hop queries from your documents

### Load sample documents
I am using documents from [sample of GitLab handbook](https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown). You can download it by running the below command.

```
! git clone https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown

```

```python
from langchain_community.document_loaders import DirectoryLoader


path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
```

### Create KG

Create a base knowledge graph with the documents


```python
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType


kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )
```

### Set up the LLM and Embedding Model
You may use any of [your choice](./../../customizations/customize_models.md), here I am using models from open-ai.

```python
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

openai_client = OpenAI()
llm = llm_factory("gpt-4o-mini", client=openai_client)
embedding = OpenAIEmbeddings(client=openai_client)
```

### Setup the transforms


Here we are using 2 extractors and 2 relationship builders.
- Headline extractor: Extracts headlines from the documents
- Keyphrase extractor: Extracts keyphrases from the documents
- Headline splitter: Splits the document into nodes based on headlines



```python
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
)


headline_extractor = HeadlinesExtractor(llm=llm)
headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
keyphrase_extractor = KeyphrasesExtractor(
    llm=llm, property_name="keyphrases", max_num=10
)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor,
]

apply_transforms(kg, transforms=transforms)
```

Output
```
Applying KeyphrasesExtractor:   6%| | 2/36 [00:01<00:20,  1Property 'keyphrases' already exists in node '514fdc'. Skipping!
Applying KeyphrasesExtractor:  11%| | 4/36 [00:01<00:10,  2Property 'keyphrases' already exists in node '84a0f6'. Skipping!
Applying KeyphrasesExtractor:  64%|▋| 23/36 [00:03<00:01,  Property 'keyphrases' already exists in node '93f19d'. Skipping!
Applying KeyphrasesExtractor:  72%|▋| 26/36 [00:04<00:00, 1Property 'keyphrases' already exists in node 'a126bf'. Skipping!
Applying KeyphrasesExtractor:  81%|▊| 29/36 [00:04<00:00,  Property 'keyphrases' already exists in node 'c230df'. Skipping!
Applying KeyphrasesExtractor:  89%|▉| 32/36 [00:04<00:00, 1Property 'keyphrases' already exists in node '4f2765'. Skipping!
Property 'keyphrases' already exists in node '4a4777'. Skipping!
```

### Configure personas

You can also do this automatically by using the [automatic persona generator](./_persona_generator.md)


```python
from ragas.testset.persona import Persona

person1 = Persona(
    name="gitlab employee",
    role_description="A junior gitlab employee curious on workings on gitlab",
)
persona2 = Persona(
    name="Hiring manager at gitlab",
    role_description="A hiring manager at gitlab trying to underestand hiring policies in gitlab",
)
persona_list = [person1, persona2]
```

##

## SingleHop Query

Inherit from `SingleHopQuerySynthesizer` and modify the function that generates scenarios for query creation.

**Steps**:
- find qualified set of nodes for the query creation. Here I am selecting all nodes with keyphrases extracted.
- For each qualified set
    - Match the keyphrase with one or more persona.
    - Create all possible combinations of (Node, Persona, Query Style, Query Length)
    - Samples the required number of queries from the combinations


```python
from ragas.testset.synthesizers.single_hop import (
    SingleHopQuerySynthesizer,
    SingleHopScenario,
)
from dataclasses import dataclass
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)


@dataclass
class MySingleHopScenario(SingleHopQuerySynthesizer):

    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(self, n, knowledge_graph, persona_list, callbacks):

        property_name = "keyphrases"
        nodes = []
        for node in knowledge_graph.nodes:
            if node.type.name == "CHUNK" and node.get_property(property_name):
                nodes.append(node)

        number_of_samples_per_node = max(1, n // len(nodes))

        scenarios = []
        for node in nodes:
            if len(scenarios) >= n:
                break
            themes = node.properties.get(property_name, [""])
            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            base_scenarios = self.prepare_combinations(
                node,
                themes,
                personas=persona_list,
                persona_concepts=persona_concepts.mapping,
            )
            scenarios.extend(
                self.sample_combinations(base_scenarios, number_of_samples_per_node)
            )

        return scenarios

query = MySingleHopScenario(llm=llm)

scenarios = await query.generate_scenarios(
    n=5, knowledge_graph=kg, persona_list=persona_list
)

scenarios[0]
```
Output
```
SingleHopScenario(
nodes=1
term=what is an ally
persona=name='Hiring manager at gitlab' role_description='A hiring manager at gitlab trying to underestand hiring policies in gitlab'
style=Web search like queries
length=long)
```



```python
result = await query.generate_sample(scenario=scenarios[-1])
```

### Modify prompt to customize the query style
Here I am replacing the default prompt with an instruction to generate only Yes/No questions. This is an optional step.


```python
instruction = """Generate a Yes/No query and answer based on the specified conditions (persona, term, style, length)
and the provided context. Ensure the answer is entirely faithful to the context, using only the information
directly from the provided context.

### Instructions:
1. **Generate a Yes/No Query**: Based on the context, persona, term, style, and length, create a question
that aligns with the persona's perspective, incorporates the term, and can be answered with 'Yes' or 'No'.
2. **Generate an Answer**: Using only the content from the provided context, provide a 'Yes' or 'No' answer
to the query. Do not add any information not included in or inferable from the context."""
```


```python
prompt = query.get_prompts()["generate_query_reference_prompt"]
prompt.instruction = instruction
query.set_prompts(**{"generate_query_reference_prompt": prompt})
result = await query.generate_sample(scenario=scenarios[-1])
```


```python
result.user_input
```
Output
```
'Does the Diversity, Inclusion & Belonging (DIB) Team at GitLab have a structured approach to encourage collaborations among team members through various communication methods?'
```

```python
result.reference
```
Output
```
'Yes'
```
