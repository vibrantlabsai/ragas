# Create custom multi-hop queries from your documents

In this tutorial you will get to learn how to create custom multi-hop queries from your documents. This is a very powerful feature that allows you to create queries that are not possible with the standard query types. This also helps you to create queries that are more specific to your use case.

### Load sample documents
I am using documents from [sample of GitLab handbook](https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown). You can download it by running the below command.


```python
! git clone https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown
```


```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

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

### Setup Extractors and Relationship builders

To create multi-hop queries you need to understand the set of documents that can be used for it. Ragas uses relationships between documents/nodes to quality nodes for creating multi-hop queries. To concretize, if Node A and Node B are connected by a relationship (say entity or keyphrase overlap) then you can create a multi-hop query between them.

Here we are using 2 extractors and 2 relationship builders.
- Headline extractor: Extracts headlines from the documents
- Keyphrase extractor: Extracts keyphrases from the documents
- Headline splitter: Splits the document into nodes based on headlines
- OverlapScore Builder: Builds relationship between nodes based on keyphrase overlap


```python
from ragas.testset.transforms import Parallel, apply_transforms
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
    OverlapScoreBuilder,
)


headline_extractor = HeadlinesExtractor(llm=llm)
headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
keyphrase_extractor = KeyphrasesExtractor(
    llm=llm, property_name="keyphrases", max_num=10
)
relation_builder = OverlapScoreBuilder(
    property_name="keyphrases",
    new_property_name="overlap_score",
    threshold=0.01,
    distance_threshold=0.9,
)

transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor,
    relation_builder,
]

apply_transforms(kg, transforms=transforms)
```
Output
```
Applying KeyphrasesExtractor:   6%|██████▏                                                                                                         | 2/36 [00:01<00:17,  1.94it/s]Property 'keyphrases' already exists in node 'a2f389'. Skipping!
Applying KeyphrasesExtractor:  17%|██████████████████▋                                                                                             | 6/36 [00:01<00:04,  6.37it/s]Property 'keyphrases' already exists in node '3068c0'. Skipping!
Applying KeyphrasesExtractor:  53%|██████████████████████████████████████████████████████████▌                                                    | 19/36 [00:02<00:01,  8.88it/s]Property 'keyphrases' already exists in node '854bf7'. Skipping!
Applying KeyphrasesExtractor:  78%|██████████████████████████████████████████████████████████████████████████████████████▎                        | 28/36 [00:03<00:00,  9.73it/s]Property 'keyphrases' already exists in node '2eeb07'. Skipping!
Property 'keyphrases' already exists in node 'd68f83'. Skipping!
Applying KeyphrasesExtractor:  83%|████████████████████████████████████████████████████████████████████████████████████████████▌                  | 30/36 [00:03<00:00,  9.35it/s]Property 'keyphrases' already exists in node '8fdbea'. Skipping!
Applying KeyphrasesExtractor:  89%|██████████████████████████████████████████████████████████████████████████████████████████████████▋            | 32/36 [00:04<00:00,  7.76it/s]Property 'keyphrases' already exists in node 'ef6ae0'. Skipping!
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

### Create multi-hop query

Inherit from `MultiHopQuerySynthesizer` and modify the function that generates scenarios for query creation.

**Steps**:
- find qualified set of (nodeA, relationship, nodeB) based on the relationships between nodes
- For each qualified set
    - Match the keyphrase with one or more persona.
    - Create all possible combinations of (Nodes, Persona, Query Style, Query Length)
    - Samples the required number of queries from the combinations



```python
from dataclasses import dataclass
import typing as t
from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)


@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):

    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph,
        persona_list,
        callbacks,
    ) -> t.List[MultiHopScenario]:

        # query and get (node_a, rel, node_b) to create multi-hop queries
        results = kg.find_two_nodes_single_rel(
            relationship_condition=lambda rel: (
                True if rel.type == "keyphrases_overlap" else False
            )
        )

        num_sample_per_triplet = max(1, n // len(results))

        scenarios = []
        for triplet in results:
            if len(scenarios) < n:
                node_a, node_b = triplet[0], triplet[-1]
                overlapped_keywords = triplet[1].properties["overlapped_items"]
                if overlapped_keywords:

                    # match the keyword with a persona for query creation
                    themes = list(dict(overlapped_keywords).keys())
                    prompt_input = ThemesPersonasInput(
                        themes=themes, personas=persona_list
                    )
                    persona_concepts = (
                        await self.theme_persona_matching_prompt.generate(
                            data=prompt_input, llm=self.llm, callbacks=callbacks
                        )
                    )

                    overlapped_keywords = [list(item) for item in overlapped_keywords]

                    # prepare and sample possible combinations
                    base_scenarios = self.prepare_combinations(
                        [node_a, node_b],
                        overlapped_keywords,
                        personas=persona_list,
                        persona_item_mapping=persona_concepts.mapping,
                        property_name="keyphrases",
                    )

                    # get number of required samples from this triplet
                    base_scenarios = self.sample_diverse_combinations(
                        base_scenarios, num_sample_per_triplet
                    )

                    scenarios.extend(base_scenarios)

        return scenarios

query = MyMultiHopQuery(llm=llm)
scenarios = await query.generate_scenarios(
    n=10, knowledge_graph=kg, persona_list=persona_list
)

scenarios[4]
```
Output
```
MultiHopScenario(
nodes=2
combinations=['Diversity Inclusion & Belonging', 'Diversity, Inclusion & Belonging Goals']
style=Web search like queries
length=short
persona=name='Hiring manager at gitlab' role_description='A hiring manager at gitlab trying to underestand hiring policies in gitlab')
```

### Run the multi-hop query


```python
result = await query.generate_sample(scenario=scenarios[-1])
result.user_input
```

Output
```
'How does GitLab ensure that its DIB roundtables are effective in promoting diversity and inclusion?'
```


Yay! You have created a multi-hop query. Now you can create any such queries by creating and exploring relationships between documents.

##
