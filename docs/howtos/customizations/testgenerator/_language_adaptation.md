## Synthetic test generation from non-English corpus

In this notebook, you'll learn how to adapt synthetic test data generation to non-English corpus settings. For the sake of this tutorial, I am generating queries in Spanish from Spanish Wikipedia articles.

### Download and Load corpus


```python
! git clone https://huggingface.co/datasets/vibrantlabsai/Sample_non_english_corpus
```

    Cloning into 'Sample_non_english_corpus'...
    remote: Enumerating objects: 12, done.[K
    remote: Counting objects: 100% (8/8), done.[K
    remote: Compressing objects: 100% (8/8), done.[K
    remote: Total 12 (delta 0), reused 0 (delta 0), pack-reused 4 (from 1)[K
    Unpacking objects: 100% (12/12), 11.43 KiB | 780.00 KiB/s, done.



```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader


path = "Sample_non_english_corpus/"
loader = DirectoryLoader(path, glob="**/*.txt")
docs = loader.load()
```

    /opt/homebrew/Caskroom/miniforge/base/envs/ragas/lib/python3.9/site-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.20) or chardet (5.2.0)/charset_normalizer (None) doesn't match a supported version!
      warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "



```python
len(docs)
```




    6



### Initialize required models


```python
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import openai

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
openai_client = openai.OpenAI()
generator_embeddings = OpenAIEmbeddings(client=openai_client)
```

    /opt/homebrew/Caskroom/miniforge/base/envs/ragas/lib/python3.9/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


### Setup Persona and transforms
you may automatically create personas using this [notebook](./_persona_generator.md). For the sake of simplicity, I am using a pre-defined person, two basic transforms and simple query distribution.


```python
from ragas.testset.persona import Persona

personas = [
    Persona(
        name="curious student",
        role_description="A student who is curious about the world and wants to learn more about different cultures and languages",
    ),
]
```


```python
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter

transforms = [HeadlineSplitter(), NERExtractor()]
```

### Initialize test generator


```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm, embedding_model=generator_embeddings, persona_list=personas
)
```

### Load and Adapt Queries

Here we load the required query types and adapt them to the target language.


```python
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
]

for query, _ in distribution:
    prompts = await query.adapt_prompts("spanish", llm=generator_llm)
    query.set_prompts(**prompts)
```

### Generate


```python
dataset = generator.generate_with_langchain_docs(
    docs[:],
    testset_size=5,
    transforms=transforms,
    query_distribution=distribution,
)
```

    Applying HeadlineSplitter:   0%|          | 0/6 [00:00<?, ?it/s]unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    unable to apply transformation: 'headlines' property not found in this node
    Generating Scenarios: 100%|██████████| 1/1 [00:07<00:00,  7.75s/it]
    Generating Samples: 100%|██████████| 5/5 [00:03<00:00,  1.65it/s]



```python
eval_dataset = dataset.to_evaluation_dataset()
```


```python
print("Query:", eval_dataset[0].user_input)
print("Reference:", eval_dataset[0].reference)
```

    Query: Quelles sont les caractéristiques du Bronx en tant que borough de New York?
    Reference: Le Bronx est l'un des cinq arrondissements de New York, qui est la plus grande ville des États-Unis. Bien que le contexte ne fournisse pas de détails spécifiques sur le Bronx, il mentionne que New York est une ville cosmopolite avec de nombreux quartiers ethniques, ce qui pourrait inclure des caractéristiques culturelles variées présentes dans le Bronx.


That's it. You can customize the test generation process as per your requirements.


