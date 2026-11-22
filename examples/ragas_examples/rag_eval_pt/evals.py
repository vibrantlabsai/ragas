"""Exemplo de avaliação de sistema RAG em português usando Ragas.

Este exemplo demonstra como avaliar um sistema RAG que opera com documentos em português,
utilizando o dataset e métricas do Ragas com conteúdo PT-BR.

Execução:
    export OPENAI_API_KEY='sua_chave'
    python -m ragas_examples.rag_eval_pt.evals
"""

import asyncio
import os
import sys
from pathlib import Path

from openai import OpenAI

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

sys.path.insert(0, str(Path(__file__).parent))
from rag import criar_cliente_rag

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
rag_client = criar_cliente_rag(cliente_llm=openai_client, diretorio_log="evals/logs")
llm = llm_factory("gpt-4o", client=openai_client)


def carregar_dataset():
    dataset = Dataset(
        name="dataset_rag_ptbr",
        backend="local/csv",
        root_dir="evals",
    )

    amostras = [
        {
            "pergunta": "O que é RAG e quais são suas etapas?",
            "criterios_avaliacao": (
                "- RAG significa Retrieval-Augmented Generation\n"
                "- Combina busca de documentos com geração de texto\n"
                "- Possui três etapas: indexação, recuperação e geração"
            ),
        },
        {
            "pergunta": "Como o RAG reduz alucinações em LLMs?",
            "criterios_avaliacao": (
                "- Fundamenta respostas em documentos reais\n"
                "- Não depende apenas do conhecimento interno do modelo\n"
                "- Usa busca semântica para encontrar contexto relevante"
            ),
        },
        {
            "pergunta": "Quais tecnologias são usadas em sistemas RAG avançados?",
            "criterios_avaliacao": (
                "- Re-ranking para melhorar ordenação dos documentos\n"
                "- Busca híbrida combinando keyword e vetorial\n"
                "- Frameworks como LangChain e LangGraph\n"
                "- Bancos vetoriais como ChromaDB e Pinecone"
            ),
        },
        {
            "pergunta": "Quais são os desafios de implementar RAG no Brasil?",
            "criterios_avaliacao": (
                "- Documentos em português com poucos recursos de NLP\n"
                "- Conformidade com LGPD para dados sensíveis\n"
                "- Domínios específicos como saúde e direito\n"
                "- Necessidade de avaliação com métricas de fidelidade"
            ),
        },
    ]

    for amostra in amostras:
        linha = {
            "pergunta": amostra["pergunta"],
            "criterios_avaliacao": amostra["criterios_avaliacao"],
        }
        dataset.append(linha)

    dataset.save()
    return dataset


metrica_correcao = DiscreteMetric(
    name="correcao_ptbr",
    prompt=(
        "Verifique se a resposta contém os pontos mencionados nos critérios de avaliação "
        "e retorne 'pass' ou 'fail'.\n"
        "Resposta: {response}\n"
        "Critérios de Avaliação: {criterios_avaliacao}"
    ),
    allowed_values=["pass", "fail"],
)


@experiment()
async def executar_experimento(linha):
    resposta = rag_client.consultar(linha["pergunta"])

    pontuacao = metrica_correcao.score(
        llm=llm,
        response=resposta.get("resposta", " "),
        criterios_avaliacao=linha["criterios_avaliacao"],
    )

    visao_experimento = {
        **linha,
        "resposta": resposta.get("resposta", ""),
        "pontuacao": pontuacao.value,
        "arquivo_log": resposta.get("logs", " "),
    }
    return visao_experimento


async def main():
    dataset = carregar_dataset()
    print("Dataset PT-BR carregado:", dataset)

    resultados = await executar_experimento.arun(dataset)
    print("Experimento concluído!")
    print("Resultados:", resultados)

    resultados.save()
    csv_path = Path(".") / "experiments" / f"{resultados.name}.csv"
    print(f"\nResultados salvos em: {csv_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
