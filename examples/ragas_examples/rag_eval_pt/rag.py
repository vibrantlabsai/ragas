import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

DOCUMENTOS = [
    "RAG (Retrieval-Augmented Generation) é uma técnica que combina busca de documentos com geração de texto por LLMs.",
    "Um sistema RAG típico possui três etapas: indexação, recuperação e geração.",
    "A indexação transforma documentos em embeddings vetoriais armazenados em um banco de dados vetorial como ChromaDB ou Pinecone.",
    "A recuperação busca os documentos mais relevantes para uma consulta usando similaridade semântica.",
    "A geração utiliza um LLM como GPT-4 para produzir uma resposta baseada nos documentos recuperados.",
    "RAG reduz alucinações porque fundamenta as respostas em documentos reais, não apenas no conhecimento interno do modelo.",
    "Sistemas RAG avançados usam re-ranking, busca híbrida (keyword + vetorial) e agentes para melhorar a qualidade.",
    "A avaliação de sistemas RAG é feita com métricas como fidelidade (faithfulness), relevância da resposta e precisão do contexto.",
    "O LangChain e o LangGraph são frameworks populares para construir pipelines de RAG em Python.",
    "O Brasil possui desafios específicos para RAG: documentos em português, legislação (LGPD), e domínios como saúde e direito.",
]


@dataclass
class EventoRastreio:
    """Evento único no rastreio da aplicação RAG"""

    tipo_evento: str
    componente: str
    dados: Dict[str, Any]


class BuscadorPalavrasChave:
    """Buscador simples por correspondência de palavras-chave"""

    def __init__(self):
        self.documentos = []

    def ajustar(self, documentos: List[str]):
        self.documentos = documentos

    def _contar_correspondencias(self, consulta: str, documento: str) -> int:
        palavras_consulta = consulta.lower().split()
        palavras_documento = documento.lower().split()
        return sum(1 for p in palavras_consulta if p in palavras_documento)

    def buscar_top_k(self, consulta: str, k: int = 3) -> List[tuple]:
        pontuacoes = []
        for i, doc in enumerate(self.documentos):
            contagem = self._contar_correspondencias(consulta, doc)
            pontuacoes.append((i, contagem))
        pontuacoes.sort(key=lambda x: x[1], reverse=True)
        return pontuacoes[:k]


class RAGExemplo:
    """
    Sistema RAG simples com rastreamento de eventos.

    1. Aceita um cliente LLM
    2. Usa busca por palavras-chave para recuperar documentos relevantes
    3. Gera resposta baseada nos documentos recuperados
    """

    def __init__(
        self,
        cliente_llm,
        buscador: Optional[BuscadorPalavrasChave] = None,
        prompt_sistema: Optional[str] = None,
        diretorio_log: str = "logs",
    ):
        self.cliente_llm = cliente_llm
        self.buscador = buscador or BuscadorPalavrasChave()
        self.prompt_sistema = prompt_sistema or (
            "Responda a seguinte pergunta com base nos documentos fornecidos:\n"
            "Pergunta: {consulta}\n"
            "Documentos:\n"
            "{contexto}\n"
            "Resposta:"
        )
        self.documentos = []
        self._ajustado = False
        self.rastros = []
        self.diretorio_log = diretorio_log
        os.makedirs(self.diretorio_log, exist_ok=True)

        self.rastros.append(
            EventoRastreio(
                tipo_evento="inicializacao",
                componente="sistema_rag",
                dados={
                    "tipo_buscador": type(self.buscador).__name__,
                    "tamanho_prompt": len(self.prompt_sistema),
                },
            )
        )

    def adicionar_documentos(self, documentos: List[str]):
        self.rastros.append(
            EventoRastreio(
                tipo_evento="operacao_documento",
                componente="sistema_rag",
                dados={
                    "operacao": "adicionar",
                    "novos_documentos": len(documentos),
                    "total_anterior": len(self.documentos),
                },
            )
        )
        self.documentos.extend(documentos)
        self.buscador.ajustar(self.documentos)
        self._ajustado = True

    def recuperar_documentos(self, consulta: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._ajustado:
            raise ValueError("Nenhum documento foi adicionado.")

        top_docs = self.buscador.buscar_top_k(consulta, k=top_k)
        recuperados = []
        for idx, pontuacao in top_docs:
            if pontuacao > 0:
                recuperados.append({
                    "conteudo": self.documentos[idx],
                    "pontuacao": pontuacao,
                    "id_documento": idx,
                })
        return recuperados

    def gerar_resposta(self, consulta: str, top_k: int = 3) -> str:
        if not self._ajustado:
            raise ValueError("Nenhum documento foi adicionado.")

        docs_recuperados = self.recuperar_documentos(consulta, top_k)
        if not docs_recuperados:
            return "Não encontrei documentos relevantes para responder sua pergunta."

        partes_contexto = []
        for i, doc in enumerate(docs_recuperados, 1):
            partes_contexto.append(f"Documento {i}:\n{doc['conteudo']}")
        contexto = "\n\n".join(partes_contexto)

        prompt = self.prompt_sistema.format(consulta=consulta, contexto=contexto)

        try:
            resposta = self.cliente_llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.prompt_sistema},
                    {"role": "user", "content": prompt},
                ],
            )
            texto_resposta = resposta.choices[0].message.content.strip()
            return texto_resposta
        except Exception as e:
            return f"Erro ao gerar resposta: {str(e)}"

    def consultar(self, pergunta: str, top_k: int = 3, id_execucao: Optional[str] = None) -> Dict[str, Any]:
        if id_execucao is None:
            id_execucao = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(pergunta) % 10000:04d}"

        self.rastros = []
        self.rastros.append(
            EventoRastreio(
                tipo_evento="consulta_inicio",
                componente="sistema_rag",
                dados={"id_execucao": id_execucao, "pergunta": pergunta, "top_k": top_k},
            )
        )

        try:
            docs_recuperados = self.recuperar_documentos(pergunta, top_k)
            resposta = self.gerar_resposta(pergunta, top_k)

            resultado = {"resposta": resposta, "id_execucao": id_execucao}
            caminho_logs = self._exportar_rastros(id_execucao, pergunta, resultado)
            return {"resposta": resposta, "id_execucao": id_execucao, "logs": caminho_logs}
        except Exception as e:
            return {"resposta": f"Erro: {str(e)}", "id_execucao": id_execucao, "logs": ""}

    def _exportar_rastros(self, id_execucao: str, consulta: str, resultado: Dict) -> str:
        timestamp = datetime.now().isoformat()
        nome_arquivo = f"rag_exec_{id_execucao}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        caminho = os.path.join(self.diretorio_log, nome_arquivo)

        dados_log = {
            "id_execucao": id_execucao,
            "timestamp": timestamp,
            "consulta": consulta,
            "resultado": resultado,
            "total_documentos": len(self.documentos),
            "rastros": [asdict(r) for r in self.rastros],
        }
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados_log, f, indent=2, ensure_ascii=False)

        print(f"Rastros exportados para: {caminho}")
        return caminho


def criar_cliente_rag(cliente_llm, diretorio_log: str = "logs") -> RAGExemplo:
    buscador = BuscadorPalavrasChave()
    cliente = RAGExemplo(cliente_llm=cliente_llm, buscador=buscador, diretorio_log=diretorio_log)
    cliente.adicionar_documentos(DOCUMENTOS)
    return cliente


if __name__ == "__main__":
    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        print("Erro: variável de ambiente OPENAI_API_KEY não definida.")
        print("Defina sua chave da OpenAI: export OPENAI_API_KEY='sua_chave'")
        exit(1)

    llm = OpenAI(api_key=api_key)
    rag_client = criar_cliente_rag(llm)

    pergunta = "O que é RAG e como ele reduz alucinações?"
    print(f"Pergunta: {pergunta}")
    resposta = rag_client.consultar(pergunta, top_k=3)

    print("Resposta:", resposta["resposta"])
    print(f"ID Execução: {resposta['logs']}")
