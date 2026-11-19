import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

DOCUMENTS = [
    "Ragas are melodic frameworks in Indian classical music.",
    "There are many types of ragas, each with its own mood and time of day.",
    "Ragas are used to evoke specific emotions in the listener.",
    "The performance of a raga involves improvisation within a set structure.",
    "Ragas can be performed on various instruments or sung vocally.",
]


@dataclass
class TraceEvent:
    """Single event in the RAG application trace"""

    event_type: str
    component: str
    data: Dict[str, Any]


class BaseRetriever:
    """
    Base class for retrievers.
    Subclasses should implement the fit and get_top_k methods.
    """

    def __init__(self):
        self.documents = []

    def fit(self, documents: List[str]):
        """Store the documents"""
        self.documents = documents

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k most relevant documents for the query."""
        raise NotImplementedError("Subclasses should implement this method.")


class SimpleKeywordRetriever(BaseRetriever):
    """Ultra-simple keyword matching retriever"""

    def __init__(self):
        super().__init__()

    def _count_keyword_matches(self, query: str, document: str) -> int:
        """Count how many query words appear in the document"""
        query_words = query.lower().split()
        document_words = document.lower().split()
        matches = 0
        for word in query_words:
            if word in document_words:
                matches += 1
        return matches

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Get top k documents by keyword match count"""
        scores = []

        for i, doc in enumerate(self.documents):
            match_count = self._count_keyword_matches(query, doc)
            scores.append((i, match_count))

        # Sort by match count (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]


class ExampleRAG:
    """
    Simple RAG system that:
    1. accepts a llm client
    2. uses simple keyword matching to retrieve relevant documents
    3. uses the llm client to generate a response based on the retrieved documents when a query is made
    """

    def __init__(
        self,
        llm_client,
        retriever: Optional[BaseRetriever] = None,
        system_prompt: Optional[str] = None,
        logdir: str = "logs",
    ):
        """
        Initialize RAG system

        Args:
            llm_client: LLM client with a generate() method
            retriever: Document retriever (defaults to SimpleKeywordRetriever)
            system_prompt: System prompt template for generation
            logdir: Directory for trace log files
        """
        self.llm_client = llm_client
        self.retriever = retriever or SimpleKeywordRetriever()
        self.system_prompt = (
            system_prompt
            or """Answer the following question based on the provided documents:
                                Question: {query}
                                Documents:
                                {context}
                                Answer:
                            """
        )
        self.documents = []
        self.is_fitted = False
        self.traces = []
        self.logdir = logdir

        # Create log directory if it doesn't exist
        os.makedirs(self.logdir, exist_ok=True)

        # Initialize tracing
        self.traces.append(
            TraceEvent(
                event_type="init",
                component="rag_system",
                data={
                    "retriever_type": type(self.retriever).__name__,
                    "system_prompt_length": len(self.system_prompt),
                    "logdir": self.logdir,
                },
            )
        )

    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base"""
        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "add_documents",
                    "num_new_documents": len(documents),
                    "total_documents_before": len(self.documents),
                    "document_lengths": [len(doc) for doc in documents],
                },
            )
        )

        self.documents.extend(documents)
        # Refit retriever with all documents
        self.retriever.fit(self.documents)
        self.is_fitted = True

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def set_documents(self, documents: List[str]):
        """Set documents (replacing any existing ones)"""
        old_doc_count = len(self.documents)

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "set_documents",
                    "num_new_documents": len(documents),
                    "old_document_count": old_doc_count,
                    "document_lengths": [len(doc) for doc in documents],
                },
            )
        )

        self.documents = documents
        self.retriever.fit(self.documents)
        self.is_fitted = True

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for the query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries containing document info
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_start",
                    "query": query,
                    "query_length": len(query),
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        top_docs = self.retriever.get_top_k(query, k=top_k)

        retrieved_docs = []
        for idx, score in top_docs:
            if score > 0:  # Only include documents with positive similarity scores
                retrieved_docs.append(
                    {
                        "content": self.documents[idx],
                        "similarity_score": score,
                        "document_id": idx,
                    }
                )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_complete",
                    "num_retrieved": len(retrieved_docs),
                    "scores": [doc["similarity_score"] for doc in retrieved_docs],
                    "document_ids": [doc["document_id"] for doc in retrieved_docs],
                },
            )
        )

        return retrieved_docs

    def generate_response(self, query: str, top_k: int = 3) -> str:
        """
        Generate response to query using retrieved documents

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Generated response
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Generate response using LLM client
        prompt = self.system_prompt.format(query=query, context=context)

        self.traces.append(
            TraceEvent(
                event_type="llm_call",
                component="openai_api",
                data={
                    "operation": "generate_response",
                    "model": "gpt-4o",
                    "query": query,
                    "prompt_length": len(prompt),
                    "context_length": len(context),
                    "num_context_docs": len(retrieved_docs),
                },
            )
        )

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            if not response.choices:
                raise ValueError("LLM API returned no choices")
            response_text = response.choices[0].message.content.strip()

            self.traces.append(
                TraceEvent(
                    event_type="llm_response",
                    component="openai_api",
                    data={
                        "operation": "generate_response",
                        "response_length": len(response_text),
                        "usage": (
                            response.usage.model_dump() if response.usage else None
                        ),
                        "model": "gpt-4o",
                    },
                )
            )

            return response_text

        except Exception as e:
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="openai_api",
                    data={"operation": "generate_response", "error": str(e)},
                )
            )
            return f"Error generating response: {str(e)}"

    def query(
        self, question: str, top_k: int = 3, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response

        Args:
            question: User question
            top_k: Number of documents to retrieve
            run_id: Optional run ID for tracing (auto-generated if not provided)

        Returns:
            Dictionary containing response and retrieved documents
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"

        # Reset traces for this query
        self.traces = []

        self.traces.append(
            TraceEvent(
                event_type="query_start",
                component="rag_system",
                data={
                    "run_id": run_id,
                    "question": question,
                    "question_length": len(question),
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        try:
            retrieved_docs = self.retrieve_documents(question, top_k)
            response = self.generate_response(question, top_k)

            result = {"answer": response, "run_id": run_id}

            self.traces.append(
                TraceEvent(
                    event_type="query_complete",
                    component="rag_system",
                    data={
                        "run_id": run_id,
                        "success": True,
                        "response_length": len(response),
                        "num_retrieved": len(retrieved_docs),
                    },
                )
            )

            logs_path = self.export_traces_to_log(run_id, question, result)
            return {"answer": response, "run_id": run_id, "logs": logs_path}

        except Exception as e:
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="rag_system",
                    data={"run_id": run_id, "operation": "query", "error": str(e)},
                )
            )

            # Return error result
            logs_path = self.export_traces_to_log(run_id, question, None)
            return {
                "answer": f"Error processing query: {str(e)}",
                "run_id": run_id,
                "logs": logs_path,
            }

    def export_traces_to_log(
        self,
        run_id: str,
        query: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ):
        """Export traces to a log file with run_id"""
        timestamp = datetime.now().isoformat()
        log_filename = (
            f"rag_run_{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        )
        log_filepath = os.path.join(self.logdir, log_filename)

        log_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "query": query,
            "result": result,
            "num_documents": len(self.documents),
            "traces": [asdict(trace) for trace in self.traces],
        }

        with open(log_filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"RAG traces exported to: {log_filepath}")
        return log_filepath


def default_rag_client(llm_client, logdir: str = "logs") -> ExampleRAG:
    """
    Create a default RAG client with OpenAI LLM and optional retriever.

    Args:
        retriever: Optional retriever instance (defaults to SimpleKeywordRetriever)
        logdir: Directory for trace logs
    Returns:
        ExampleRAG instance
    """
    retriever = SimpleKeywordRetriever()
    client = ExampleRAG(llm_client=llm_client, retriever=retriever, logdir=logdir)
    client.add_documents(DOCUMENTS)  # Add default documents
    return client


if __name__ == "__main__":
    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_openai_api_key'")
        exit(1)

    # Initialize RAG system with tracing enabled
    llm = OpenAI(api_key=api_key)
    r = SimpleKeywordRetriever()
    rag_client = ExampleRAG(llm_client=llm, retriever=r, logdir="logs")

    # Add documents (this will be traced)
    rag_client.add_documents(DOCUMENTS)

    # Run query with tracing
    query = "What is Ragas"
    print(f"Query: {query}")
    response = rag_client.query(query, top_k=3)

    print("Response:", response["answer"])
    print(f"Run ID: {response['logs']}")
