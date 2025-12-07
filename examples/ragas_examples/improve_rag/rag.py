"""
RAG implementation supporting both naive and agentic modes.

Usage:
    retriever = BM25Retriever()                        # create retriever
    rag = RAG(llm_client, retriever)                   # naive mode (default)
    rag = RAG(llm_client, retriever, mode="agentic")   # agentic mode
    result = await rag.query("What is...?")            # returns: {answer, retrieved_documents, num_retrieved}
"""

import logging
from typing import Any, Dict, Optional

import mlflow
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever as LangchainBM25Retriever
from openai import AsyncOpenAI

import datasets

# Configure logger
logger = logging.getLogger(__name__)


class BM25Retriever:
    """Simple BM25-based retriever for document search."""
    
    def __init__(self, dataset_name="m-ric/huggingface_doc", default_k=3):
        self.default_k = default_k
        self.retriever = self._build_retriever(dataset_name)
    
    def _build_retriever(self, dataset_name: str) -> LangchainBM25Retriever:
        """Build a BM25 retriever from HuggingFace docs."""
        knowledge_base = datasets.load_dataset(dataset_name, split="train")
        
        # Create documents
        source_documents = [
            Document(
                page_content=row["text"],
                metadata={"source": row["source"].split("/")[1]},
            )
            for row in knowledge_base
        ]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        
        all_chunks = []
        for document in source_documents:
            chunks = text_splitter.split_documents([document])
            all_chunks.extend(chunks)
        
        # Simple deduplication
        unique_chunks = []
        seen_content = set()
        for chunk in all_chunks:
            if chunk.page_content not in seen_content:
                seen_content.add(chunk.page_content)
                unique_chunks.append(chunk)
        
        return LangchainBM25Retriever.from_documents(
            documents=unique_chunks,
            k=1,  # Will be overridden by retrieve method
        )
    
    def retrieve(self, query: str, top_k: int = None):
        """Retrieve documents for a given query."""
        if top_k is None:
            top_k = self.default_k
        self.retriever.k = top_k
        return self.retriever.invoke(query)


class RAG:
    """RAG system that can operate in naive or agentic mode."""

    def __init__(self, llm_client: AsyncOpenAI, retriever: BM25Retriever, mode="naive", system_prompt=None, model="gpt-5-mini", default_k=3):
        # Enable MLflow autolog for OpenAI API calls
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.openai.autolog()
        
        self.llm_client = llm_client
        self.retriever = retriever
        self.mode = mode.lower()
        self.model = model
        self.default_k = default_k
        self.system_prompt = system_prompt or "Answer only based on documents. Be concise.\n\nQuestion: {query}\nDocuments:\n{context}\nAnswer:"
        self._agent = None
        
        if self.mode == "agentic":
            self._setup_agent()

    def _setup_agent(self):
        """Setup agent for agentic mode."""
        try:
            from agents import Agent, function_tool
        except ImportError:
            raise ImportError("agents package required for agentic mode")

        @function_tool
        def retrieve(query: str) -> str:
            """Search Hugging Face docs for technical info, APIs, commands, and examples.
            Use exact terms (e.g., "from_pretrained", "ESPnet upload", "torchrun"). 
            Try 2-3 targeted searches: specific terms → tool names → alternatives."""
            docs = self.retriever.retrieve(query, self.default_k)
            if not docs:
                return f"No documents found for '{query}'. Try different search terms or break down the query into smaller parts."
            return "\n\n".join([f"Doc {i}: {doc.page_content}" for i, doc in enumerate(docs, 1)])

        self._agent = Agent(
            name="RAG Assistant",
            model=self.model,
            instructions="Search with exact terms first (commands, APIs, tool names). Try 2-3 different searches if needed. Only answer from retrieved documents. Preserve exact syntax and technical details.",
            tools=[retrieve]
        )

    async def _naive_query(self, question: str, top_k: int) -> Dict[str, Any]:
        """Handle naive mode: retrieve once, then generate."""
        # Retrieve documents
        docs = self.retriever.retrieve(question, top_k)
        
        if not docs:
            return {"answer": "No relevant documents found.", "retrieved_documents": [], "num_retrieved": 0}
        
        # Generate response
        context = "\n\n".join([f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(docs, 1)])
        prompt = self.system_prompt.format(query=question, context=context)
        
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Get the active trace ID
        trace_id = mlflow.get_last_active_trace_id()
        
        return {
            "answer": response.choices[0].message.content.strip(),
            "retrieved_documents": [{"content": doc.page_content, "metadata": doc.metadata, "document_id": i} for i, doc in enumerate(docs)],
            "num_retrieved": len(docs),
            "mlflow_trace_id": trace_id
        }

    async def _agentic_query(self, question: str, top_k: int) -> Dict[str, Any]:
        """Handle agentic mode: agent controls retrieval strategy."""
        try:
            from agents import Runner
        except ImportError:
            raise ImportError("agents package required for agentic mode")
        
        # Let agent handle the retrieval and reasoning
        result = await Runner.run(self._agent, input=question)
        
        # Get the active trace ID
        trace_id = mlflow.get_last_active_trace_id()
        
        # In agentic mode, the agent controls retrieval internally
        # so we don't return specific retrieved documents
        return {
            "answer": result.final_output,
            "retrieved_documents": [],  # Agent handles retrieval internally
            "num_retrieved": 0,  # Cannot determine exact count from agent execution
            "mlflow_trace_id": trace_id
        }

    async def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Query the RAG system."""
        if top_k is None:
            top_k = self.default_k
            
        try:
            if self.mode == "naive":
                return await self._naive_query(question, top_k)
            elif self.mode == "agentic":
                return await self._agentic_query(question, top_k)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        except Exception as e:
            # Try to get trace ID even in error cases
            trace_id = mlflow.get_last_active_trace_id()
            return {
                "answer": f"Error: {str(e)}", 
                "retrieved_documents": [], 
                "num_retrieved": 0,
                "mlflow_trace_id": trace_id
            }


# Demo
async def main():
    import os
    import pathlib

    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    
    # Load .env from root
    root_dir = pathlib.Path(__file__).parent.parent.parent.parent
    load_dotenv(root_dir / ".env")
    
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Suppress HTTP request logs from OpenAI/httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    
    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # Test with a question that failed in previous evaluation
    query = "What command is used to upload an ESPnet model to a Hugging Face repository?"
    
    logger.info("RAG DEMO")
    logger.info("=" * 40)
    
    # Create retriever (shared by both modes)
    logger.info("Creating BM25 retriever...")
    retriever = BM25Retriever()
    
    # Test naive mode
    logger.info("NAIVE MODE:")
    rag = RAG(openai_client, retriever)
    result = await rag.query(query)
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"MLflow Trace ID: {result.get('mlflow_trace_id', 'N/A')}")
    
    
    # Test agentic mode
    logger.info("AGENTIC MODE:")
    try:
        rag = RAG(openai_client, retriever, mode="agentic")
        result = await rag.query(query)
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"MLflow Trace ID: {result.get('mlflow_trace_id', 'N/A')}")
    except ImportError:
        logger.warning("Agentic mode unavailable (agents package missing)")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
