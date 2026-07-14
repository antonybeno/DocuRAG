from typing import List
import time
import logging

from document import Document

from src.app import QueryResponse
from src.monitoring.observability import (
    tracer,
    retrieval_latency,
    llm_latency,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
            self,
            vector_store,
            retriever,
            evaluator,
            llm
    ):
        self.vector_store = vector_store
        self.retriever = retriever
        self.evaluator = evaluator
        self.llm = llm

    def initialize(self):
        self.initialize_bm25_from_vector_db()

    def initialize_bm25_from_vector_db(self):
        docs = self.vector_store.get_all_documents()
        if docs:
            self.retriever.create_bm25_index(docs)
        else:
            logger.warning("No documents available for BM25")

    def generate(self, question: str, documents: List[Document]):
        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )
        prompt = f"""
                    You are a technical assistant.
                    
                    Rules:
                    - Answer only from context.
                    - If missing say "Not found in documents".
                    - Do not hallucinate.
                    
                    Context:
                    
                    {context}
                    
                    Question:
                    
                    {question}
                    """
        response = self.llm.invoke(prompt, timeout=30)
        return response.content

    def query(self, question: str) -> QueryResponse:
        try:
            with tracer.start_as_current_span("retrieve_documents") as retrieval_span:
                retrieval_start = time.perf_counter()

                retrieved_docs = self.retriever.retrieve(question, self.vector_store)

                retrieval_duration = time.perf_counter() - retrieval_start
                retrieval_latency.record(retrieval_duration)
                retrieval_span.set_attribute("query.retrieved_documents_count", len(retrieved_docs))
                retrieval_span.set_attribute("query.retrieved_documents_duration", retrieval_duration)

            with tracer.start_as_current_span("llm_generate") as llm_spam:
                llm_start = time.perf_counter()

                answer = self.generate(question, retrieved_docs)

                llm_duration = time.perf_counter() - llm_start
                llm_latency.record(llm_duration)
                llm_spam.set_attribute("query.model", "llama3")
                llm_spam.set_attribute("query.llm_duration", llm_duration)

            with tracer.start_as_current_span("evaluate_answer") as eval_span:
                eval_start = time.perf_counter()

                evaluation = self.evaluator.evaluate(question, answer, retrieved_docs)

                eval_duration = time.perf_counter() - eval_start
                eval_span.set_attribute("query.evaluation_duration", eval_duration)

            return QueryResponse(
                answer=answer,
                sources=retrieved_docs,
                retrieval_confidence=evaluation["retrieval_similarity"],
                hallucination_score=1 - evaluation["answer_grounding"]
            )
        except Exception as e:
            logger.exception(f"RAG pipeline failed: {e}")
            raise
