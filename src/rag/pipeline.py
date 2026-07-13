from typing import Dict
import time
import logging

from src.monitoring.observability import (
    tracer,
    rag_queries_total,
    retrieved_documents,
    total_latency,
    retrieval_accuracy,
    hallucination_rate
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

        self.initialize_bm25_from_vector_db()

    def initialize_bm25_from_vector_db(self):

        docs = self.vector_store.get_all_documents()

        if docs:
            self.retriever.create_bm25_index(docs)
        else:
            logger.warning("No documents available for BM25")

    def retrieve(
            self,
            question: str
    ):

        start = time.time()

        docs = self.retriever.retrieve(
            question,
            self.vector_store
        )

        from src.monitoring.observability import retrieval_latency

        retrieval_latency.record(
            time.time() - start
        )

        return docs

    def generate(
            self,
            question: str,
            documents
    ):

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

        start = time.time()

        response = self.llm.invoke(
            prompt
        )

        from src.monitoring.observability import llm_latency

        llm_latency.record(
            time.time() - start
        )

        return response.content

    def query(
            self,
            question: str
    ) -> Dict:

        pipeline_start = time.time()

        with tracer.start_as_current_span(
                "rag.query"
        ) as span:

            try:

                retrieved_docs = self.retrieve(
                    question
                )

                answer = self.generate(
                    question,
                    retrieved_docs
                )

                evaluation = (
                    self.evaluator.evaluate(
                        question,
                        answer,
                        retrieved_docs
                    )
                )

                retrieval_confidence = (
                    evaluation[
                        "retrieval_similarity"
                    ]
                )

                hallucination_score = (
                        1 -
                        evaluation[
                            "answer_grounding"
                        ]
                )

                retrieved_documents.add(
                    len(retrieved_docs)
                )

                retrieval_accuracy.add(
                    retrieval_confidence * 100,
                    attributes={
                        "metric": "retrieval"
                    }
                )

                hallucination_rate.add(
                    hallucination_score * 100,
                    attributes={
                        "metric": "hallucination"
                    }
                )

                rag_queries_total.add(
                    1,
                    attributes={
                        "status": "success"
                    }
                )

                total_latency.record(
                    time.time()
                    -
                    pipeline_start
                )

                span.set_attribute(
                    "retrieval_accuracy",
                    retrieval_confidence
                )

                span.set_attribute(
                    "hallucination_rate",
                    hallucination_score
                )

                return {
                    "answer": answer,
                    "documents": retrieved_docs,
                    "metrics": {
                        "retrieval_accuracy":
                            retrieval_confidence,
                        "hallucination_rate":
                            hallucination_score
                    },
                    "retrieval_confidence":
                        retrieval_confidence,
                    "hallucination_score":
                        hallucination_score
                }
            except Exception as e:
                rag_queries_total.add(
                    1,
                    attributes={
                        "status": "error"
                    }
                )

                logger.exception(
                    f"RAG pipeline failed: {e}"
                )
                raise
