import re
import time
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.config import RERANK_TOP_N
from src.middleware import current_request_id

log = logging.getLogger("rag")


# ── Greeting / small-talk patterns ────────────────────────────────────────────
_GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"hi+|hello+|hey+|howdy|greetings|good\s*(morning|afternoon|evening|day)|"
    r"how are you|how('?re| are) you doing|what'?s up|sup|yo|"
    r"thanks?|thank you|bye|goodbye|see you|take care|"
    r"who are you|what (can|do) you do|help me"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)

_GREETING_RESPONSE = (
    "Hello! 👋 I'm your Oracle CPQ assistant — here to help you understand "
    "Configure, Price, Quote concepts, workflows, BML formulas, pricing rules, "
    "and everything in the CPQ documentation.\n\n"
    "Feel free to ask me anything about Oracle CPQ!"
)


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_PATTERNS.match(text.strip()))


def format_docs(docs):
    """
    Format retrieved docs into structured context blocks.
    Each block is prefixed with its source document and section so the LLM
    can cite them accurately in the answer.

    Note: PDF source filtering is handled natively by Qdrant during retrieval,
    so all docs arriving here are already from CPQ PDFs.
    """
    parts = []
    for doc in docs:
        source_doc = (
            doc.metadata.get("source_doc")
            or doc.metadata.get("file_name")
            or doc.metadata.get("source")
            or "Unknown Document"
        )
        section = doc.metadata.get("section", "General")
        header  = f"[Source: {source_doc} | Section: {section}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class ConversationalRAG:

    def __init__(self, retriever, reranker, llm):

        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert Oracle CPQ documentation assistant. "
                "Your job is to understand the user's question in context and provide a clear, "
                "accurate, and helpful answer based **exclusively** on the provided context passages "
                "from the CPQ documentation.\n\n"

                "## How to answer\n"
                "1. **Understand the intent** — Interpret the user's question contextually. "
                "They may use different terminology, ask high-level conceptual questions, or "
                "refer to CPQ features indirectly. Map their question to the relevant information "
                "in the context, even if the exact keywords don't appear.\n\n"
                "2. **Synthesize across documents** — The context may contain passages from multiple "
                "CPQ documents and sections. Combine and connect related information to give a "
                "comprehensive answer. Don't treat each passage in isolation.\n\n"
                "3. **Cite your sources** — After your answer, include a 'Sources' section listing "
                "the documents and sections you drew from, in this format:\n"
                "   📄 **Sources:**\n"
                "   - *<document_name>* — <section_name>\n"
                "   List each unique source you referenced.\n\n"
                "4. **Stay grounded** — Every claim in your answer must be supported by the context. "
                "You may rephrase and explain in your own words for clarity, but do NOT add information "
                "that is not present or implied by the context passages.\n\n"

                "## Hard rules\n"
                "  - ONLY use information from the provided context passages (sourced from CPQ PDFs).\n"
                "  - NEVER fabricate, guess, or use external knowledge not present in the context.\n"
                "  - If the context does not contain enough information to answer, say: "
                "'I don't have enough information on that in the CPQ documents. "
                "Could you rephrase or ask about a related CPQ topic?'\n"
                "  - If the question is unrelated to CPQ, politely redirect: "
                "'I'm specialized in Oracle CPQ topics. Could you ask something about CPQ?'\n"
                "  - For greetings or small-talk, skip citations and respond warmly but briefly."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "Context:\n{context}\n\nQuestion: {input}")
        ])

        self.greeting_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a friendly Oracle CPQ assistant. The user has sent a greeting or conversational "
                "message. Respond warmly and naturally. Briefly mention that you are an Oracle CPQ expert "
                "and invite them to ask their CPQ question. Keep it short and welcoming."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])

        self.parser = StrOutputParser()

    def build_chain(self):

        # Prompt to reformulate a follow-up question into a standalone question
        contextualize_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Given the conversation history and the latest user question, "
                "reformulate the question into a standalone question that can be understood "
                "without the conversation history. "
                "Expand abbreviations and implicit references using the conversation context. "
                "Include relevant CPQ terminology that might help retrieve better results. "
                "Do NOT answer the question — only rewrite it if needed, otherwise return it as-is."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        contextualize_chain = contextualize_prompt | self.llm | self.parser

        def retrieve_and_rerank(x):
            req_id = current_request_id.get("")
            query = x["input"]
            chat_history = x.get("chat_history", [])

            # Reformulate the query using history so retrieval gets the right context
            if chat_history:
                t0 = time.perf_counter()
                query = contextualize_chain.invoke({"input": query, "chat_history": chat_history})
                log.info(
                    "Contextualized query in %.0fms",
                    (time.perf_counter() - t0) * 1000,
                    extra={"request_id": req_id, "step": "contextualize"},
                )

            # Retrieval
            t0 = time.perf_counter()
            docs = self.retriever.invoke(query)
            retrieval_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "Retrieved %d docs in %.0fms",
                len(docs), retrieval_ms,
                extra={"request_id": req_id, "step": "retrieval", "duration_ms": round(retrieval_ms)},
            )

            # Reranking
            t0 = time.perf_counter()
            reranked_docs = self.reranker.rerank(query, docs)
            rerank_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "Reranked to top %d in %.0fms",
                min(len(reranked_docs), RERANK_TOP_N), rerank_ms,
                extra={"request_id": req_id, "step": "rerank", "duration_ms": round(rerank_ms)},
            )

            if not reranked_docs:
                return "[No relevant documents found in the CPQ knowledge base.]"

            return format_docs(reranked_docs[:RERANK_TOP_N])

        # Full RAG chain (for CPQ questions)
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: retrieve_and_rerank(x)
            )
            | self.prompt
            | self.llm
            | self.parser
        )

        # Greeting chain (skips retrieval entirely)
        greeting_chain = (
            RunnablePassthrough.assign(context=lambda x: "")
            | self.greeting_prompt
            | self.llm
            | self.parser
        )

        # Router: pick the right chain based on input
        def route_and_invoke(x):
            req_id = current_request_id.get("")
            is_greet = _is_greeting(x["input"])
            chain = greeting_chain if is_greet else rag_chain

            t0 = time.perf_counter()
            result = chain.invoke(x)
            total_ms = (time.perf_counter() - t0) * 1000

            log.info(
                "RAG chain completed in %.0fms (greeting=%s)",
                total_ms, is_greet,
                extra={"request_id": req_id, "step": "rag_total", "duration_ms": round(total_ms)},
            )
            return result

        return RunnableLambda(route_and_invoke)