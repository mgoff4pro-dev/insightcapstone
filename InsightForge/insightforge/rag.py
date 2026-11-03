"""
Steps 3-6: Custom retriever, prompt engineering, chained prompts, RAG system and memory.

Pipeline for every question (3 chained prompts):

    question + chat history
        │  (1) CONDENSE prompt  -> standalone question (uses memory to resolve "what about the West?")
        ▼
    InsightRetriever (custom)  -> exact pandas stats + FAISS semantic matches   [RAG retrieval]
        │  (2) ANALYSIS prompt   -> pulls out only the relevant numbers/facts
        ▼
        │  (3) INSIGHT prompt    -> final answer + insights + recommendations (uses memory)
        ▼
    answer  -> saved back into conversation memory
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import ConfigDict

from .knowledge_base import KBEntry, build_knowledge_base, is_research_question, keyword_search, load_pdf_chunks, retrieve

try:  # FAISS is optional — the retriever still works on routed statistics without it
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover
    FAISS = None

DEFAULT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"


def to_document(e: KBEntry) -> Document:
    return Document(page_content=f"{e.title}\n{e.text}", metadata={"topic": e.topic, "title": e.title, **e.metadata})


def get_llm(api_key: Optional[str] = None, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatOpenAI:
    kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def build_vectorstore(kb: list[KBEntry], api_key: Optional[str] = None):
    if FAISS is None or not kb:
        return None
    emb = OpenAIEmbeddings(model=EMBED_MODEL, **({"api_key": api_key} if api_key else {}))
    return FAISS.from_documents([to_document(e) for e in kb], emb)


# ---------------------------------------------------------------- custom retriever
class InsightRetriever(BaseRetriever):
    """Custom LangChain retriever that extracts relevant statistics with pandas.

    1. Entity filtering: products/regions/years/genders named in the question are
       computed live with pandas (e.g. "Widget A in the North in 2023").
    2. Topic routing: keyword routing to pre-computed summary tables.
    3. Semantic search: FAISS similarity over the same knowledge base (if available).
    4. Research documents: for strategy / "how can we…" questions, the most relevant passages
       from the PDF research papers (FAISS, or keyword ranking if no vector store).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame
    kb: list
    vectorstore: Any = None
    research_chunks: list = []
    research_vectorstore: Any = None
    k_semantic: int = 2
    k_research: int = 3
    max_docs: int = 7

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        docs = [to_document(e) for e in retrieve(self.df, self.kb, query, max_entries=self.max_docs)]
        if self.vectorstore is not None:
            titles = {d.metadata["title"] for d in docs}
            for d in self.vectorstore.similarity_search(query, k=self.k_semantic):
                if d.metadata.get("title") not in titles and len(docs) < self.max_docs + self.k_semantic:
                    docs.append(d)
        if self.research_chunks and is_research_question(query):
            if self.research_vectorstore is not None:
                docs.extend(self.research_vectorstore.similarity_search(query, k=self.k_research))
            else:
                docs.extend(to_document(e) for e in keyword_search(self.research_chunks, query, self.k_research))
        return docs


def format_docs(docs: list[Document]) -> str:
    """Data statistics first, research excerpts in a separate, clearly labelled section."""
    data = [d for d in docs if d.metadata.get("topic") != "research"]
    research = [d for d in docs if d.metadata.get("topic") == "research"]
    out = "## DATA STATISTICS\n" + "\n\n".join(f"### {d.metadata.get('title', 'Context')}\n{d.page_content}"
                                               for d in data)
    if research:
        out += "\n\n## RESEARCH EXCERPTS (from business-intelligence papers)\n" + "\n\n".join(
            f"### Source: {d.metadata.get('title')}\n{d.page_content}" for d in research)
    return out


# ---------------------------------------------------------------- prompt engineering
SYSTEM_ROLE = (
    "You are InsightForge, a senior business intelligence analyst for a small-to-mid-size company. "
    "You only use numbers that appear in the DATA STATISTICS section of the context. Never invent figures. "
    "RESEARCH EXCERPTS are general findings from published papers: use them only to support or explain "
    "recommendations, cite them by source name, and never treat their numbers as the company's data. "
    "If the context does not contain the answer, say what is missing."
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest question as a standalone question about the sales data, "
               "using the chat history to resolve references like 'it', 'that region', 'what about…'. "
               "Return only the rewritten question. If it is already standalone, return it unchanged."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_ROLE),
    ("human",
     "Data context (retrieved statistics):\n{context}\n\n"
     "Question: {question}\n\n"
     "Step 1 — extract ONLY the facts needed to answer the question. "
     "List them as short bullet points with exact numbers, and note any comparisons "
     "(highest/lowest, % differences, trends). If years are marked as partial, compare them using "
     "average daily sales, not totals. If research excerpts are relevant, add a separate "
     "'Research:' list of 1-3 findings with their source. Do not give recommendations yet."),
])

INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_ROLE +
     " Write for a busy business owner: clear, specific, and actionable."),
    MessagesPlaceholder("history"),
    ("human",
     "Question: {question}\n\nVerified facts from the data:\n{facts}\n\n"
     "Step 2 — answer the question using these facts. Structure:\n"
     "**Answer:** a direct 1-3 sentence answer with the key numbers.\n"
     "**Insights:** 2-3 bullets on what the numbers mean (patterns, trends, outliers).\n"
     "**Recommendations:** 2-3 concrete, data-backed actions (cite a research source where one supports it).\n"
     "Keep it under 250 words."),
])


# ---------------------------------------------------------------- assistant with memory
class InsightForgeAssistant:
    """RAG business-intelligence assistant with conversation memory."""

    def __init__(self, df: pd.DataFrame, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 use_vectorstore: bool = True, memory_window: int = 6, pdf_folder: Optional[str] = None):
        self.df = df
        self.kb = build_knowledge_base(df)
        self.research_chunks = load_pdf_chunks(pdf_folder) if pdf_folder else []
        self.llm = get_llm(api_key, model)
        vs = build_vectorstore(self.kb, api_key) if use_vectorstore else None
        rvs = build_vectorstore(self.research_chunks, api_key) if use_vectorstore else None
        self.retriever = InsightRetriever(df=df, kb=self.kb, vectorstore=vs,
                                          research_chunks=self.research_chunks, research_vectorstore=rvs)
        self.memory = InMemoryChatMessageHistory()   # Step 6: memory integration
        self.memory_window = memory_window            # number of past exchanges kept in the prompt

        parser = StrOutputParser()
        self.condense_chain = CONDENSE_PROMPT | self.llm | parser
        self.analysis_chain = ANALYSIS_PROMPT | self.llm | parser
        self.insight_chain = INSIGHT_PROMPT | self.llm | parser

    def _history(self):
        return self.memory.messages[-2 * self.memory_window:]

    def ask(self, question: str) -> dict:
        history = self._history()
        standalone = self.condense_chain.invoke({"history": history, "question": question}) if history else question
        docs = self.retriever.invoke(standalone)
        context = format_docs(docs)
        facts = self.analysis_chain.invoke({"context": context, "question": standalone})
        answer = self.insight_chain.invoke({"history": history, "question": question, "facts": facts})

        self.memory.add_user_message(question)
        self.memory.add_ai_message(answer)
        return {
            "question": question,
            "standalone_question": standalone,
            "answer": answer,
            "facts": facts,
            "sources": [d.metadata.get("title") for d in docs],
            "context": context,
        }

    def answer_only(self, question: str) -> str:
        """Stateless answer (no memory) — used for evaluation so tests don't leak into each other."""
        docs = self.retriever.invoke(question)
        facts = self.analysis_chain.invoke({"context": format_docs(docs), "question": question})
        return self.insight_chain.invoke({"history": [], "question": question, "facts": facts})

    def clear_memory(self):
        self.memory.clear()
