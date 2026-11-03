# InsightForge — AI-Powered Business Intelligence Assistant

Advanced Generative AI capstone. InsightForge analyzes business sales data, answers questions in plain English with insights and recommendations, and visualizes the results — built with **pandas, LangChain, RAG (FAISS) and OpenAI LLMs**, with a **Streamlit** UI and **QAEvalChain** evaluation.

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."          # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."

jupyter notebook InsightForge.ipynb      # step-by-step walkthrough (Steps 1–7)
streamlit run app.py                     # the interactive app
```

**Data:** `data/sales_data.csv` is the course dataset (2,500 daily records, Jan 2022 – Nov 2028). `data/pdfs/` holds the four research papers from the course's PDF folder, which the assistant uses for strategy questions. You can also upload a different CSV with the same columns in the app's sidebar.

**Course helper guide:** the guide suggests exporting the notebook to a `.py` file and stripping print statements before running Streamlit. That step isn't needed here, because `app.py` is already a clean Streamlit script. Just run `streamlit run app.py` and open the Streamlit port in the lab.

## Project structure

```
InsightForge/
├── InsightForge.ipynb          # walkthrough notebook — run this for your submission
├── app.py                      # Streamlit UI (chat, dashboard, data & stats, evaluation)
├── insightforge/
│   ├── data_analysis.py        # Steps 1–3: load/explore data, advanced summaries & statistics
│   ├── knowledge_base.py       # Step 2: knowledge base + routing/entity logic for the custom retriever
│   ├── rag.py                  # Steps 3–6: custom retriever, prompts, prompt chain, RAG, memory
│   ├── evaluation.py           # Step 7: QAEvalChain evaluation against pandas ground truth
│   └── visualizations.py       # Step 7: the four required chart groups (+ satisfaction)
├── data/sales_data.csv          # course dataset
├── data/pdfs/                   # course research PDFs (knowledge base for strategy questions)
├── outputs/charts/             # chart PNGs for your report
└── requirements.txt
```

## How each requirement is met

| Requirement | Implementation |
|---|---|
| 1. Data preparation | `load_data()` validates columns, adds Year / Quarter / Month / Age_Group (no cleaning, per the brief) |
| 2. Knowledge base | `build_knowledge_base()` turns 15 computed summaries into topic-tagged documents; `load_pdf_chunks()` splits the 4 research PDFs into 238 chunks with source/page metadata; both are embedded in FAISS |
| 3. Advanced data summary | Sales by year/quarter/month with growth % (partial years compared on average daily sales), product & regional analysis, product×region matrix, segmentation by gender/age, median, std dev, variance, quartiles, skew, correlations |
| 3. RAG integration | pandas processing · **custom `InsightRetriever(BaseRetriever)`** (entity filtering + topic routing + FAISS semantic search + research-PDF passages) · grounding prompts |
| 4. Chain prompts | Condense → Analysis (fact extraction) → Insight (answer, insights, recommendations), built with LCEL |
| 5. RAG system | `InsightForgeAssistant.ask()` retrieves stats, then runs the chain over them |
| 6. Memory | `InMemoryChatMessageHistory` with a 6-exchange window; follow-ups are rewritten into standalone questions |
| 7. Model evaluation | `QAEvalChain` grades 8 test questions whose answers are computed with pandas; reports accuracy and latency |
| 7. Visualization | Sales trends, product comparison, regional analysis, customer demographics |
| 7. Streamlit UI | Chat with sources + "how this answer was built" panel, filterable dashboard, stats tables, evaluation runner |

## Architecture

```
question ─► (1) Condense prompt + memory ─► standalone question
                                               │
                         InsightRetriever ◄────┘
           ├─ entity filter (pandas, live)  e.g. "Widget A, North, 2023"
           ├─ topic routing → summary tables
           ├─ FAISS semantic search
           └─ research PDF passages (strategy questions only)
                                               │ retrieved statistics
                    (2) Analysis prompt ───────┘ → verified facts
                    (3) Insight prompt + memory → Answer / Insights / Recommendations
```

**Design choices:**
- The LLM never does arithmetic over raw rows. All numbers are computed with pandas and handed to it as context, and the prompts forbid inventing figures.
- Data statistics and research excerpts go into separate, labelled sections of the prompt. The model may cite research to support recommendations but may not treat a paper's numbers as the company's data.
- 2028 is a partial year (Jan–Nov 4). Growth vs a partial year is reported on average daily sales, so it isn't mistaken for a decline.

## Notes
- Default model is `gpt-4o-mini` (switchable in the app sidebar or via `InsightForgeAssistant(model=...)`).
- Works with LangChain 0.3 and 1.x (QAEvalChain import falls back automatically).
- An evaluation run makes about 25 LLM calls (8 questions × 2 chain steps + grading), so it costs a few cents at most with `gpt-4o-mini`.
