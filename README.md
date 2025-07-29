# LLM, Multimodal RAG, and AI Agent

The application provides a chatbot with conversation memories.
In the backend, it has an AI Agent to call tools/functions.

[Work in progress]


Multi-agent
- Salesperson Agent
    - system: promotes loan products
    - tool: RAG tool (retrieves product terms from docs)
    - tool: web search for economic growth


Credit Analyst Agent
    - warns of risk
    - loan default ML model
    - multimodal RAG with image, from model notebook

 
The Data Analyst Agent
    - fact-checks
    - tabular RAG (SQL) over historical loan data
    - uses a scraper to fetch real-time economic indicators (e.g., interest rates).

Manager Agent
    - concludes with a report balancing opportunity and risk.
    