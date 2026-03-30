# 💡 Agentic RAG System with Model Context Protocol (MCP)

This project implements an intelligent agent system designed to automate the processing of incoming emails, specifically focusing on **customer orders**. It combines a Large Language Model (LLM) with a **Retrieval-Augmented Generation (RAG)** pipeline and the **Model Context Protocol (MCP)** to provide a powerful, context-aware, and action-oriented AI workflow.

The architecture is built to classify emails, retrieve relevant product context, and then use the LLM to process the order by calling specific external tools (MCP servers) like a PostgreSQL database or a PDF exporter.

## 📁 Project Structure

```
.
├── folder/
│   ├── config/
│   │   └── mcp_servers.json   # Configuration for all MCP tool servers (PostgreSQL, PDF exporter)
│   ├── reports/
│   │   └── (empty)            # Output directory for exported reports (markdown2pdf MCP tool)
│   └── src/
│       ├── main.py            # Main application file. Orchestrates email classification and agent execution.
│       ├── rag_setup.py       # Initializes the RAG vector store by extracting data from PostgreSQL.
│       └── similarity_service.py # Service for performing semantic search against the vector store.
├── .env                       # Environment variables (API keys, database credentials, RAG paths).
└── requirements.txt           # Python dependency list.
```

## 🚀 Key Technologies

| Technology | Purpose |
| :--- | :--- |
| **Model Context Protocol (MCP)** | Standardized protocol (`mcp-use` library) for the LLM agent to access external data and tools (`postgres`, `markdown2pdf`). |
| **Retrieval-Augmented Generation (RAG)** | Used to provide the LLM with up-to-date, relevant context (product data) by searching a vector store. |
| **ChromaDB** | The vector store used to index product information. |
| **Sentence Transformers** | Used to generate embeddings (vectors) for text, enabling semantic search. |
| **OpenRouter / Gemini** | The core LLM provider (`google/gemini-2.5-flash`) used for email classification and agentic decision-making/tool-use. |
| **PostgreSQL** | The source database for product data, accessed by both the RAG system and the MCP agent. |
| **Node.js (`npx`)** | Required to run the external MCP Server tools (PostgreSQL, PDF) as separate processes. |

## ⚙️ Prerequisites

Before starting, ensure you have the following installed and configured:

1.  **Python 3.8+**
2.  **Node.js and npm/npx:** Required to run the external MCP server processes.
3.  **PostgreSQL Database:** A running PostgreSQL instance with the product data available.
4.  **OpenRouter API Key:** A valid key for the LLM integration.

## 🛠️ Setup Instructions

### 1\. Configure Environment Variables

Create and populate the **`.env`** file at the root of the project with your specific credentials and paths.

| Variable | Description |
| :--- | :--- |
| `OPENROUTER_API_KEY` | Your API key for OpenRouter. |
| `OPEN_ROUTER_MODEL_NAME` | The model name (e.g., `google/gemini-2.5-flash`). |
| `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT` | Credentials for your PostgreSQL database. |
| `VECTOR_STORE_PATH` | The local path where ChromaDB will store RAG data (e.g., `./vector_store`). **Must be a directory.** |
| `COLLECTION_NAME` | The name of the vector store collection (e.g., `products`). |
| `EMBEDDING_MODEL` | The model name for text embeddings (e.g., `paraphrase-multilingual-mpnet-base-v2`). |

### 2\. Install Python Dependencies

Install all necessary Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3\. Initialize the RAG Vector Store (Crucial Step\!)

The RAG system must be populated with data before the main agent can use it. This script connects to PostgreSQL, extracts your product data, and embeds it into the ChromaDB vector store.

**You must run this command once to set up the RAG index:**

```bash
python rag_setup.py
```

*(If successful, you should see a new directory created at the path defined by `VECTOR_STORE_PATH` in your `.env` file.)*

## ▶️ Running the Application

After completing the setup and initialization, you can run the main agent script.

```bash
python main.py
```

### Execution Flow in `main.py`:

1.  Loads environment variables and initializes the LLM (`google/gemini-2.5-flash`).
2.  Initializes the `MCPAgent` and connects it to the MCP servers defined in `mcp_servers.json` (PostgreSQL and markdown2pdf).
3.  **Email Classification:** The LLM classifies a placeholder email as either `ORDER` or `OTHER`.
4.  **If `ORDER`:**
      * The `ProductSimilarityService` (RAG system) searches the ChromaDB vector store to find the most similar products mentioned in the email.
      * The LLM Agent is activated with the **original email content** and the **similar products context** (from RAG).
      * The agent uses its tools (e.g., the `postgres` MCP server to check inventory or the `markdown2pdf` MCP server to generate a report) to complete the order processing logic.
5.  **If `OTHER`:** A simple log message is output, and processing stops.