from langchain.schema import Document
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import psycopg2
import pandas as pd
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSetup:
    """
    RAG system setup class
    Handles database connection, document extraction, and vector store initialization
    
    """

    def __init__(self):
        load_dotenv()
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.model_name = os.getenv("EMBEDDING_MODEL")


    def get_db_connection(self) -> Optional[psycopg2.extensions.connection]:
        """
        Establishes and returns a connection to the PostgreSQL database.

        This function reads the required database connection parameters (host,
        database name, user, password, and port) from environment variables.
        It's recommended to store these in a `.env` file for security and
        portability.

        Required environment variables:
        - DB_HOST: The hostname or IP address of the database server.
        - DB_NAME: The name of the database.
        - DB_USER: The username for the database connection.
        - DB_PASSWORD: The password for the database user.
        - DB_PORT: The port number (defaults to 5432 if not set).

        Returns:
            psycopg2.connection: A connection object if the connection is
                                successful, otherwise None.
        """
        load_dotenv()

        try:
            connector = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                port=os.getenv("DB_PORT", 5432)
            )
            return connector
        except psycopg2.OperationalError as e:
            logger.error(f"Could not connect to database: {e}")
            return None


    def extract_postgres_to_documents(self, connector: psycopg2.extensions.connection, table_name: str = None, query: str = None) -> List[Document]:    
        """
        Extract data from PostgreSQL and convert to Document objects
        
        Args:
            connector: Database connection object
            table_name: Name of the table to extract from
            query: Custom SQL query (optional)
        
        Returns:
            List of Document objects
        """
        if query is None and table_name:
            query = f"SELECT code, description FROM {table_name}"

        try:
            df = pd.read_sql_query(query, connector)
            documents = []
            for _, row in df.iterrows():
                product_id = str(row.get('code', '')).strip()
                description = str(row.get('description', '')).strip()
                
                # Skip empty documents
                if not product_id and not description:
                    continue

                content = product_id + ' ' + description
                metadata = {
                    "source": "postgresql",
                    "table": table_name,
                    "product_id": product_id,
                    "description": description 
                }         

                documents.append(Document(page_content=content, metadata=metadata))
            
            logger.info(f"{len(documents)} documents created from PostgreSQL table '{table_name}'")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting documents: {e}")
            return []
        
    
    def setup_complete_rag_system(self, documents: List[Document]):    
        """
    Set up a complete RAG (Retrieval-Augmented Generation) system with vector storage.
    
    This method initializes a persistent vector store, loads an embeddings model,
    and synchronizes documents between SQL database and vector store, only adding
    new documents to optimize performance.
    
    Args:
        documents (List[Document]): List of documents from SQL database to index
        
    Returns:
        tuple: (collection, model) - ChromaDB collection and SentenceTransformer model
        
    Raises:
        Exception: If vector store initialization or model loading fails
    """
        logger.info("Setting up RAG system...")
        
        # Initialize vector store
        try:
            # Create persistent client to maintain vector store between sessions
            client = chromadb.PersistentClient(path=self.vector_store_path)

            # Get existing collection or create new one if it doesn't exist
            collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Product similarity search collection"}
            )
            logger.info(f"Vector store: {collection.count()} existing documents")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
        
        # Initialize embeddings model for converting text to vectors
        try:
            # Load SentenceTransformer model for generating document embeddings
            model = SentenceTransformer(self.model_name)
            logger.info(f"Embeddings model '{self.model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            raise
        
        # Extract primary keys from document metadata to use as vector store IDs
        # Using SQL primary keys ensures consistency between SQL and vector store
        ids = [str(doc.metadata['product_id']) for doc in documents if doc.metadata.get('product_id')]
        
        # Check which documents already exist
        try:
            if collection.count() > 0:
                # Fetch existing documents by their IDs
                existing_data = collection.get(ids=ids)
                existing_ids = set(existing_data['ids'])
            else:
                existing_ids = set()
        except Exception as e:
            logger.warning(f"No existing documents found or error: {e}")
            existing_ids = set()
        
        logger.info(f"Found {len(existing_ids)} existing documents out of {len(documents)} total in SQL")
        
        # Find NEW documents that exist in SQL but not in vector store
        new_docs = []
        new_ids = []
        for doc in documents:
            doc_id = str(doc.metadata.get('product_id', ''))
            # Only add documents that have product_id and aren't already in vector store
            if doc_id and doc_id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(doc_id)
        
        # Process new documents if any are found
        if new_docs:
            logger.info(f"🆕 Adding {len(new_docs)} new documents to vector store...")
            
            # Extract text content from new documents for embedding generation
            texts = [doc.page_content for doc in new_docs]
            
            # Generate embeddings for new documents only (optimization)
            embeddings = model.encode(texts)
            
            # Extract metadata for vector store indexing
            metadatas = [doc.metadata for doc in new_docs]

            # Add only new documents to collection
            collection.add(
                ids=new_ids,                    # Unique identifiers from SQL primary keys
                embeddings=embeddings.tolist(), # Convert numpy arrays to Python lists
                metadatas=metadatas,            # Metadata for filtering and retrieval
                documents=texts                 # Original text content
            )
            logger.info(f"Added {len(new_docs)} new products to vector store")
        else:
            logger.info("Vector store is already up-to-date with SQL database")
        
        logger.info("RAG system setup completed successfully!")
        return collection, model


    def run_full_setup(self, table_name: str = "products") -> bool:
        """
    Run the complete RAG setup pipeline from database extraction to vector store initialization.
    
    This method orchestrates the entire RAG setup process:
    1. Establishes database connection
    2. Extracts documents from the specified database table
    3. Initializes the RAG system with the extracted documents
    4. Handles cleanup and error reporting
    
    Args:
        table_name (str): Name of the database table to extract data from. 
                         Defaults to "products".
    
    Returns:
        bool: True if the RAG setup completed successfully, False otherwise
        """
        try:
            # Establish connection to database
            conn = self.get_db_connection()
            if conn is None:
                # Critical failure - cannot proceed without database connection
                logger.error("Database connection failed")
                return False
            
            # Extract documents from database
            documents = self.extract_postgres_to_documents(
                connector=conn,      # Active database connection
                table_name=table_name  # Target table to extract data from
            )
            
            # Close database connection
            conn.close()
            
            # Validate that documents were successfully extracted
            if not documents:
                logger.error("No documents extracted. Setup failed.")
                return False
            
            # Initialize the complete RAG system with extracted documents
            collection, model = self.setup_complete_rag_system(documents)
            
            return True # The RAG system is now ready for similarity search operations
            
        except Exception as e:
            logger.error(f"RAG setup failed: {e}")
            return False


# Main execution - (for manual runs)
if __name__ == "__main__":
    logger.info("Starting manual RAG setup...")

    # Creates an instance of the class that manages the entire setup process
    rag_setup = RAGSetup()

    # Execute the complete RAG setup pipeline
    success = rag_setup.run_full_setup(table_name="products")
    
    if success:
        # Successful setup - system is ready for use
        logger.info("RAG setup completed successfully!")
    else:
        # Failed setup - log error and exit with non-zero status code
        logger.error("RAG setup failed!")
        exit(1) # Exit with error code to indicate failure to calling process



