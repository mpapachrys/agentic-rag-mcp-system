import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductSimilarityService:
    """
    Service for performing product similarity searches using a ChromaDB vector store
    and a SentenceTransformer model.
    
    This service provides semantic search capabilities to find products similar
    to a given query by leveraging vector embeddings and cosine similarity.
    """

    def __init__(self):
        """
        Initialize the Product Similarity Service.
        
        Loads configuration from environment variables and initializes:
        - ChromaDB vector store connection
        - SentenceTransformer embeddings model
        - Service parameters (top-k results, collection name, etc.)
        """
        load_dotenv()

        # Initialize service configuration from environment variables
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH") # Path to persistent vector store
        self.collection_name = os.getenv("COLLECTION_NAME")  # Name of ChromaDB collection
        self.model_name = os.getenv("EMBEDDING_MODEL") # SentenceTransformer model name
        self.results = int(os.getenv("SIMILARITY_SEARCH_TOP_K")) # Default number of results to return
        
        # Initialize core components
        # ChromaDB collection for vector operations
        self.collection = self._initialize_vector_store()
        # Embeddings model for encoding text
        self.model = self._initialize_embeddings_model()
        
        
        logger.info(f"Similarity service loaded with {self.collection.count()} products")

    def _initialize_vector_store(self):
        """
        Initializes and returns the ChromaDB collection.
        
        Returns:
            chromadb.Collection: Initialized ChromaDB collection
            
        Raises:
            Exception: If vector store initialization fails
        """
        try:
            # Create persistent client to maintain vector store between sessions
            client = chromadb.PersistentClient(path=self.vector_store_path)
            # Get the existing collection (*must already be created!!*)
            collection = client.get_collection(name=self.collection_name)
            return collection  
        
        except Exception as e:
            logger.error(f"Error initializing similarity service: {e}")
            raise

    def _initialize_embeddings_model(self):
        """
        Initializes and returns the SentenceTransformer model.
        
        Returns:
            SentenceTransformer: Initialized embeddings model
            
        Raises:
            Exception: If model loading fails
        """
        try:
            # Load the specified SentenceTransformer model for embedding generation
            model = SentenceTransformer(self.model_name)
            logger.info(f"Embeddings model '{self.model_name}' loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}")
            raise

    def search_and_return_similar_products(self, query: str, n_results: int = None):
        """
        Searches for similar products based on the provided query.
        
        This method:
        -Encodes the query text into a vector embedding
        -Performs similarity search in the vector store
        -Returns the most similar products
        
        Args:
            query (str): The search query text (e.g., product description, features)
            n_results (int, optional): Number of similar products to return. 
                                     Uses default from env if not specified.
        
        Returns:
            list: List of similar product documents (text content)
            
        Raises:
            Exception: If service is not properly initialized
        """

        # Validate that service components are properly initialized
        if not self.collection or not self.model:
            raise Exception("Similarity service not properly initialized")
        
        # Use default results count if not specified
        if n_results is None:
            n_results = self.results

        try:
            # Convert string query to numerical vector representation
            query_embedding = self.model.encode([query]).tolist()
            
            # Query the vector store for similar products using cosine similarity
            results = self.collection.query(
                query_embeddings=query_embedding,  # The encoded query vector
                n_results=n_results,               # Number of results to return
                include=["metadatas", "documents", "distances"]  # Data to include in response
            )

            logger.info(f"Found {len(results['documents'][0])} similar products")

            # Return the document contents of similar products
            # results['documents'][0] contains the list of matching product texts
            return results['documents'][0] 
        
        except Exception as e:
            logger.error(f"Error searching similar products: {e}")
            return []  # Return empty list on error to allow graceful degradation
            
        