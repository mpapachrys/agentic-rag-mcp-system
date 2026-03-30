#Postgres mcp -> https://github.com/modelcontextprotocol/servers-archived/tree/main/src/postgres
#PDF mcp -> https://github.com/2b3pro/markdown2pdf-mcp/blob/main/README.md

import asyncio
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
from similarity_service import ProductSimilarityService


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Classification Tags
CLASS_ORDER = "ORDER"
CLASS_OTHER = "OTHER"

# LLM/Agent Prompts defining the roles and expected outputs for different tasks.
CLASSIFICATION_PROMPT = """
You are an expert Email Classifier Assistant. Your whose sole task is to analyze the provided email content and output a SINGLE, precise classification tag.
Your output must adhere to the following rules and definitions without exception:
RULES FOR OUTPUT:

STRICTLY output only the corresponding classification tag.
DO NOT output any other text, reasoning, explanations, filler words, markdown, or punctuation.
Your output must be one of the following two tags: '{order}' or '{other}'.

CLASSIFICATION DEFINITIONS:
**{order}**: A definite commitment to purchase goods or services. Crucially, this classification requires the body of the email to explicitly list one or more specific products, product IDs, or clear, recognizable product descriptions in free text (e.g., specific product names like "Ταψί Διάτρητο", IDs like "01.10741", or descriptive phrases like "καρέκλα για γραφείο" or "λευκός δερμάτινος καναπές τριθέσιος"). The description, whether a formal ID or free text, must clearly represent an item for immediate purchase/invoicing.
**{other}**: Any general inquiry, complaint, technical issue,request,suggestion or communication that does not fit the order category.
"""

EXTRACTION_AVAILABILITY_ORDER_PROMPT = """
You are an expert order processing assistant. Follow these steps EXACTLY in order:

STEP 1: EXTRACT PRODUCTS FROM EMAIL
From the email, extract all products with their quantities (using RAG similarity results as context):
EMAIL: {email_body}
RAG SIMILARITY RESULTS (for context): {context}

Products to extract:
- 30.40113: 20 τεμ.
- 96.10033: 15 τεμ.  
- 30.60012: 12 τεμ.

STEP 2: CHECK AVAILABILITY FOR EACH PRODUCT
table: products
columns: code, description, quantity
Use the 'query' tool to check availability for each product code.
If available quantity meets requested quantity then status "Διαθέσιμο" else "Μη διαθέσιμο"

STEP 3: GENERATE FINAL REPORT
After checking ALL products, create a markdown report using this template:

# **Customer Order Fulfillment Report**
## **Customer Details**
(Notes for the customer, if any info dont exist, dont include them in the report.)
* **Name:** 
* **Email:** 
* **Phone:** 
* **Address:** 
* **Website:** 

## **Order Details** 
(Notes: Create a table for the order details with the following columns and the following html for the pdf export:
<table border="1" style="border-collapse: collapse; width: 100%;">
<tr>
<th style="padding: 8px; text-align: left;">Product ID</th>
<th style="padding: 8px; text-align: left;">Product Name</th>
<th style="padding: 8px; text-align: center;">Requested Quantity</th>
<th style="padding: 8px; text-align: center;">Available Quantity</th>
<th style="padding: 8px; text-align: right;">Status</th>
</tr>
<tr>
<td style="padding: 8px;">A-101</td>
<td style="padding: 8px;">Wireless Mouse</td>
<td style="padding: 8px; text-align: center;">50</td>
<td style="padding: 8px; text-align: center;">75</td>
<td style="padding: 8px; text-align: right;">Fulfilled</td>
</tr>
</table> )

| Product ID | Product Name | Requested Quantity | Available Quantity | Status |
| :--- |:--- | :---: | :---: |---: |

## **Notes**
1. Extract EVERY requirement from the customer email
2. Categorize them logically (Οδηγίες, Σημειώσεις etc.)
3. Use clear bullet points with emojis for visual organization
4. Keep the original Greek language from the email
5. Make it scannable and actionable for the operations team
6. Include ALL special conditions and preferences

STEP 4: CREATE PDF WITH HTML STYLING
Create pdf report from markdown you created:
- markdown: [the markdown content you created]
- filename: "order_report.pdf"

IMPORTANT: 
- Check each product ONLY ONCE
- After getting all product data, generate the markdown immediately
- Do not repeat database queries
- Use the exact template provided
- The table MUST use proper markdown syntax
"""

def setup_llm():
    """
    Initialize and configure the ChatOpenAI language model using OpenRouter API.
    
    This function loads environment variables, validates the API key, and creates
    a ChatOpenAI instance with the specified configuration.
    
    Returns:
        ChatOpenAI: Configured language model instance
        
    Raises:
        ValueError: If OPENROUTER_API_KEY is not set in environment variables
    """
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Retrieve OpenRouter API key from environment variables
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    # Validate that the API key is present
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    else:
        # Retrieve model name and API base URL from environment variables
        model = os.getenv("OPEN_ROUTER_MODEL_NAME")
        openrouter_api_base = os.getenv("OPENROUTER_API_BASE")
        
        # Create and return configured ChatOpenAI instance
        return ChatOpenAI(
            model=model,                    # Specify the language model to use
            openai_api_key=openrouter_api_key,  # Set the API key for authentication
            openai_api_base=openrouter_api_base, # Set the API endpoint URL
            temperature=0.0                 # Set temperature to 0 for deterministic outputs
        )

async def email_classification(llm: ChatOpenAI, email_body: str) -> str:
    """
    Classify an email using a language model with a predefined classification prompt.
    
    This function takes an email body and uses an LLM to classify it (with funzy way) into predefined
    categories based on the classification prompt template.
    
    Args:
        llm (ChatOpenAI): Configured language model instance for making API calls
        email_body (str): The content of the email to be classified
        
    Returns:
        str: The classified category, cleaned and standardized to uppercase
    """
    
    # Log the start of the classification process for monitoring and debugging
    logger.info("-Classification process started...")
    
    # Construct the message payload for the LLM API call
    message = [
        {
            "role": "system",  # System message sets the context and instructions
            "content": CLASSIFICATION_PROMPT.format(
                order=CLASS_ORDER,   # Insert order-related classification categories
                other=CLASS_OTHER    # Insert other classification categories
            )
        },
        {
            "role": "user",    # User message contains the actual email content to classify
            "content": email_body
        }
    ]
    
    # Asynchronously invoke the LLM to get the classification.
    # This makes an API call to the language model with the constructed messages
    email_class = await llm.ainvoke(message)
    
    # Clean and standardize the output by removing whitespace and converting to uppercase
    # This ensures consistent formatting for downstream processing
    return email_class.content.strip().upper()

async def order_process(email_body: str, agent: MCPAgent, rag_context):
    """
    Asynchronously processes an email order by extracting order details 
    and checking product availability using an MCP agent.
    
    This function:
    1. Formats a comprehensive prompt with email content and RAG context
    2. Executes the agent to handle order extraction and availability checks
    3. Returns the processed order results
    
    Args:
        email_body (str): The raw content of the order email to be processed
        agent (MCPAgent): The MCP agent instance responsible for order processing
        rag_context: Additional context from Retrieval-Augmented Generation system
                    containing product information, inventory data, etc.
    
    Returns:
        The result of the agent execution containing extracted order details
        and availability status
    """

    # Construct the formatted prompt message for the agent
    # The prompt template includes:
    # - email_body: The original email content
    # - context: Additional context from RAG system
    message = EXTRACTION_AVAILABILITY_ORDER_PROMPT.format(
        email_body=email_body,
        context=rag_context,
    )
    
    # Execute the agent asynchronously to process the order with the constructed message
    # The agent will handle order extraction and availability checking
    await agent.run(message)


async def main():
    """
    Main asynchronous function that orchestrates the email processing workflow.
    
    This function:
    1. Loads configuration and initializes services
    2. Classifies incoming emails
    3. Routes emails to appropriate processing pipelines based on classification
    4. Handles order processing with product similarity search
    """
    # Load environment variables for configuration from .env file
    load_dotenv()

    # Define the path to the MCP server configuration file
    config_file = "config/mcp_servers.json"

    # Validate that configuration file path is specified
    if not config_file:
        raise ValueError("MCP_CONFIG_FILE is not set")

    # Set up the Language Model, handling potential configuration errors
    try:
        llm = setup_llm()  # Initialize the language model with API configuration
    except ValueError as e:
        # Handle configuration errors (e.g., missing API keys) gracefully
        print(f"Error: {e}")
        return  # Exit if LLM setup fails since it's critical for operation
    
    # Initialize the MCPClient with the configuration file for tool connections
    client = MCPClient.from_dict(config_file)
    
    # Create MCPAgent instance that will use the LLM and client for tool operations
    agent = MCPAgent(
        llm=llm,           # Provide the configured language model
        client=client,      # Provide the MCP client for tool access
        max_steps=10        # Limit agent execution steps to prevent infinite loops
    )
    

    # Assign or get the email content to process
    email = "EMAIL PLACEHOLDER HERE"
    
    # Classify the email using the LLM to determine processing path
    email_class = await email_classification(llm, email)

    # Route email to appropriate processing pipeline based on classification
    if email_class == CLASS_ORDER:
        logger.info("Starting order processing...")

        # Initialize similarity service for product matching
        similarity_service = ProductSimilarityService()
        # Search for products similar to those mentioned in the email
        similar_products = similarity_service.search_and_return_similar_products(email)

        # Process the order with the agent, providing email content and similar products as context
        await order_process(email, agent, similar_products)
    elif email_class == CLASS_OTHER:
        logger.info("This is not an order email")
    else:
        logger.info('Warning: Received an unknown classification tag')

if __name__ == "__main__":
    asyncio.run(main())