# Zomato Restaurant RAG Chatbot

## Overview
A Retrieval Augmented Generation (RAG) chatbot for restaurant information, created for the Zomato Gen AI Internship Assignment. This system combines web scraping, vector databases, and large language models to answer natural language questions about restaurants with accurate, contextual responses.

## Description
This project builds an end-to-end Generative AI solution that enables users to ask questions like:
- "Which restaurant has the best vegetarian options in their menu?"
- "Does ABC restaurant have any gluten-free appetizers?"
- "What's the location for XYZ restaurant?"
- "Compare the menus of restaurants A and B"

## Features

### Multi-Agentic RAG Architecture
- **Relevancy Analysis Agent**: Determines if user queries are related to restaurant information
- **Retrieval Agent**: Identifies and fetches the most relevant restaurant data from the knowledge base
- **Web Search Tool**: Augments knowledge base with real-time web data when local information is insufficient
- **Generation Agent**: Synthesizes retrieved data into natural, conversational responses

### Intelligent Web Scraping Pipeline
- **Crawler Agent**: Systematically extracts raw HTML content from restaurant websites
- **Extractive Agent**: Uses LLMs to identify and structure relevant restaurant information
- **Two-Layer Extraction**: Combines traditional scraping tools with AI-based content filtering
- **Domain-Respectful Design**: Follows robots.txt rules and implements rate limiting

### Advanced Knowledge Base
- **Vector Database Integration**: Uses ChromaDB for efficient semantic search
- **Multi-Collection Architecture**: Specialized collections for restaurants, menus, and user interactions
- **Advanced Text Processing**: Implements tokenization, lemmatization, and stopword removal
- **Dietary Information Extraction**: Automatically identifies and categorizes dietary restrictions

### Natural Language Understanding
- **Query Classification**: Identifies the type and intent of user questions
- **Contextual Response Generation**: Maintains conversation history for coherent interactions
- **Format-Aware Responses**: Adapts output style based on query type (lists, comparisons, etc.)
- **Error Handling**: Gracefully manages edge cases and ambiguous queries

### User Experience
- **Intuitive Chat Interface**: Built with Gradio for easy interaction
- **Example Suggestions**: Provides sample questions to guide users
- **Alternative UI Option**: Includes Streamlit implementation for different deployment needs

## Project Structure
- `web_scrapper.py`: Coordinates the scraping pipeline
- `scrape_all.py`: Implements website crawling functionality
- `agentic_extractor.py`: Uses LLM to extract structured data
- `knowledge_base.py`: Vector database implementation with ChromaDB
- `rag_chatbot.py`: RAG chatbot with Gradio interface

## Installation

### Requirements
- Python 3.8+ 
- Chrome browser (for Selenium)

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/your-username/Zomato_Restaurant_ChatBot.git
   cd Zomato_Restaurant_ChatBot
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set environment variables:
- Create a `.env` file with the following:
  ```
  GROQ_API_KEY=your_groq_api_key
  SERPER_API_KEY=your_serper_api_key
  HF_API_TOKEN=your_hf_api_key
  ```
  
## Usage

### 1. Web Scraping
  To collect restaurant data:
      
      ```
      python web_scrapper.py
      ```
  This will:
- Crawl the configured restaurant websites
- Extract structured data using the agentic extractor
- Save the results to `restaurant_data.json`

### 2. Build Knowledge Base
  To create the vector database:
      
      ```
      python knowledge_base.py
      ```
  This will:
- Process the raw restaurant data
- Create ChromaDB collections for efficient retrieval
- Save the knowledge base to `restaurant_kb.pkl`

### 3. Run Chatbot
  To start the chatbot interface:
     
     ```
     python rag_chatbot.py
     ```
  This will:
- Initialize the chatbot with the knowledge base
- Start a Gradio web server (default: http://localhost:7860)
- Provide a chat interface for interacting with the system

## Demo-Video
https://github.com/user-attachments/assets/00d633a5-1f88-4de8-9126-85ff080c536a

## Scraped Dataset
 - Attached in the files
 - 
## Link to the Website
 - https://huggingface.co/spaces/AaryaPakhale/Zomato_Restaurant_ChatBot
 - 
## System Architecture
https://github.com/user-attachments/assets/e7e70d80-596b-4b67-833b-afc0c4cf488f

## Technologies Used
- **Web Scraping**: BeautifulSoup4, Selenium
- **Vector Database**: ChromaDB
- **NLP Processing**: NLTK, scikit-learn
- **LLM Integration**: Groq API (Llama3, Mixtral)
- **User Interfaces**: Gradio, Streamlit (alternative)
- **API Integration**: Serper for web search
  
# Challenges Faced and Solutions Implemented

## Web Scraping Challenges

### Challenge: Dynamic JavaScript Content
 - Modern restaurant websites heavily rely on JavaScript to render content, making traditional HTML scraping ineffective.

 - **Solution:** Implemented a dual-approach system with Selenium for JavaScript-rendered content and BeautifulSoup for static content. Added appropriate wait times to ensure full page rendering before extraction.

### Challenge: Inconsistent Website Structures
 - Each restaurant website has unique layouts, making it difficult to create a universal scraper.

 - **Solution:** Developed an LLM-based agentic extractor that understands context and can identify restaurant information regardless of the specific HTML structure or naming conventions.

### Challenge: Rate Limiting and Anti-Scraping Measures
 - Some websites implement measures to prevent scraping.

 - **Solution:** Built robust retry mechanisms with exponential backoff, respected robots.txt policies, implemented appropriate request delays, and rotated user agents to mimic human browsing behavior.

## Data Processing Challenges

### Challenge: Unstructured and Inconsistent Data
 - Raw extracted data varied widely in format, completeness, and terminology.

 - **Solution:** Created a comprehensive text preprocessing pipeline with normalization, tokenization, and lemmatization to standardize data. Used the LLM to identify and categorize information regardless of how it was presented on the source website.

### Challenge: Menu Item Classification
 - Identifying dietary restrictions and special features from menu descriptions was difficult due to inconsistent labeling.

 - **Solution:** Implemented a specialized dietary information extractor that recognizes various ways restaurants indicate features like vegetarian, vegan, or gluten-free options, using pattern matching and keyword identification.

### Challenge: Missing or Partial Information
 - Some restaurant websites had incomplete information about hours, locations, or menu details.

 - **Solution:** Created a fallback system that combines data across multiple pages and leverages web search to fill information gaps. The system also explicitly acknowledges uncertainty when information is incomplete.

## RAG Implementation Challenges

### Challenge: Query Understanding
 - User queries can be ambiguous or contain multiple intents.

 - **Solution:** Implemented a relevancy analysis agent that correctly identifies the query intent and type before retrieval, allowing for more targeted information retrieval and response generation.

### Challenge: Balancing Retrieval Precision and Recall
 - Finding the right information without retrieving excessive irrelevant content was difficult.

 - **Solution:** Designed a multi-collection vector database architecture that separates restaurants and menu items, enabling more focused retrieval. Implemented custom similarity thresholds that adapt based on query type.

### Challenge: Response Quality with Limited Data
 - When the knowledge base had insufficient information, responses could be incomplete or unhelpful.

 - **Solution:** Created a sufficiency detector that determines when knowledge base information is inadequate and automatically augments results with web search data. The response generator then synthesizes information from both sources.

## Technical Implementation Challenges

### Challenge: Large-Scale Vector Storage
 - As the database grew, efficient vector search became more challenging.

 - **Solution:** Integrated ChromaDB with custom collection structures and metadata filtering to maintain fast retrieval times even with growing data volumes.

### Challenge: API Cost Management
 - LLM API calls can be expensive, especially with high volumes.

 - **Solution:** Implemented tiered architecture where simpler queries are resolved with basic keyword matching, reserving LLM calls for complex tasks. Added token tracking and rate limiting to prevent unnecessary API usage.

### Challenge: Deployment Resource Constraints
 - The full system requires significant resources to run all components simultaneously.

 - **Solution:** Designed a modular architecture that separates the web scraping, knowledge base creation, and chatbot components. This allows for staged execution on limited hardware and more efficient resource allocation.

## Future Improvements
- Expand restaurant database with more diverse options
- Implement more advanced conversation history tracking
- Add multimedia content (restaurant images, maps)
- Support for reservation queries and booking integration
- Implement user feedback mechanisms for response quality



## License
MIT

## Author
Aarya Yogesh Pakhale


   
