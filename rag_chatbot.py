import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import gradio as gr
import requests
from knowledge_base import RestaurantKnowledgeBase
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Keys - Set these in your environment variables for security
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

class RestaurantChatbot:
    def __init__(self, knowledge_base_path: str = 'restaurant_kb.pkl', chroma_dir: str = 'chroma_db'):
        """Initialize the RAG chatbot with knowledge base and APIs"""
        
        # Initialize knowledge base
        logger.info("Initializing knowledge base...")
        self.kb = RestaurantKnowledgeBase(chroma_dir=chroma_dir)
        
        # Load pre-built knowledge base if available
        if self.kb.load_knowledge_base(knowledge_base_path):
            logger.info("Knowledge base loaded successfully")
        else:
            logger.error(f"Could not load knowledge base from {knowledge_base_path}")
            raise ValueError("Knowledge base not found")
        
        # Conversation history
        self.conversation_history = []
        
        logger.info("Chatbot initialization complete")
    
    def is_query_relevant(self, query: str) -> bool:
        """Determine if a query is relevant to restaurant information using Groq API"""
        
        # Fast keyword check first for efficiency
        restaurant_keywords = [
            "restaurant", "food", "menu", "eat", "dining", "cuisine", 
            "dish", "meal", "dinner", "lunch", "breakfast", "brunch",
            "vegetarian", "vegan", "gluten", "allergen", "price", 
            "reservation", "location", "hours", "open", "chef",
            "dessert", "appetizer", "drink", "beverage"
        ]
        
        if any(keyword in query.lower() for keyword in restaurant_keywords):
            return True
        
        # For more ambiguous queries, use Groq's classification
        try:
            prompt = f"""Determine if this user question is relevant to restaurants or food information.
            
Question: "{query}"

Rate this on a scale of 1-10 where:
1-3: Not relevant to restaurants or food
4-7: Somewhat relevant to restaurants or food
8-10: Highly relevant to restaurants or food

Output only a single number rating (1-10) and nothing else."""

            response = self._call_groq_api(prompt, max_tokens=5)
            
            # Try to extract a number from the response
            try:
                relevance_score = int(response.strip())
                return relevance_score >= 5  # Consider 5+ as relevant
            except ValueError:
                # If we can't parse a number, be conservative and assume it might be relevant
                return True
                
        except Exception as e:
            logger.warning(f"Error in relevance check: {e}. Assuming query is relevant.")
            return True  # Fail open - assume it's relevant if API fails
    
    def search_web(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search the web using Serper API for additional information"""
        try:
            if not SERPER_API_KEY:
                logger.warning("No Serper API key found. Skipping web search.")
                return []
                
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            
            # Add 'restaurant' to the query for better results
            search_query = f"restaurant {query}"
            payload = {
                "q": search_query,
                "num": num_results
            }
            
            response = requests.post(
                'https://google.serper.dev/search',
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                results = response.json().get('organic', [])
                return [
                    {
                        'title': result.get('title', ''),
                        'body': result.get('snippet', '')
                    } 
                    for result in results
                ]
            else:
                logger.error(f"Serper API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def query_knowledge_base(self, query: str) -> Tuple[List[Dict], bool]:
        """Query the knowledge base and determine if information is sufficient"""
        
        # First try keyword search
        results = self.kb.query_by_keyword(query)
        
        # If no results, try menu items search
        if not results:
            menu_results = self.kb.query_menu_items(query)
            if menu_results:
                return menu_results, True
        
        # Check for dietary questions
        dietary_keywords = {"vegetarian", "vegan", "gluten-free", "gluten free", "allergen", "allergy"}
        if any(kw in query.lower() for kw in dietary_keywords):
            for kw in dietary_keywords:
                if kw in query.lower():
                    dietary_results = self.kb.query_dietary_options(kw.replace(" ", "-"))
                    if dietary_results:
                        return dietary_results, True
        
        # Check if information is sufficient (at least 2 relevant results)
        is_sufficient = len(results) >= 2
        
        return results, is_sufficient
    
    def _call_groq_api(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Helper method to call Groq API"""
        if not GROQ_API_KEY:
            raise ValueError("Groq API key not found. Please set the GROQ_API_KEY environment variable.")
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        # Groq supports multiple models - choose the appropriate one for your needs
        # LLama-3 is extremely fast and efficient
        payload = {
            "model": "llama3-8b-8192",  # You can also use "mixtral-8x7b-32768" for better quality
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Implement exponential backoff for rate limiting
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429:  # Rate limit error
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit. Retrying in {delay} seconds.")
                    time.sleep(delay)
                else:
                    logger.error(f"Groq API error: {response.status_code} - {response.text}")
                    raise Exception(f"API error: {response.status_code}")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Error calling Groq API: {e}. Retrying in {delay} seconds.")
                time.sleep(delay)
                
        raise Exception("Failed to get response from Groq API after multiple retries")
    
    def generate_response(self, query: str, kb_results: List[Dict], web_results: Optional[List[Dict]] = None) -> str:
        """Generate a natural language response using Groq API"""
        
        # Prepare context from knowledge base
        kb_context = ""
        
        if kb_results:
            kb_context = "Information from our restaurant database:\n"
            for i, result in enumerate(kb_results[:3], 1):
                if 'restaurant' in result:
                    kb_context += f"{i}. Restaurant: {result['restaurant']}\n"
                    if 'data' in result and isinstance(result['data'], dict):
                        data = result['data']
                        if 'location' in data:
                            kb_context += f"   Location: {data['location']}\n"
                        if 'opening_hours' in data and data['opening_hours']:
                            kb_context += f"   Hours: {json.dumps(data['opening_hours'])}\n"
                        if 'features' in data and data['features']:
                            kb_context += f"   Features: {', '.join(data['features'])}\n"
                
                # Include menu items if available
                elif 'item_name' in result:
                    kb_context += f"{i}. Menu Item: {result['item_name']}\n"
                    if 'description' in result and result['description']:
                        kb_context += f"   Description: {result['description']}\n"
                    if 'restaurant' in result:
                        kb_context += f"   Available at: {result['restaurant']}\n"
                    if 'dietary_info' in result and result['dietary_info']:
                        kb_context += f"   Dietary Information: {', '.join(result['dietary_info'])}\n"
        
        # Add web search context if provided
        web_context = ""
        if web_results:
            web_context = "\nAdditional information found online:\n"
            for i, result in enumerate(web_results[:2], 1):
                web_context += f"{i}. {result['title']}: {result['body'][:200]}...\n"
        
        # Combine contexts
        context = kb_context + web_context
        
        # Create prompt for LLM
        prompt = f"""You are a helpful restaurant information chatbot named "Zomato Assistant". Use the following information to answer the user's question.
        
{context}

User Question: {query}

Instructions:
1. Provide a concise and helpful response based on the information above.
2. If the information doesn't fully answer the question, acknowledge what you know and what you're unsure about.
3. If there's no relevant information available, politely state that you don't have that specific information.
4. Do not make up or hallucinate any information that isn't provided in the context.
5. Be conversational but precise.

Your response:"""
        
        # Generate response with Groq API
        try:
            response = self._call_groq_api(prompt, max_tokens=512)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."
    
    def process_query(self, query: str) -> str:
        """Main method to process a user query and return a response"""
        
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Check if query is relevant
        if not self.is_query_relevant(query):
            response = "I'm a restaurant information assistant. I can help you with questions about restaurants, menus, dietary options, and more. Could you please ask me something related to restaurants or food?"
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Query knowledge base
        kb_results, is_sufficient = self.query_knowledge_base(query)
        
        # If knowledge base info is insufficient, search the web
        web_results = None
        if not is_sufficient:
            web_results = self.search_web(query)
        
        # Generate response
        response = self.generate_response(query, kb_results, web_results)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

# Gradio Interface
def create_gradio_interface(chatbot: RestaurantChatbot):
    """Create a Gradio interface for the chatbot"""
    
    # Welcome message
    welcome_message = """
    # 🍽️ Zomato Restaurant Assistant
    
    Hello! I'm your Zomato's restaurant assistant. I can help you with:
    
    - Finding restaurants with specific features or cuisine
    - Menu items and dietary information
    - Restaurant hours and locations
    - Comparing restaurant options
    
    Ask me anything about our restaurants!
    """
    
    # Chatbot function for Gradio
    def respond(message, history):
        return chatbot.process_query(message)
    
    # Create Gradio interface with more styling
    interface = gr.ChatInterface(
        respond, 
        title="Zomato Restaurant Assistant",
        description=welcome_message,
        theme="soft",
        examples=[
            "Which restaurants offer vegetarian options?",
            "Does Taj have gluten-free desserts?",
            "What's the price range for Paradise Biryani?",
            "Compare the spice levels in dishes at Spice Affair and Taj",
            "What time does Blue Ginger close on weekends?"
        ]
    )
    
    return interface

def main():
    # Initialize the chatbot
    try:
        chatbot = RestaurantChatbot()
        interface = create_gradio_interface(chatbot)
        interface.launch(share=True, server_name="0.0.0.0", server_port=7860)
        
    except Exception as e:
        logger.error(f"Error starting chatbot: {e}")
        raise

if __name__ == "__main__":
    main()