import json
import re
import string
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import chromadb
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class RestaurantKnowledgeBase:
    def __init__(self, data_file='restaurant_data.json', chroma_dir: str = "chroma_db"):
        self.data_file = data_file
        self.raw_data = []
        self.processed_data = []
        self.knowledge_base = {}
        self.index = defaultdict(list)
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
            
            # Create or get collections
            self.restaurants_collection = self.chroma_client.get_or_create_collection(
                name="restaurants",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.menus_collection = self.chroma_client.get_or_create_collection(
                name="menus",
                metadata={"hnsw:space": "cosine"}
            )
            
            self.interactions_collection = self.chroma_client.get_or_create_collection(
                name="interactions",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Connected to ChromaDB at {chroma_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

        # Initialize text processing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Custom stop words for restaurant domain
        self.custom_stop_words = {'restaurant', 'cafe', 'food', 'menu', 'order', 'delivery', 
                                'location', 'contact', 'phone', 'email'}
        
        # Load data if file exists
        if os.path.exists(data_file):
            self.load_data()
        else:
            logger.warning(f"Data file {data_file} not found. Please run the scraper first.")

    def load_data(self):
        """Load scraped data from JSON file"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        logger.info(f"Loaded data for {len(self.raw_data)} restaurants")

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text for knowledge base
        - Lowercase conversion
        - Remove punctuation
        - Tokenization
        - Stopword removal
        - Lemmatization
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and token not in self.custom_stop_words:
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)
                
        return ' '.join(processed_tokens)

    def extract_price_range(self, menu_items: List[Dict]) -> str:
        """Determine price range based on menu items"""
        if not menu_items:
            return "$"
            
        prices = []
        for item in menu_items:
            price_str = item.get('price', '')
            if isinstance(price_str, str):
                # Extract numbers from price string
                nums = re.findall(r'\d+', price_str.replace(',', ''))
                if nums:
                    try:
                        price = float(nums[0])
                        prices.append(price)
                    except ValueError:
                        pass
        
        if not prices:
            return "$"
            
        avg_price = sum(prices) / len(prices)
        
        # Determine price range
        if avg_price < 15:  # Adjusted for USD
            return "$"
        elif avg_price < 30:
            return "$$"
        else:
            return "$$$"

    def extract_cuisine_type(self, restaurant_name: str, features: List[str], menu_items: List[Dict]) -> str:
        """Determine cuisine type based on restaurant name, features and menu items"""
        cuisine_keywords = {
            'american': ['american', 'burger', 'grill', 'diner', 'steakhouse'],
            'italian': ['italian', 'pizza', 'pasta', 'risotto', 'gelato'],
            'chinese': ['chinese', 'noodle', 'dimsum', 'wonton', 'szechuan'],
            'japanese': ['japanese', 'sushi', 'ramen', 'tempura', 'teriyaki', 'miso'],
            'mexican': ['mexican', 'taco', 'burrito', 'quesadilla', 'enchilada'],
            'thai': ['thai', 'curry', 'pad thai', 'basil'],
            'vietnamese': ['vietnamese', 'pho', 'banh mi'],
            'mediterranean': ['mediterranean', 'hummus', 'falafel', 'gyro', 'kebab'],
            'cafe': ['cafe', 'coffee', 'sandwich', 'pastry', 'breakfast']
        }
        
        # Combine name and menu items for analysis
        all_text = restaurant_name.lower() + ' '
        all_text += ' '.join(features).lower() + ' '
        
        # Extract menu item names and descriptions
        menu_text = ""
        for item in menu_items:
            name = item.get('name', '').lower()
            desc = item.get('description', '').lower()
            menu_text += f" {name} {desc}"
        
        all_text += menu_text
        
        # Count matches for each cuisine
        cuisine_matches = {}
        for cuisine, keywords in cuisine_keywords.items():
            count = sum(1 for keyword in keywords if keyword in all_text)
            if count > 0:
                cuisine_matches[cuisine] = count
        
        # Return the cuisine with the most matches
        if cuisine_matches:
            return max(cuisine_matches, key=cuisine_matches.get)
        
        return "other"

    def extract_dietary_info(self, item_name: str, item_description: str = "") -> List[str]:
        """Extract dietary information from menu item names and descriptions"""
        dietary_info = []
        
        # Handle cases where item_name or item_description might not be a string
        if not isinstance(item_name, str):
            item_name = str(item_name) if item_name is not None else ""
        if not isinstance(item_description, str):
            item_description = str(item_description) if item_description is not None else ""
        
        # Combine name and description
        text = (item_name + " " + item_description).lower()
        
        # Check for vegetarian indicators
        veg_keywords = ['veg', 'vegetarian', 'v', '[v]']
        if any(keyword in text for keyword in veg_keywords):
            dietary_info.append("vegetarian")
            
        # Check for vegan indicators
        vegan_keywords = ['vegan', 'plant-based', 'dairy-free', 've', '[ve]']
        if any(keyword in text for keyword in vegan_keywords):
            dietary_info.append("vegan")
            
        # Check for gluten-free indicators
        gf_keywords = ['gluten-free', 'gluten free', 'gf', '[gf]']
        if any(keyword in text for keyword in gf_keywords):
            dietary_info.append("gluten-free")
            
        return dietary_info

    def extract_structured_fields(self, restaurant: Dict) -> Dict:
        """Extract and structure key information from raw restaurant data"""
        structured = {
            'name': restaurant.get('restaurant_name', ''),
            'source_url': restaurant.get('source_url', ''),
            'scraped_at': datetime.now().isoformat(),
            'categories': set(),
            'features': set(),
            'location': '',  # Initialize as empty string and set properly below
            'menu_items': [],
            'contact_info': {},
            'opening_hours': {}
        }
        
        # Process location field carefully
        location = restaurant.get('location', '')
        if isinstance(location, str):
            structured['location'] = location
        elif isinstance(location, list):
            # Join list of strings
            location_strs = []
            for loc in location:
                if isinstance(loc, str):
                    location_strs.append(loc)
                elif isinstance(loc, dict):
                    # Extract address or other relevant info from dict
                    loc_parts = []
                    for key in ['address', 'city', 'state', 'country']:
                        if key in loc and loc[key]:
                            loc_parts.append(str(loc[key]))
                    if loc_parts:
                        location_strs.append(', '.join(loc_parts))
            structured['location'] = ', '.join(location_strs)
        elif isinstance(location, dict):
            # Extract address from dictionary
            loc_parts = []
            for key in ['address', 'city', 'state', 'country']:
                if key in location and location[key]:
                    loc_parts.append(str(location[key]))
            structured['location'] = ', '.join(loc_parts)
        
        # Process special features
        if 'special_features' in restaurant and isinstance(restaurant['special_features'], list):
            for feature in restaurant['special_features']:
                if feature:
                    # Handle special feature tags like [gf], [ve], [v]
                    if feature in ['[gf]', '[ve]', '[v]']:
                        if feature == '[gf]':
                            structured['features'].add('gluten-free')
                        elif feature == '[ve]':
                            structured['features'].add('vegan')
                        elif feature == '[v]':
                            structured['features'].add('vegetarian')
                    else:
                        processed = self.preprocess_text(feature)
                        if processed:
                            structured['features'].add(processed)
        
        # Process menu items - adapted to match restaurant_data.json structure
        if 'menu_items' in restaurant and isinstance(restaurant['menu_items'], list):
            for item in restaurant['menu_items']:
                if isinstance(item, dict):
                    item_name = item.get('name', '')
                    item_description = item.get('description', '')
                    
                    if item_name:
                        dietary_info = self.extract_dietary_info(item_name, item_description)
                        structured['menu_items'].append({
                            'name': item_name,
                            'description': item_description,
                            'processed_name': self.preprocess_text(item_name),
                            'processed_description': self.preprocess_text(item_description),
                            'dietary_info': dietary_info
                        })
        
        # Process contact info
        if 'contact_info' in restaurant:
            contact = restaurant['contact_info']
            # Check if contact_info is a dictionary
            if isinstance(contact, dict):
                structured['contact_info'] = {
                    'phone': contact.get('phone', ''),
                    'email': contact.get('email', ''),
                    'address': contact.get('address', structured['location'])
                }
            else:
                # Handle case where contact_info is a string or other type
                structured['contact_info'] = {
                    'phone': '',
                    'email': '',
                    'address': structured['location']
                }
        
        # Process opening hours
        if 'hours' in restaurant:
            # Ensure hours is a dictionary
            if isinstance(restaurant['hours'], dict):
                structured['opening_hours'] = restaurant['hours']
            else:
                # If not a dictionary, create an empty one
                structured['opening_hours'] = {}
        
        # Convert sets to lists for JSON serialization
        structured['categories'] = list(structured['categories'])
        structured['features'] = list(structured['features'])
        
        return structured

    def build_knowledge_base(self):
        """Transform raw data into structured knowledge base and store in ChromaDB"""
        logger.info("Building knowledge base...")
        
        # Clear existing collections by deleting and recreating them
        try:
            self.chroma_client.delete_collection("restaurants")
            self.chroma_client.delete_collection("menus")
            self.chroma_client.delete_collection("interactions")
        except Exception as e:
            logger.warning(f"Could not delete collections: {e}")
        
        # Recreate collections
        self.restaurants_collection = self.chroma_client.get_or_create_collection(
            name="restaurants",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.menus_collection = self.chroma_client.get_or_create_collection(
            name="menus",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.interactions_collection = self.chroma_client.get_or_create_collection(
            name="interactions",
            metadata={"hnsw:space": "cosine"}
        )

        restaurant_documents = []
        restaurant_metadatas = []
        restaurant_ids = []
        
        menu_documents = []
        menu_metadatas = []
        menu_ids = []
        
        for idx, restaurant in enumerate(self.raw_data):
            processed = self.extract_structured_fields(restaurant)
            self.processed_data.append(processed)
            
            # Create a document for text search
            document_parts = [
                processed['name'],
                ' '.join(processed['categories']),
                ' '.join(processed['features']),
                ' '.join([item['name'] + ' ' + item.get('description', '') for item in processed['menu_items']])
            ]
            document = ' '.join(document_parts)
            processed_document = self.preprocess_text(document)
            
            # Add to knowledge base with restaurant name as key
            self.knowledge_base[processed['name']] = {
                'processed_document': processed_document,
                'structured_data': processed
            }
            
            # Extract additional info for ChromaDB schema
            cuisine = self.extract_cuisine_type(processed['name'], 
                                               processed['features'], 
                                               processed['menu_items'])
            price_range = self.extract_price_range(processed['menu_items'])
            
            # Hours
            hours = processed['opening_hours'] if processed['opening_hours'] else {}
            
            # Prepare restaurant data for ChromaDB
            restaurant_id = f"rest_{idx}"
            restaurant_documents.append(processed_document)
            
            # Ensure no None values in metadata (ChromaDB requirement)
            restaurant_metadata = {
                "name": processed['name'] or "",
                "cuisine": cuisine or "",
                "price_range": price_range or "",
                "features": '|'.join(processed['features']) if processed['features'] else "",
                "location": processed['location'] if isinstance(processed['location'], str) else (', '.join(processed['location']) if isinstance(processed['location'], list) else ""),
                "hours": json.dumps(hours) if hours else "{}",
                "source_url": processed.get('source_url', '') or ""
            }
            
            # Final check to ensure no list values remain
            for key, value in restaurant_metadata.items():
                if isinstance(value, list):
                    restaurant_metadata[key] = ', '.join(map(str, value))
            
            restaurant_metadatas.append(restaurant_metadata)
            restaurant_ids.append(restaurant_id)
            
            # Prepare menu items for ChromaDB
            for item_idx, item in enumerate(processed['menu_items']):
                item_name = item['name']
                item_description = item.get('description', '')
                
                # Create comprehensive document for menu item
                item_document = f"{item_name} {item_description}"
                processed_item_document = self.preprocess_text(item_document)
                
                menu_id = f"menu_{idx}_{item_idx}"
                menu_documents.append(processed_item_document)
                
                # Ensure no None values in metadata (ChromaDB requirement)
                menu_metadata = {
                    "restaurant_id": restaurant_id,
                    "restaurant_name": processed['name'] or "",
                    "name": item_name or "",
                    "description": item_description or "",
                    "dietary_info": '|'.join(item.get('dietary_info', [])) if item.get('dietary_info') else ""
                }
                
                # Final check to ensure no list values remain
                for key, value in menu_metadata.items():
                    if isinstance(value, list):
                        menu_metadata[key] = ', '.join(map(str, value))
                
                menu_metadatas.append(menu_metadata)
                menu_ids.append(menu_id)
        
        # Add restaurants to ChromaDB
        if restaurant_documents:
            self.restaurants_collection.add(
                documents=restaurant_documents,
                metadatas=restaurant_metadatas,
                ids=restaurant_ids
            )
        
        # Add menu items to ChromaDB
        if menu_documents:
            self.menus_collection.add(
                documents=menu_documents,
                metadatas=menu_metadatas,
                ids=menu_ids
            )
        
        logger.info(f"Knowledge base created with {len(self.knowledge_base)} restaurants")
        logger.info(f"ChromaDB restaurants collection contains {self.restaurants_collection.count()} documents")
        logger.info(f"ChromaDB menus collection contains {self.menus_collection.count()} documents")

    def create_index(self):
        """Create inverted index for efficient search and compute embeddings"""
        logger.info("Creating inverted index...")
        
        # First create document vectors using TF-IDF
        documents = [data['processed_document'] for data in self.knowledge_base.values()]
        self.document_vectors = self.vectorizer.fit_transform(documents)
        
        # Build inverted index from vocabulary
        terms = self.vectorizer.get_feature_names_out()
        for i, term in enumerate(terms):
            # Get documents where this term appears (non-zero TF-IDF)
            doc_indices = self.document_vectors[:, i].nonzero()[0]
            restaurant_names = list(self.knowledge_base.keys())
            self.index[term] = [restaurant_names[idx] for idx in doc_indices]
        
        logger.info(f"Index created with {len(self.index)} terms")

    def save_knowledge_base(self, filename='restaurant_kb.pkl'):
        """Save knowledge base and index to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'knowledge_base': self.knowledge_base,
                'index': self.index,
                'vectorizer': self.vectorizer,
                'document_vectors': self.document_vectors
            }, f)
        logger.info(f"Knowledge base saved to {filename}")
        # No need to explicitly persist ChromaDB as the PersistentClient 
        # automatically persists changes to disk

    def load_knowledge_base(self, filename='restaurant_kb.pkl'):
        """Load knowledge base from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.knowledge_base = data['knowledge_base']
                self.index = data['index']
                self.vectorizer = data['vectorizer']
                self.document_vectors = data['document_vectors']
            logger.info(f"Knowledge base loaded from {filename}")
            return True
        return False

    def query_by_keyword(self, query: str, top_n: int = 5) -> List[Dict]:
        """Search restaurants by keyword using vector space model"""
        processed_query = self.preprocess_text(query)
        if not processed_query:
            return []
            
        # Vectorize query
        query_vec = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.document_vectors)
        
        # Get top matching restaurants
        restaurant_names = list(self.knowledge_base.keys())
        top_indices = np.argsort(similarities[0])[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[0][idx] > 0:
                results.append({
                    'restaurant': restaurant_names[idx],
                    'score': float(similarities[0][idx]),
                    'data': self.knowledge_base[restaurant_names[idx]]['structured_data']
                })
        
        # Log the interaction
        self.log_interaction(query, f"Found {len(results)} restaurants matching '{query}'")
        
        return results

    def query_by_cuisine(self, cuisine: str) -> List[Dict]:
        """Find restaurants that serve a specific cuisine type"""
        cuisine = cuisine.lower()
        
        # Query ChromaDB
        results = self.restaurants_collection.query(
            query_texts=[cuisine],
            n_results=10,
            where={"cuisine": {"$eq": cuisine}}
        )
        
        formatted_results = []
        for i, restaurant_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            formatted_results.append({
                'restaurant': metadata['name'],
                'data': self.knowledge_base.get(metadata['name'], {}).get('structured_data', metadata)
            })
        
        # Log the interaction
        self.log_interaction(f"cuisine:{cuisine}", f"Found {len(formatted_results)} restaurants serving '{cuisine}' cuisine")
        
        return formatted_results

    def query_by_feature(self, feature: str) -> List[Dict]:
        """Find restaurants with specific features"""
        processed_feature = self.preprocess_text(feature)
        if not processed_feature:
            return []
            
        # Query ChromaDB - need to use $in instead of $contains
        results = self.restaurants_collection.query(
            query_texts=[processed_feature],
            n_results=10
        )
        
        # Filter results manually for features
        formatted_results = []
        for i, restaurant_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            features = metadata.get('features', '').split('|')
            
            # Check if the desired feature is in the features list
            if processed_feature in features:
                formatted_results.append({
                    'restaurant': metadata['name'],
                    'data': self.knowledge_base.get(metadata['name'], {}).get('structured_data', metadata)
                })
        
        # Log the interaction
        self.log_interaction(f"feature:{feature}", f"Found {len(formatted_results)} restaurants with feature '{feature}'")
        
        return formatted_results

    def query_by_location(self, location: str) -> List[Dict]:
        """Find restaurants in a specific location/city"""
        location = location.lower()
        
        # Query ChromaDB without using $contains
        results = self.restaurants_collection.query(
            query_texts=[location],
            n_results=10
        )
        
        # Filter manually based on location
        formatted_results = []
        for i, restaurant_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            if location.lower() in metadata.get('location', '').lower():
                formatted_results.append({
                    'restaurant': metadata['name'],
                    'data': self.knowledge_base.get(metadata['name'], {}).get('structured_data', metadata)
                })
        
        # Log the interaction
        self.log_interaction(f"location:{location}", f"Found {len(formatted_results)} restaurants in location '{location}'")
        
        return formatted_results

    def query_by_price_range(self, price_range: str) -> List[Dict]:
        """Find restaurants in a specific price range"""
        if price_range not in ["$", "$$", "$$$"]:
            return []
            
        # Query ChromaDB
        results = self.restaurants_collection.query(
            query_texts=[price_range],
            n_results=10,
            where={"price_range": {"$eq": price_range}}
        )
        
        formatted_results = []
        for i, restaurant_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            formatted_results.append({
                'restaurant': metadata['name'],
                'data': self.knowledge_base.get(metadata['name'], {}).get('structured_data', metadata)
            })
        
        # Log the interaction
        self.log_interaction(f"price_range:{price_range}", 
                            f"Found {len(formatted_results)} restaurants in price range '{price_range}'")
        
        return formatted_results

    def query_menu_items(self, query: str, dietary_restrictions: Optional[List[str]] = None) -> List[Dict]:
        """Search for specific menu items across all restaurants"""
        processed_query = self.preprocess_text(query)
        if not processed_query:
            return []
            
        # Query ChromaDB without using $contains
        results = self.menus_collection.query(
            query_texts=[processed_query],
            n_results=20
        )
        
        formatted_results = []
        for i, menu_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            restaurant_name = metadata['restaurant_name']
            
            # Filter for dietary restrictions manually
            if dietary_restrictions:
                if isinstance(dietary_restrictions, str):
                    dietary_restrictions = [dietary_restrictions]
                
                # Get the dietary info from metadata
                item_dietary_info = metadata['dietary_info'].split('|') if metadata['dietary_info'] else []
                
                # Check if all required restrictions are met
                if not all(restriction in item_dietary_info for restriction in dietary_restrictions):
                    continue
            
            formatted_results.append({
                'item_name': metadata['name'],
                'description': metadata['description'],
                'restaurant': restaurant_name,
                'dietary_info': metadata['dietary_info'].split('|') if metadata['dietary_info'] else []
            })
        
        # Log the interaction
        self.log_interaction(f"menu_item:{query}", 
                            f"Found {len(formatted_results)} menu items matching '{query}'")
        
        return formatted_results

    def query_dietary_options(self, restriction: str, item_type: str = "") -> List[Dict]:
        """Find menu items that meet specific dietary restrictions"""
        restriction = restriction.lower()
        processed_item_type = self.preprocess_text(item_type)
        
        # Build query
        query_text = processed_item_type if processed_item_type else restriction
        
        # Query ChromaDB without using $contains
        results = self.menus_collection.query(
            query_texts=[query_text],
            n_results=30
        )
        
        formatted_results = []
        for i, menu_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            restaurant_name = metadata['restaurant_name']
            
            # Filter manually for dietary restriction
            item_dietary_info = metadata['dietary_info'].split('|') if metadata['dietary_info'] else []
            if restriction not in item_dietary_info:
                continue
            
            formatted_results.append({
                'item_name': metadata['name'],
                'description': metadata['description'],
                'restaurant': restaurant_name,
                'dietary_info': item_dietary_info
            })
        
        # Log the interaction
        self.log_interaction(f"dietary:{restriction}:{item_type}", 
                            f"Found {len(formatted_results)} {restriction} {item_type} menu items")
        
        return formatted_results

    def get_restaurant_details(self, name: str) -> Optional[Dict]:
        """Get complete details for a specific restaurant"""
        # Query ChromaDB
        results = self.restaurants_collection.get(
            where={"name": {"$eq": name}}
        )
        
        if not results['ids']:
            return None
        
        # Get the first matching restaurant
        metadata = results['metadatas'][0]
        
        # Get menu items for this restaurant
        menu_items = self.menus_collection.get(
            where={"restaurant_name": {"$eq": name}}
        )
        
        # Format menu items
        formatted_menu_items = []
        for item_metadata in menu_items['metadatas']:
            formatted_menu_items.append({
                "name": item_metadata['name'],
                "description": item_metadata['description'],
                "dietary_info": item_metadata['dietary_info'].split('|') if item_metadata['dietary_info'] else []
            })
        
        # Combine data
        result = dict(metadata)
        result["menu_items"] = formatted_menu_items
        
        # Parse hours from JSON string back to dict
        try:
            result["hours"] = json.loads(result["hours"])
        except:
            result["hours"] = {}
        
        # Log the interaction
        self.log_interaction(f"details:{name}", f"Retrieved details for restaurant '{name}'")
        
        return result

    def get_all_restaurants(self) -> List[str]:
        """Get list of all restaurants in knowledge base"""
        # Get all restaurants from ChromaDB
        results = self.restaurants_collection.get()
        return [metadata['name'] for metadata in results['metadatas']]

    def log_interaction(self, query: str, response: str, useful: Optional[bool] = None):
        """Log user interaction for future improvements"""
        try:
            # Build metadata dictionary, ensuring no None values
            metadata = {
                "response": response or "",
                "timestamp": datetime.now().isoformat(),
                "user_id": "anonymous"
            }
            
            # Only add 'useful' field if it's not None
            if useful is not None:
                metadata["useful"] = useful
                
            self.interactions_collection.add(
                documents=[query or ""],
                metadatas=[metadata],
                ids=[f"interaction_{datetime.now().timestamp()}"]
            )
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    def record_feedback(self, query: str, useful: bool) -> bool:
        """Record user feedback about query usefulness"""
        try:
            self.log_interaction(query, f"Feedback for query: {query}", useful)
            return True
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False

    def get_restaurant_operating_hours(self, restaurant_name: str) -> Dict:
        """Get operating hours for a specific restaurant"""
        restaurant = self.get_restaurant_details(restaurant_name)
        if not restaurant:
            return {}
        
        return restaurant.get("hours", {})
    
    def compare_restaurants(self, restaurant_names: List[str], criteria: str = "") -> Dict:
        """Compare multiple restaurants based on criteria"""
        if not criteria:
            criteria = "menu"
            
        results = {}
        for name in restaurant_names:
            restaurant = self.get_restaurant_details(name)
            if restaurant:
                results[name] = restaurant
        
        return results


def main():
    # Initialize knowledge base with ChromaDB
    kb = RestaurantKnowledgeBase(chroma_dir="chroma_db")
    
    # Check if we can load pre-built knowledge base
    if not kb.load_knowledge_base():
        # Otherwise build from scratch
        kb.build_knowledge_base()
        kb.create_index()
        kb.save_knowledge_base()
    
    # Example queries
    logger.info("\n=== Example Queries ===")
    
    # Keyword search
    logger.info("\nKeyword search for 'chicken':")
    chicken_results = kb.query_by_keyword('chicken')
    for res in chicken_results[:3]:
        logger.info(f"- {res['restaurant']} (score: {res['score']:.2f})")
    
    # Cuisine search
    logger.info("\nRestaurants with 'american' cuisine:")
    american_results = kb.query_by_cuisine('american')
    for res in american_results[:3]:
        logger.info(f"- {res['restaurant']}")
    
    # Feature search
    logger.info("\nRestaurants with 'vegan' options:")
    vegan_results = kb.query_by_feature('vegan')
    for res in vegan_results[:3]:
        logger.info(f"- {res['restaurant']}")
    
    # Location search
    logger.info("\nRestaurants in 'alhambra':")
    location_results = kb.query_by_location('alhambra')
    for res in location_results[:3]:
        logger.info(f"- {res['restaurant']}")
    
    # Price range search
    logger.info("\nRestaurants in '$' price range:")
    budget_results = kb.query_by_price_range('$')
    for res in budget_results[:3]:
        logger.info(f"- {res['restaurant']}")
    
    # Menu item search
    logger.info("\nMenu items containing 'salmon':")
    salmon_items = kb.query_menu_items('salmon')
    for res in salmon_items[:3]:
        logger.info(f"- {res['item_name']} at {res['restaurant']}")
    
    # Menu item search with dietary restrictions
    logger.info("\nVegetarian menu items:")
    veg_items = kb.query_dietary_options('vegetarian')
    for res in veg_items[:3]:
        logger.info(f"- {res['item_name']} at {res['restaurant']}")
    
    # Get full details for one restaurant
    sample_restaurant = kb.get_all_restaurants()[0]
    logger.info(f"\nFull details for '{sample_restaurant}':")
    details = kb.get_restaurant_details(sample_restaurant)
    logger.info(json.dumps(details, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()