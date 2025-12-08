# File: src/models/news_selector.py

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Any
import os
import openai
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsSelector:
    """
    A sophisticated news selector that filters news based on either:
    1. Cosine similarity (local embedding model).
    2. API-based reasoning (OpenAI GPT).
    """
    def __init__(self, method='cosine', model_name='all-MiniLM-L6-v2', device='cpu', 
                 threshold=0.3, top_k=3, api_key=None, api_base=None, api_model="gpt-4"):
        """
        Initializes the NewsSelector.
        
        Args:
            method (str): 'cosine' or 'api'.
            model_name (str): Sentence-transformer model name (for cosine).
            device (str): Device for local model.
            threshold (float): Cosine similarity threshold.
            top_k (int): Max news items to select.
            api_key (str): OpenAI API key (optional if in env).
            api_base (str): OpenAI API base URL (optional).
            api_model (str): OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo').
        """
        self.method = method
        self.top_k = top_k
        self.device = device
        
        # --- Cosine Setup ---
        self.threshold = threshold
        self.model_name = model_name
        self.embedding_model = None # Lazy load only if needed

        # --- API Setup ---
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.api_model = api_model
        
        if self.method == 'cosine':
            logger.info(f"Initializing NewsSelector in COSINE mode with model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name, device=self.device)
        elif self.method == 'api':
             logger.info(f"Initializing NewsSelector in API mode with model: {api_model}")
             if not self.api_key:
                 logger.warning("NewsSelector initialized in API mode but no API key provided. Ensure it is set in env vars.")
        else:
            raise ValueError(f"Unknown news selection method: {self.method}")

    def select(self, logic: str, candidate_news: Union[List[str], List[Dict], str]) -> str:
        """
        Dispatches to the configured selection method.
        """
        if self.method == 'cosine':
            return self.select_by_cosine(logic, candidate_news)
        elif self.method == 'api':
            return self.select_by_api(logic, candidate_news)
        else:
            return "Invalid selection method configured."

    def select_by_api(self, logic: str, candidate_news: Union[List[str], List[Dict], str]) -> str:
        """
        Uses OpenAI API to select relevant news based on a short logic prompt.
        
        Args:
            logic: Short logic/strategy (e.g., "Focus on extreme weather events")
            candidate_news: List of news dicts or strings
            
        Returns:
            Selected news as formatted string
        """
        # Prepare candidate news for prompt
        news_list = []
        if isinstance(candidate_news, list):
            if candidate_news and isinstance(candidate_news[0], dict):
                for i, item in enumerate(candidate_news):
                    summary = item.get('summary') or item.get('content') or item.get('title') or str(item)
                    time_str = item.get('publication_time', '')
                    category = item.get('category', '')
                    news_list.append(f"{i+1}. [{time_str}] {summary}")
            else:
                news_list = [f"{i+1}. {item}" for i, item in enumerate(candidate_news)]
        else:
            news_list = [str(candidate_news)]
        
        news_context = "\n".join(news_list)

        # Simple prompt: use short logic to filter news
        prompt = f"""You are a news filtering assistant for time series forecasting.

Agent's Strategy: "{logic}"

Candidate News:
{news_context}

Task: Based on the agent's strategy, select the most relevant news items from the candidates above. Return ONLY the numbers of selected news (e.g., "1, 3, 5"). If no news is relevant, return "None".

Selected News Numbers:"""

        try:
            if self.api_key:
                openai.api_key = self.api_key
            if self.api_base:
                openai.api_base = self.api_base
            
            response = openai.ChatCompletion.create(
                model=self.api_model,
                messages=[
                    {"role": "system", "content": "You are a concise news filtering assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            # Parse response to get selected indices
            content = response["choices"][0]["message"]["content"].strip()
            
            if "None" in content or not content:
                return "No relevant news selected by API."
            
            # Extract numbers from response
            import re
            numbers = re.findall(r'\d+', content)
            selected_indices = [int(n) - 1 for n in numbers if 0 < int(n) <= len(news_list)]
            
            if not selected_indices:
                return "No relevant news selected by API."
            
            # Return selected news
            if isinstance(candidate_news, list) and candidate_news and isinstance(candidate_news[0], dict):
                selected_news = [candidate_news[i].get('summary', str(candidate_news[i])) for i in selected_indices if i < len(candidate_news)]
            else:
                selected_news = [candidate_news[i] if isinstance(candidate_news, list) else str(candidate_news) for i in selected_indices if i < len(news_list)]
            
            return "\n- ".join(selected_news) if selected_news else "No relevant news selected."
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error using API for news selection: {e}"


    @torch.no_grad()
    def select_by_cosine(self, logic: str, candidate_news: Union[List[str], List[Dict]]) -> str:
        """
        Selects news using cosine similarity.
        """
        # Pre-process candidates to strings if they are dicts
        news_strings = []
        original_map = [] # Map index back to original object if needed
        
        if not candidate_news:
            return "No candidate news provided."
            
        if isinstance(candidate_news, list):
             if candidate_news and isinstance(candidate_news[0], dict):
                for item in candidate_news:
                    # Use summary or title for embedding
                    val = item.get('summary') or item.get('title') or str(item)
                    news_strings.append(val)
                    original_map.append(item)
             else:
                news_strings = candidate_news
                original_map = candidate_news
        else:
             # If string, split by newlines? Assumption: candidate_news should be a list for cosine
             # But if it's a huge string, let's treat it as one item
             news_strings = [str(candidate_news)]
             original_map = [str(candidate_news)]

        if not news_strings:
             return "No candidate news to select from."

        # 1. Encode
        logic_embedding = self.embedding_model.encode(logic, convert_to_tensor=True)
        news_embeddings = self.embedding_model.encode(news_strings, convert_to_tensor=True)
        
        # 2. Normalize
        logic_embedding = F.normalize(logic_embedding, p=2, dim=0)
        news_embeddings = F.normalize(news_embeddings, p=2, dim=1)
        
        # 3. Similarity
        similarities = torch.matmul(news_embeddings, logic_embedding)
        
        # 4. Filter
        relevant_indices = torch.where(similarities > self.threshold)[0]
        
        if len(relevant_indices) == 0:
            # Fallback to top 1
            best_index = torch.argmax(similarities)
            selection = news_strings[best_index]
            # Try to return original object representation if dict, but the signature says return str.
            # So we return the string content.
            return f"- {selection}"
        else:
            sorted_indices = relevant_indices[torch.argsort(similarities[relevant_indices], descending=True)]
            top_k_indices = sorted_indices[:self.top_k]
            selected_items = [news_strings[i] for i in top_k_indices]
            return "\n- ".join(selected_items)

if __name__ == '__main__':
    # --- Test Block ---
    print("--- Testing NewsSelector (Cosine) ---")
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    # Test Cosine
    selector_cos = NewsSelector(method='cosine', device=test_device, threshold=0.25, top_k=2)
    logic = "Impact of extreme weather on power grid."
    news_pool = [
        "A heatwave is sweeping across the country.",
        "The new iPhone was released today.",
        "Heavy storms damaged power lines in the north.",
        "Local sports team won the finals."
    ]
    print(f"Logic: {logic}")
    print(f"Selected (Cosine):\n{selector_cos.select(logic, news_pool)}")

    print("\n--- Testing NewsSelector (API - Mock) ---")
    # We won't actually call the API in this auto-test unless we have a key, 
    # but we can initialize it to check for syntax errors.
    try:
        selector_api = NewsSelector(method='api', api_key="sk-fake-key")
        print("API Selector initialized successfully.")
    except Exception as e:
        print(f"API Selector init failed: {e}")
