# File: src/models/news_selector.py

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List

class NewsSelector:
    """
    A sophisticated news selector that filters news based on semantic similarity
    between an agent's logic and the candidate news items.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu', threshold=0.3, top_k=3):
        """
        Initializes the NewsSelector.
        
        Args:
            model_name (str): The name of the sentence-transformer model to use.
            device (str): The device to run the model on ('cuda' or 'cpu').
            threshold (float): The cosine similarity threshold for selecting news.
            top_k (int): The maximum number of news items to select.
        """
        print(f"Initializing NewsSelector with model: {model_name}")
        self.device = device
        # We use the same lightweight model as in the loss function for efficiency.
        # This model is not trained; it's used as a fixed feature extractor.
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        self.threshold = threshold
        self.top_k = top_k

    @torch.no_grad()
    def select(self, logic: str, candidate_news: List[str]) -> str:
        """
        Selects the most relevant news for a given logic.
        
        Args:
            logic (str): The agent's current logic string.
            candidate_news (List[str]): A list of news strings to choose from.
            
        Returns:
            str: A formatted string of the selected news, ready for the prompt.
        """
        if not candidate_news:
            return "No candidate news provided."
            
        # 1. Encode both the logic and all candidate news into embeddings.
        logic_embedding = self.embedding_model.encode(logic, convert_to_tensor=True)
        news_embeddings = self.embedding_model.encode(candidate_news, convert_to_tensor=True)
        
        # 2. Normalize embeddings to unit vectors.
        logic_embedding = F.normalize(logic_embedding, p=2, dim=0)
        news_embeddings = F.normalize(news_embeddings, p=2, dim=1)
        
        # 3. Compute cosine similarity between the logic and each news item.
        # Shape: (num_news,)
        similarities = torch.matmul(news_embeddings, logic_embedding)
        
        # 4. Filter news based on the similarity threshold and select the top_k.
        # Find indices of news that are above the threshold
        relevant_indices = torch.where(similarities > self.threshold)[0]
        
        # If no news meets the threshold, we might relax the condition or return a default message
        if len(relevant_indices) == 0:
            # Fallback: select the single most similar news item, regardless of threshold
            best_index = torch.argmax(similarities)
            selected_news = [candidate_news[best_index]]
        else:
            # Sort the relevant news by similarity score in descending order
            sorted_indices = relevant_indices[torch.argsort(similarities[relevant_indices], descending=True)]
            
            # Select the top_k most relevant news items
            top_k_indices = sorted_indices[:self.top_k]
            selected_news = [candidate_news[i] for i in top_k_indices]

        if not selected_news:
            return "No news was deemed relevant by the selection model."

        return "\n- ".join(selected_news)

if __name__ == '__main__':
    # --- This block is for testing the NewsSelector ---
    print("--- Testing NewsSelector ---")

    # 1. Setup
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    selector = NewsSelector(device=test_device, threshold=0.25, top_k=2)
    
    # 2. Define a test case
    test_logic = "Focus on government policies affecting the technology sector and supply chains."
    test_news_pool = [
        "Market sentiment is optimistic, tech stocks are soaring.", # Relevant
        "Bad News: A key local factory has halted production due to supply chain issues.", # Relevant
        "Weather Forecast: Continued sunny skies expected for the next week.", # Irrelevant
        "Good News: The government has announced a new economic stimulus package.", # Semi-relevant
        "Sports Update: The home team won the championship.", # Irrelevant
        "The Federal Reserve hinted at new regulations for major tech companies.", # Highly relevant
    ]
    
    # 3. Perform selection
    print(f"\nAgent Logic: '{test_logic}'")
    print("\nCandidate News:")
    for n in test_news_pool:
        print(f"  - {n}")
        
    selected_string = selector.select(test_logic, test_news_pool)
    
    # 4. Print and verify the result
    print("\n--- Selected News ---")
    print(selected_string)
    
    # Verification
    assert "regulations for major tech companies" in selected_string, "Highly relevant news was missed."
    assert "supply chain issues" in selected_string, "Relevant news was missed."
    assert "Sports Update" not in selected_string, "Irrelevant news was selected."
    assert len(selected_string.split('\n- ')) <= selector.top_k, f"Should select at most {selector.top_k} news."

    print("\n--- Testing fallback mechanism (no news above threshold) ---")
    selector_high_thresh = NewsSelector(device=test_device, threshold=0.9, top_k=2)
    fallback_selection = selector_high_thresh.select(test_logic, test_news_pool)
    print(fallback_selection)
    # The result should be the single most relevant news, which is the one about regulations
    assert "regulations for major tech companies" in fallback_selection
    assert '\n' not in fallback_selection # Should only be one line

    print("\nNewsSelector test completed successfully!")