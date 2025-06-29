import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Any


class SimilarityEngine:
    """
    Engine for finding similar tickets using TF-IDF and cosine similarity
    """

    def __init__(self, tickets_df: pd.DataFrame):
        """
        Initialize the similarity engine with tickets data

        Args:
            tickets_df: DataFrame containing ticket data
        """
        self.tickets_df = tickets_df
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=self._get_portuguese_stopwords(),
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95)

        # Prepare text data for vectorization
        self.ticket_texts = self._prepare_ticket_texts()

        # Fit vectorizer and create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.ticket_texts)

    def _get_portuguese_stopwords(self) -> List[str]:
        """
        Get Portuguese stopwords for better text processing
        """
        return [
            'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles',
            'aquilo', 'as', 'até', 'com', 'como', 'da', 'das', 'de', 'dela',
            'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'ela',
            'elas', 'ele', 'eles', 'em', 'entre', 'essa', 'essas', 'esse',
            'esses', 'esta', 'estas', 'este', 'estes', 'eu', 'foi', 'for',
            'foram', 'há', 'isso', 'isto', 'já', 'mais', 'mas', 'me', 'mesmo',
            'meu', 'meus', 'minha', 'minhas', 'muito', 'na', 'nas', 'não',
            'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa',
            'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por',
            'qual', 'quando', 'que', 'quem', 'são', 'se', 'sem', 'ser', 'seu',
            'seus', 'só', 'sua', 'suas', 'também', 'te', 'tem', 'ter', 'todo',
            'todos', 'tu', 'tua', 'tuas', 'tudo', 'um', 'uma', 'você', 'vocês'
        ]

    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for better similarity matching
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters but keep accented characters
        text = re.sub(r'[^\w\sáàâãéèêíìîóòôõúùûç]', ' ', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _prepare_ticket_texts(self) -> List[str]:
        """
        Prepare ticket texts by combining relevant fields
        """
        texts = []

        for _, ticket in self.tickets_df.iterrows():
            # Combine subject and description for better matching
            combined_text = ""

            if 'subject' in ticket:
                combined_text += self._clean_text(ticket['subject']) + " "

            if 'description' in ticket:
                combined_text += self._clean_text(ticket['description']) + " "

            # Add other relevant fields if available
            for field in ['category', 'priority', 'status']:
                if field in ticket and pd.notna(ticket[field]):
                    combined_text += self._clean_text(ticket[field]) + " "

            texts.append(combined_text.strip())

        return texts

    def find_similar_tickets(self,
                             query_text: str,
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar tickets to the given query text

        Args:
            query_text: Text to search for similar tickets
            top_k: Number of top similar tickets to return

        Returns:
            List of dictionaries containing ticket data and similarity scores
        """
        if not query_text.strip():
            return []

        # Clean and vectorize query text
        cleaned_query = self._clean_text(query_text)
        query_vector = self.vectorizer.transform([cleaned_query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector,
                                         self.tfidf_matrix).flatten()

        # Get top k similar tickets
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]

            # Skip very low similarity scores
            if similarity_score < 0.1:
                continue

            ticket_data = self.tickets_df.iloc[idx].to_dict()

            results.append({
                'similarity_score': float(similarity_score),
                'ticket_data': ticket_data,
                'ticket_index': int(idx)
            })

        return results

    def get_feature_importance(self, query_text: str,
                               ticket_index: int) -> Dict[str, float]:
        """
        Get feature importance for understanding why tickets are similar

        Args:
            query_text: Original query text
            ticket_index: Index of the ticket to analyze

        Returns:
            Dictionary of features and their importance scores
        """
        cleaned_query = self._clean_text(query_text)
        query_vector = self.vectorizer.transform([cleaned_query])
        ticket_vector = self.tfidf_matrix[ticket_index]

        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # Calculate element-wise product
        importance_scores = query_vector.multiply(
            ticket_vector).toarray().flatten()

        # Get top important features
        top_features = {}
        for idx, score in enumerate(importance_scores):
            if score > 0:
                top_features[feature_names[idx]] = float(score)

        # Sort by importance
        sorted_features = dict(
            sorted(top_features.items(), key=lambda x: x[1], reverse=True))

        return dict(list(
            sorted_features.items())[:10])  # Return top 10 features
