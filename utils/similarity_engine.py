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
            max_features=10000,  # Aumentado para capturar mais termos
            stop_words=self._get_combined_stopwords(),  # Stopwords bilíngues
            ngram_range=(1, 3),  # Incluindo trigramas para frases como "Array boundary exceeded"
            min_df=1,  # Mantém termos que aparecem pelo menos 1 vez
            max_df=0.9,  # Reduzido para manter mais termos técnicos
            analyzer='word',
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b',  # Padrão mais simples para capturar termos técnicos
            binary=False,  # Usa frequência TF-IDF real
            sublinear_tf=True,  # Aplica log scaling para melhor performance
            norm='l2')  # Normalização L2

        # Prepare text data for vectorization
        self.ticket_texts = self._prepare_ticket_texts()

        # Fit vectorizer and create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.ticket_texts)

    def _get_combined_stopwords(self) -> List[str]:
        """
        Get combined Portuguese and English stopwords for bilingual text processing
        """
        portuguese_stopwords = [
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
        
        english_stopwords = [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'but', 'or', 'this', 'these',
            'they', 'been', 'their', 'said', 'each', 'which', 'she', 'do',
            'how', 'her', 'my', 'his', 'our', 'can', 'had', 'there', 'we',
            'what', 'when', 'where', 'who', 'why', 'would', 'could', 'should'
        ]
        
        return portuguese_stopwords + english_stopwords

    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for better similarity matching
        Preserves technical terms and English phrases
        """
        if pd.isna(text):
            return ""

        # Convert to string and preserve original casing for technical terms
        text = str(text)
        
        # Replace common separators with spaces to help with tokenization
        text = re.sub(r'[/_\-]', ' ', text)
        
        # Remove excessive punctuation but preserve some that might be important
        text = re.sub(r'[^\w\sáàâãéèêíìîóòôõúùûç.,;:()\[\]{}"]', ' ', text)
        
        # Convert to lowercase only after preserving structure
        text = text.lower()
        
        # Normalize whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove standalone punctuation
        text = re.sub(r'\s+[.,;:()\[\]{}\"]\s+', ' ', text)
        
        return text

    def _prepare_ticket_texts(self) -> List[str]:
        """
        Prepare ticket texts by combining relevant fields with weighted importance
        """
        texts = []

        for _, ticket in self.tickets_df.iterrows():
            # Combine fields with different weights (repetition increases importance)
            combined_text = ""

            # Subject is very important - repeat 3 times
            if 'subject' in ticket and pd.notna(ticket['subject']):
                subject_clean = self._clean_text(ticket['subject'])
                combined_text += f"{subject_clean} {subject_clean} {subject_clean} "

            # Description is most important - repeat 2 times  
            if 'description' in ticket and pd.notna(ticket['description']):
                desc_clean = self._clean_text(ticket['description'])
                combined_text += f"{desc_clean} {desc_clean} "

            # Solution/notes are also important
            solution_fields = ['solution', 'Últimas notas', 'ultimas_notas', 'notes']
            for field in solution_fields:
                if field in ticket and pd.notna(ticket[field]):
                    solution_clean = self._clean_text(ticket[field])
                    combined_text += f"{solution_clean} "
                    break  # Only use the first available solution field

            # Add other relevant fields once
            other_fields = ['category', 'priority', 'status', 'author', 'client', 'system']
            for field in other_fields:
                if field in ticket and pd.notna(ticket[field]):
                    field_clean = self._clean_text(ticket[field])
                    combined_text += f"{field_clean} "

            texts.append(combined_text.strip())

        return texts

    def find_similar_tickets(self,
                             query_text: str,
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar tickets to the given query text with improved matching

        Args:
            query_text: Text to search for similar tickets
            top_k: Number of top similar tickets to return

        Returns:
            List of dictionaries containing ticket data and similarity scores
        """
        if not query_text.strip():
            return []

        # Clean and vectorize query text with repetition for importance
        cleaned_query = self._clean_text(query_text)
        # Repeat query to increase weight of search terms
        enhanced_query = f"{cleaned_query} {cleaned_query}"
        query_vector = self.vectorizer.transform([enhanced_query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Also try exact phrase matching for better precision
        exact_matches = self._find_exact_phrase_matches(query_text, similarities)

        # Get top k similar tickets
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more candidates

        results = []
        seen_indices = set()
        
        # First add exact matches with boosted scores
        for idx, boosted_score in exact_matches:
            if idx not in seen_indices:
                ticket_data = self.tickets_df.iloc[idx].to_dict()
                results.append({
                    'similarity_score': float(boosted_score),
                    'ticket_data': ticket_data,
                    'ticket_index': int(idx)
                })
                seen_indices.add(idx)

        # Then add regular similarity matches
        for idx in top_indices:
            if len(results) >= top_k:
                break
                
            if idx in seen_indices:
                continue
                
            similarity_score = similarities[idx]

            # Lower threshold to catch more potential matches
            if similarity_score < 0.05:
                continue

            ticket_data = self.tickets_df.iloc[idx].to_dict()

            results.append({
                'similarity_score': float(similarity_score),
                'ticket_data': ticket_data,
                'ticket_index': int(idx)
            })
            seen_indices.add(idx)

        return results[:top_k]
    
    def _find_exact_phrase_matches(self, query_text: str, similarities: np.ndarray) -> List[tuple]:
        """
        Find tickets that contain exact phrases from the query
        """
        exact_matches = []
        query_lower = query_text.lower()
        
        # Look for exact phrase matches in original ticket data
        for idx, ticket_text in enumerate(self.ticket_texts):
            if query_lower in ticket_text.lower():
                # Boost the similarity score for exact matches
                boosted_score = min(similarities[idx] + 0.3, 1.0)
                exact_matches.append((idx, boosted_score))
        
        return sorted(exact_matches, key=lambda x: x[1], reverse=True)

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
