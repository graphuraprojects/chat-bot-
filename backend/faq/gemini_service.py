"""
Gemini AI Service for Intelligent FAQ Matching

This service integrates Google's Gemini AI to provide advanced natural language
understanding for FAQ matching. It can understand user intent, rephrase queries,
and find semantically similar FAQ entries even when the wording is completely different.

Features:
- Intent understanding and query analysis
- Semantic similarity using Gemini embeddings
- Query rephrasing and expansion
- Intelligent FAQ recommendation
- Context-aware response generation
"""

import os
import json
import logging
import warnings
from typing import List, Dict, Optional, Tuple
from django.conf import settings
from .models import FAQEntry

# Configure logging
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")


class GeminiService:
    """
    Service for integrating Gemini AI with FAQ matching system.
    
    Provides intelligent query understanding, semantic matching,
    and context-aware response generation.
    """
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.model = None
        self.embedding_model = None
        
        if GEMINI_AVAILABLE and self.api_key:
            self._initialize_gemini()
        else:
            logger.warning("Gemini service not initialized - API key missing or library unavailable")
    
    def _get_api_key(self) -> Optional[str]:
        """Get Gemini API key from settings or environment"""
        # Try Django settings first
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        
        # Fall back to environment variable
        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY')
        
        return api_key
    
    def _initialize_gemini(self):
        """Initialize Gemini AI models"""
        try:
            genai.configure(api_key=self.api_key)
            
            # Initialize text generation model
            self.model = genai.GenerativeModel('gemini-pro')
            
            # Initialize embedding model for semantic similarity
            self.embedding_model = 'models/embedding-001'
            
            logger.info("Gemini AI service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {str(e)}")
            self.model = None
            self.embedding_model = None
    
    def is_available(self) -> bool:
        """Check if Gemini service is available and configured"""
        return GEMINI_AVAILABLE and self.model is not None
    
    def understand_query_intent(self, user_query: str) -> Dict:
        """
        Analyze user query to understand intent and extract key information.
        
        Args:
            user_query: The user's question or request
            
        Returns:
            Dictionary containing intent analysis, keywords, and rephrased queries
        """
        if not self.is_available():
            return {"error": "Gemini service not available"}
        
        try:
            prompt = f"""
            Analyze this user query and help match it to FAQ entries:
            
            User Query: "{user_query}"
            
            Please provide a JSON response with:
            1. "intent": What the user is trying to accomplish
            2. "keywords": Key terms that should be searched for
            3. "rephrased_queries": 3-5 alternative ways to phrase this question
            4. "category": What category this question belongs to (e.g., "account", "technical", "billing", "support")
            5. "urgency": How urgent this seems (low/medium/high)
            
            Focus on understanding the core need, not just the exact words used.
            
            Example:
            {{
                "intent": "User wants to know how to apply for an internship",
                "keywords": ["internship", "apply", "application", "job", "career"],
                "rephrased_queries": [
                    "How do I apply for internship?",
                    "What is the internship application process?",
                    "How to get an internship position?",
                    "Internship application requirements",
                    "Steps to apply for internship program"
                ],
                "category": "career",
                "urgency": "medium"
            }}
            
            Respond only with valid JSON.
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, extract what we can
                return {
                    "intent": "Query analysis",
                    "keywords": self._extract_keywords_fallback(user_query),
                    "rephrased_queries": [user_query],
                    "category": "general",
                    "urgency": "medium",
                    "raw_response": response.text
                }
                
        except Exception as e:
            logger.error(f"Error analyzing query intent: {str(e)}")
            return {
                "error": f"Intent analysis failed: {str(e)}",
                "fallback_keywords": self._extract_keywords_fallback(user_query)
            }
    
    def find_semantic_matches(self, user_query: str, faq_entries: List[FAQEntry], max_results: int = 5) -> List[Dict]:
        """
        Find FAQ entries that are semantically similar to the user query using Gemini AI.
        
        Args:
            user_query: The user's question
            faq_entries: List of FAQ entries to search through
            max_results: Maximum number of results to return
            
        Returns:
            List of FAQ matches with semantic similarity scores
        """
        if not self.is_available():
            return []
        
        try:
            # First, understand the query intent
            intent_analysis = self.understand_query_intent(user_query)
            
            # Create a comprehensive search using Gemini's understanding
            search_queries = [user_query]
            
            if "rephrased_queries" in intent_analysis:
                search_queries.extend(intent_analysis["rephrased_queries"])
            
            # Use Gemini to evaluate each FAQ for relevance
            matches = []
            
            for faq in faq_entries:
                relevance_score = self._calculate_semantic_relevance(
                    search_queries, faq, intent_analysis
                )
                
                if relevance_score > 0.3:  # Minimum relevance threshold
                    matches.append({
                        'faq': faq,
                        'id': faq.id,
                        'question': faq.question,
                        'answer': faq.answer,
                        'keywords': faq.keywords,
                        'semantic_score': relevance_score,
                        'intent_match': intent_analysis.get('intent', ''),
                        'matched_keywords': intent_analysis.get('keywords', [])
                    })
            
            # Sort by semantic score
            matches.sort(key=lambda x: x['semantic_score'], reverse=True)
            
            return matches[:max_results]
            
        except Exception as e:
            logger.error(f"Error in semantic matching: {str(e)}")
            return []
    
    def _calculate_semantic_relevance(self, search_queries: List[str], faq: FAQEntry, intent_analysis: Dict) -> float:
        """
        Calculate semantic relevance between search queries and FAQ entry using Gemini.
        
        Args:
            search_queries: List of query variations
            faq: FAQ entry to evaluate
            intent_analysis: Intent analysis results
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            # Create a prompt for Gemini to evaluate relevance
            prompt = f"""
            Evaluate how well this FAQ entry matches the user's intent and queries.
            
            User Intent: {intent_analysis.get('intent', 'Unknown')}
            User Keywords: {', '.join(intent_analysis.get('keywords', []))}
            
            Search Queries:
            {chr(10).join(f'- {query}' for query in search_queries)}
            
            FAQ Entry:
            Question: {faq.question}
            Answer: {faq.answer[:200]}...
            Keywords: {faq.keywords}
            
            Rate the relevance on a scale of 0.0 to 1.0 where:
            - 1.0 = Perfect match, exactly what the user is looking for
            - 0.8 = Very relevant, addresses the user's need well
            - 0.6 = Somewhat relevant, partially addresses the need
            - 0.4 = Loosely related, might be helpful
            - 0.2 = Barely related, unlikely to be helpful
            - 0.0 = Not related at all
            
            Consider:
            1. Does the FAQ answer the user's specific question?
            2. Are the topics and keywords related?
            3. Would this FAQ be helpful to someone with this intent?
            
            Respond with only a number between 0.0 and 1.0.
            """
            
            response = self.model.generate_content(prompt)
            
            # Extract numeric score
            try:
                score_text = response.text.strip()
                score = float(score_text)
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                # If we can't parse the score, use a fallback method
                return self._fallback_relevance_score(search_queries, faq)
                
        except Exception as e:
            logger.error(f"Error calculating semantic relevance: {str(e)}")
            return self._fallback_relevance_score(search_queries, faq)
    
    def _fallback_relevance_score(self, search_queries: List[str], faq: FAQEntry) -> float:
        """Fallback relevance calculation using simple text matching"""
        # Simple keyword overlap as fallback
        query_words = set()
        for query in search_queries:
            query_words.update(query.lower().split())
        
        faq_words = set()
        faq_words.update(faq.question.lower().split())
        faq_words.update(faq.answer.lower().split())
        if faq.keywords:
            faq_words.update(faq.keywords.lower().split())
        
        if not query_words or not faq_words:
            return 0.0
        
        overlap = len(query_words.intersection(faq_words))
        return min(1.0, overlap / len(query_words))
    
    def _extract_keywords_fallback(self, text: str) -> List[str]:
        """Extract keywords as fallback when Gemini is not available"""
        import re
        
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:10]  # Return top 10 keywords
    
    def generate_contextual_response(self, user_query: str, matched_faq: Dict) -> str:
        """
        Generate a contextual response that addresses the user's specific query
        using the matched FAQ as a base.
        
        Args:
            user_query: Original user question
            matched_faq: The best matching FAQ entry
            
        Returns:
            Contextual response tailored to the user's query
        """
        if not self.is_available():
            # Return the FAQ answer as-is if Gemini is not available
            return matched_faq.get('answer', 'I found a relevant FAQ but cannot generate a contextual response.')
        
        try:
            prompt = f"""
            A user asked: "{user_query}"
            
            I found this relevant FAQ entry:
            Question: {matched_faq['question']}
            Answer: {matched_faq['answer']}
            
            Please generate a helpful response that:
            1. Directly addresses the user's specific question
            2. Uses the FAQ information as the source of truth
            3. Adapts the language to match the user's query style
            4. Provides a complete, helpful answer
            5. Maintains a friendly, professional tone
            
            If the FAQ doesn't fully answer the user's question, acknowledge what it does cover
            and suggest they might need additional help.
            
            Keep the response concise but complete (2-4 sentences).
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating contextual response: {str(e)}")
            # Fallback to original FAQ answer
            return matched_faq.get('answer', 'I found relevant information but cannot generate a contextual response.')
    

    def correct_query_spelling(self, user_query: str) -> str:
        """
        Uses Gemini to correct spelling and grammar of the query.
        Does NOT answer the question. Only normalizes it.
        """
        if not self.is_available():
            return user_query
        
        try:
            prompt = f"""
            Task: Correct the spelling and grammar of the user's query.
            Do NOT answer the question.
            Do NOT add extra punctuation or words if not needed.
            Output ONLY the corrected text.
            
            Input: "{user_query}"
            Corrected:
            """
            
            response = self.model.generate_content(prompt)
            corrected = response.text.strip()
            # Safety check: if response is too long or looks like an answer, return original
            if len(corrected) > len(user_query) * 2:
                return user_query
                
            return corrected
            
        except Exception as e:
            logger.error(f"Error correcting query: {str(e)}")
            return user_query

    def generate_fallback_response(self, user_query: str, user_name: str = None) -> str:
        """
        Generate a polite fallback response when no FAQ matches.
        """
        if not self.is_available():
            return "I'm sorry, I couldn't find a specific answer to your question in our database. Please contact support at support@graphura.in."
            
        try:
            addressing = f"Address the user as '{user_name}'." if user_name else ""
            
            prompt = f"""
            You are a helpful customer support assistant for Graphura.
            The user asked: "{user_query}"
            
            We do NOT have a specific FAQ entry for this in our database.
            Please apologize politely and suggest they contact support at [hr@graphura.in](mailto:hr@graphura.in) or rephrase the question.
            {addressing}
            Do NOT make up an answer.
            Do NOT hallucinate company policies.
            
            Response:
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating fallback: {str(e)}")
            return "I’m sorry, I couldn’t find a relevant answer to your question. Try rephrasing your query or contact support team at [hr@graphura.in](mailto:hr@graphura.in) for further assistance."
            
    # Deprecated / Modified to fit new logic if called
    def enhance_faq_search(self, user_query: str, max_results: int = 5, min_confidence: float = 0.3) -> List[Dict]:
        """
        Modified to follow Strict Rules: 
        1. Correct Query.
        2. NO semantic search invocation here (logic moved to views/matcher).
        This exists for backward compatibility but behaves deterministically.
        """
        warnings.warn("enhance_faq_search is deprecated. Use direct correction and matcher logic.", DeprecationWarning)
        
        # Just return empty list or basic correction to force the new flow in views
        # Or better: implementation of the NEW Logic if this IS the main entry point
        # But per plan, we are doing logic in views.
        return []

# Global instance
gemini_service = GeminiService()
