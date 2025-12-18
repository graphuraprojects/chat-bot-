"""
Basic Response Generator Implementation

This module provides template-based response generation with FAQ content integration,
formatting, and confidence scoring for the RAG system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import json

from faq.rag.interfaces.base import (
    ResponseGeneratorInterface,
    FAQEntry,
    Response,
    ConversationContext
)
from faq.rag.utils.logging import get_rag_logger
from faq.rag.utils.debug_logger import get_debug_logger


logger = get_rag_logger(__name__)


class ResponseGeneratorError(Exception):
    """Custom exception for response generator errors."""
    pass


class BasicResponseGenerator(ResponseGeneratorInterface):
    """
    Basic response generator that uses template-based generation with FAQ content
    integration and confidence scoring.

    This implementation focuses on:
    - Template-based response formatting
    - FAQ content integration and synthesis
    - Confidence scoring based on match quality
    - Source attribution and metadata
    """

    def __init__(self):
        """Initialize the basic response generator."""
        self.response_templates = self._load_response_templates()
        self.confidence_weights = self._load_confidence_weights()
        self.formatting_rules = self._load_formatting_rules()
        self.debug_logger = get_debug_logger('response_generator')

        logger.info("Basic response generator initialized")

    def _load_response_templates(self) -> Dict[str, Dict[str, str]]:
        """Load response templates for different scenarios."""
        return {
            'single_match': {
                'high_confidence': "Based on the information I found, {answer}",
                'medium_confidence': "According to the available information, {answer}",
                'low_confidence': "I found some information that might help: {answer}"
            },
            'multiple_matches': {
                'high_confidence': "I found several relevant pieces of information:\n\n{synthesized_answer}",
                'medium_confidence': "Here's what I found from multiple sources:\n\n{synthesized_answer}",
                'low_confidence': "I found some related information that might be helpful:\n\n{synthesized_answer}"
            },
            'no_match': {
                'fallback': "I couldn't find specific information about your question. You might want to try rephrasing your question or contact support for more help."
            },
            'partial_match': {
                'suggestion': "I found some related information: {answer}\n\nIf this doesn't fully answer your question, you might also want to look into {suggestions}."
            }
        }

    def _load_confidence_weights(self) -> Dict[str, float]:
        """Load weights for confidence calculation."""
        return {
            'similarity_score': 0.4,      # Weight for semantic similarity
            'match_completeness': 0.2,    # How complete the match is
            'source_quality': 0.2,        # Quality of source FAQ
            'content_relevance': 0.2      # Relevance of content to query
        }

    def _load_formatting_rules(self) -> Dict[str, Any]:
        """Load formatting rules for response presentation."""
        return {
            'max_answer_length': 500,     # Maximum length for single answer
            'max_total_length': 1000,     # Maximum total response length
            'bullet_threshold': 3,        # Use bullets when >= 3 items
            'source_attribution': True,   # Include source information
            'confidence_display': True,   # Show confidence indicators
            'preserve_formatting': True   # Preserve original FAQ formatting
        }

    def generate_response(self, query: str, retrieved_faqs: List[FAQEntry], query_id: Optional[str] = None, processed_query: Optional[ProcessedQuery] = None) -> Response:
        """
        Generate contextual response from retrieved FAQs.

        Args:
            query: Original user query
            retrieved_faqs: List of relevant FAQ entries

        Returns:
            Response object with generated text and metadata

        Raises:
            ResponseGeneratorError: If response generation fails
        """
        try:
            logger.debug(f"Generating response for query: '{query}' with {len(retrieved_faqs)} FAQs")

            if not retrieved_faqs:
                return self._generate_no_match_response(query, query_id, processed_query)

            # Determine response strategy based on number of matches
            if len(retrieved_faqs) == 1:
                return self._generate_single_match_response(query, retrieved_faqs[0], query_id, processed_query)
            else:
                return self._generate_multiple_match_response(query, retrieved_faqs)

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise ResponseGeneratorError(f"Response generation failed: {e}")

    def _generate_single_match_response(self, query: str, faq: FAQEntry, query_id: Optional[str] = None, processed_query: Optional[ProcessedQuery] = None) -> Response:
        """Generate response for a single FAQ match."""
        # Determine confidence level
        confidence_level = self._determine_confidence_level(faq.confidence_score)

        # Get appropriate template
        template = self.response_templates['single_match'][confidence_level]

        # Format the answer - EXTRACT ONLY THE ANSWER PORTION
        formatted_answer = self._format_single_answer(faq)

        # Generate response text
        response_text = template.format(answer=formatted_answer)

        # Add source attribution if enabled
        if self.formatting_rules['source_attribution']:
            response_text += f"\n\n*Source: {faq.source_document}*"

        # Create response object
        response = Response(
            text=response_text,
            confidence=faq.confidence_score,
            source_faqs=[faq],
            context_used=False,
            processing_time=0.0,
            generation_method='direct_match',
            query_id=query_id, # Use the passed query_id
            processed_query=processed_query, # Pass processed_query
            metadata={
                'template_used': f"single_match.{confidence_level}",
                'faq_category': faq.category,
                'source_document': faq.source_document
            }
        )

        return response

    def _generate_multiple_match_response(self, query: str, faqs: List[FAQEntry]) -> Response:
        """Generate response for multiple FAQ matches - SELECT BEST MATCH ONLY."""

        # CRITICAL FIX: Select only the best match instead of concatenating all
        best_faq = max(faqs, key=lambda faq: faq.confidence_score)

        logger.debug(f"Selected best match from {len(faqs)} FAQs: {best_faq.id} (confidence: {best_faq.confidence_score:.3f})")

        # Generate response using only the best match
        return self._generate_single_match_response(query, best_faq)

    def _generate_no_match_response(self, query: str, query_id: Optional[str] = None, processed_query: Optional[ProcessedQuery] = None) -> Response:
        """Generate response when no FAQs match the query."""
        fallback_text = self.response_templates['no_match']['fallback']

        response = Response(
            text=fallback_text,
            confidence=0.1,
            source_faqs=[],
            context_used=False,
            generation_method='fallback',
            query_id=query_id, # Use the passed query_id
            processed_query=processed_query, # Pass processed_query
            processing_time=0.0,
            metadata={
                'template_used': 'no_match.fallback',
                'original_query': query
            }
        )

        return response

    def _format_single_answer(self, faq: FAQEntry) -> str:
        """Format a single FAQ answer with proper formatting - ANSWER ONLY."""
        # CRITICAL: Extract only the answer portion, not the question
        answer = faq.answer.strip()

        # Remove any Q: A: patterns if they exist
        if answer.startswith('Q:'):
            # Find the answer portion after A:
            parts = answer.split('A:', 1)
            if len(parts) > 1:
                answer = parts[1].strip()

        # Respect maximum length
        max_length = self.formatting_rules['max_answer_length']
        if len(answer) > max_length:
            # Try to cut at sentence boundary
            sentences = answer.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= max_length:
                    truncated += sentence + '. '
                else:
                    break

            if truncated:
                answer = truncated.rstrip('. ') + '.'
            else:
                # Hard truncate if no sentence boundary found
                answer = answer[:max_length-3] + '...'

        return answer

    def _determine_confidence_level(self, confidence_score: float) -> str:
        """Determine confidence level category."""
        if confidence_score >= 0.8:
            return 'high_confidence'
        elif confidence_score >= 0.5:
            return 'medium_confidence'
        else:
            return 'low_confidence'

    def calculate_confidence(self, response: Response) -> float:
        """Calculate confidence score for generated response."""
        try:
            if not response.source_faqs:
                return 0.1  # Very low confidence for no sources

            # Use the confidence of the source FAQ(s)
            faq_confidences = [faq.confidence_score for faq in response.source_faqs]
            avg_confidence = sum(faq_confidences) / len(faq_confidences)

            # Ensure confidence is within bounds
            final_confidence = max(0.0, min(1.0, avg_confidence))

            logger.debug(f"Calculated confidence: {final_confidence:.3f}")
            return final_confidence

        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5  # Default confidence

    def get_generator_stats(self) -> Dict[str, Any]:
        """Get statistics about the response generator."""
        return {
            'generator_type': 'basic_template',
            'templates_loaded': len(self.response_templates),
            'confidence_weights': self.confidence_weights,
            'formatting_rules': self.formatting_rules,
            'max_answer_length': self.formatting_rules['max_answer_length'],
            'max_total_length': self.formatting_rules['max_total_length']
        }

    def synthesize_multiple_sources(self, faqs: List[FAQEntry]) -> str:
        """
        Synthesize information from multiple FAQ sources.

        Args:
            faqs: List of FAQ entries to synthesize

        Returns:
            Synthesized text combining information from all sources
        """
        if not faqs:
            return ""

        if len(faqs) == 1:
            return self._format_single_answer(faqs[0])

        # For the basic implementation, just return the best answer
        # This prevents content dumping by selecting only the highest confidence FAQ
        best_faq = max(faqs, key=lambda faq: faq.confidence_score)
        return self._format_single_answer(best_faq)

    def maintain_context(self, conversation_history: List[Dict[str, Any]]) -> ConversationContext:
        """
        Maintain conversation context across interactions.

        Args:
            conversation_history: List of previous interactions

        Returns:
            ConversationContext object
        """
        # Basic implementation - just store history
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        context = ConversationContext(
            session_id=session_id,
            history=conversation_history,
            current_topic=None,
            user_preferences={},
            last_activity=datetime.now(),
            context_embeddings=[]
        )

        logger.debug(f"Maintained context for session: {session_id}")
        return context