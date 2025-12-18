"""
Main DOCX Scraper Component

This module integrates document reading, pattern recognition, and validation
to provide a complete DOCX FAQ extraction system.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from faq.rag.interfaces.base import DOCXScraperInterface, FAQEntry, DocumentStructure, ValidationResult
from .document_reader import DOCXDocumentReader
from .pattern_recognizer import FAQPatternRecognizer, FAQPattern
from .validator import FAQValidator

logger = logging.getLogger(__name__)


class DOCXScraper(DOCXScraperInterface):
    """Complete DOCX scraper implementation with pattern recognition and validation."""
    
    def __init__(self):
        """Initialize the scraper with all components."""
        self.document_reader = DOCXDocumentReader()
        self.pattern_recognizer = FAQPatternRecognizer()
        self.validator = FAQValidator()
    
    def extract_faqs(self, docx_path: str) -> List[FAQEntry]:
        """
        Extract FAQ entries from a DOCX document.
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            List of extracted FAQ entries
        """
        try:
            logger.info(f"Starting FAQ extraction from: {docx_path}")
            
            # Load the document
            document = self.document_reader.load_document(docx_path)
            if not document:
                logger.error(f"Failed to load document: {docx_path}")
                return []
            
            # Analyze document structure
            structure = self.document_reader.analyze_document_structure(document)
            
            # Identify FAQ patterns
            patterns = self.pattern_recognizer.identify_faq_patterns(structure)
            
            if not patterns:
                logger.warning(f"No FAQ patterns found in document: {docx_path}")
                return []
            
            # Convert patterns to FAQ entries
            faqs = self._convert_patterns_to_faqs(patterns, docx_path)
            
            # Validate and clean up
            faqs = self._validate_and_clean_faqs(faqs)
            
            # Categorize FAQs
            categorized = self.validator.categorize_faqs(faqs)
            
            # Update FAQ categories
            for category, faq_list in categorized.items():
                for faq in faq_list:
                    if not faq.category or faq.category == 'general':
                        faq.category = category
            
            logger.info(f"Successfully extracted {len(faqs)} FAQ entries from {docx_path}")
            return faqs
            
        except Exception as e:
            logger.error(f"Error extracting FAQs from {docx_path}: {str(e)}")
            return []
    
    def parse_document_structure(self, document_path: str) -> DocumentStructure:
        """
        Parse and analyze document structure.
        
        Args:
            document_path: Path to the document
            
        Returns:
            DocumentStructure object
        """
        try:
            document = self.document_reader.load_document(document_path)
            if not document:
                return DocumentStructure(
                    document_type="docx",
                    sections=[],
                    tables=[],
                    lists=[],
                    paragraphs=[]
                )
            
            return self.document_reader.analyze_document_structure(document)
            
        except Exception as e:
            logger.error(f"Error parsing document structure: {str(e)}")
            return DocumentStructure(
                document_type="docx",
                sections=[],
                tables=[],
                lists=[],
                paragraphs=[]
            )
    
    def identify_faq_patterns(self, content: List[str]) -> List[Dict[str, Any]]:
        """
        Identify FAQ patterns in document content.
        
        Args:
            content: List of content strings
            
        Returns:
            List of FAQ pattern dictionaries
        """
        try:
            # Create a simple document structure from content
            structure = DocumentStructure(
                document_type="text",
                sections=[],
                tables=[],
                lists=[],
                paragraphs=content
            )
            
            patterns = self.pattern_recognizer.identify_faq_patterns(structure)
            
            # Convert patterns to dictionaries
            pattern_dicts = []
            for pattern in patterns:
                pattern_dicts.append({
                    "pattern_type": pattern.pattern_type,
                    "question": pattern.question,
                    "answer": pattern.answer,
                    "confidence": pattern.confidence,
                    "source_location": pattern.source_location,
                    "keywords": pattern.keywords,
                    "metadata": pattern.metadata
                })
            
            return pattern_dicts
            
        except Exception as e:
            logger.error(f"Error identifying FAQ patterns: {str(e)}")
            return []
    
    def validate_extraction(self, faqs: List[FAQEntry]) -> ValidationResult:
        """
        Validate extracted FAQ entries.
        
        Args:
            faqs: List of FAQ entries to validate
            
        Returns:
            ValidationResult with validation status
        """
        return self.validator.validate_extraction(faqs)
    
    def _convert_patterns_to_faqs(self, patterns: List[FAQPattern], source_document: str) -> List[FAQEntry]:
        """
        Convert FAQ patterns to FAQ entries.
        
        Args:
            patterns: List of FAQ patterns
            source_document: Source document path
            
        Returns:
            List of FAQ entries
        """
        faqs = []
        current_time = datetime.now()
        
        for pattern in patterns:
            try:
                faq_id = str(uuid.uuid4())
                
                faq = FAQEntry(
                    id=faq_id,
                    question=pattern.question.strip(),
                    answer=pattern.answer.strip(),
                    keywords=pattern.keywords,
                    category="general",  # Will be updated by categorization
                    confidence_score=pattern.confidence,
                    source_document=source_document,
                    created_at=current_time,
                    updated_at=current_time,
                    embedding=None  # Will be populated by vectorizer
                )
                
                faqs.append(faq)
                
            except Exception as e:
                logger.error(f"Error converting pattern to FAQ: {str(e)}")
                continue
        
        return faqs
    
    def _validate_and_clean_faqs(self, faqs: List[FAQEntry]) -> List[FAQEntry]:
        """
        Validate and clean FAQ entries, removing invalid ones.
        
        Args:
            faqs: List of FAQ entries to validate
            
        Returns:
            List of valid FAQ entries
        """
        valid_faqs = []
        
        for faq in faqs:
            validation_result = self.validator.validate_faq_entry(faq)
            
            if validation_result.is_valid:
                valid_faqs.append(faq)
            else:
                logger.warning(f"Removing invalid FAQ {faq.id}: {validation_result.errors}")
        
        # Check for duplicates and handle them
        if len(valid_faqs) > 1:
            duplicates = self.validator.detect_duplicates(valid_faqs)
            
            if duplicates:
                logger.info(f"Found {len(duplicates)} potential duplicate pairs")
                
                # For now, just log duplicates. In a production system,
                # you might want to merge or remove duplicates based on business rules
                for dup in duplicates:
                    if dup.match_type == 'exact':
                        logger.warning(f"Exact duplicate found: {dup.faq1_id} <-> {dup.faq2_id}")
                    else:
                        logger.info(f"Similar FAQ found: {dup.faq1_id} <-> {dup.faq2_id} "
                                  f"(similarity: {dup.similarity_score:.2f})")
        
        logger.info(f"Validated {len(valid_faqs)} out of {len(faqs)} FAQ entries")
        return valid_faqs