import uuid
import datetime
import logging
import re # For simple keyword extraction
import json
import os
from typing import Dict, Any, Optional, List

# For more advanced NLP tasks like entity extraction, you might use spaCy or NLTK
# import spacy
# nlp = spacy.load("en_core_web_sm") # Load a small English model (needs to be downloaded)

class MetadataGenerator:
    """
    Generates metadata for each summary.
    Includes unique IDs, timestamps, key terms (simple extraction), entities (placeholder for now),
    and other relevant information based on the summarization context.
    """
    def __init__(self):
        # Placeholder for more sophisticated NLP tools if needed
        # self.nlp_model = spacy.load("en_core_web_sm") if spacy_available else None
        logging.info("MetadataGenerator initialized.")

    def _extract_key_terms(self, text, num_terms=10):
        """Simple keyword extraction: most frequent non-stop words (customize as needed)."""
        if not text:
            return []
        try:
            # Simple stop words list (extend this for better quality)
            stop_words = set([
                "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would", "should", "can",
                "could", "may", "might", "must", "and", "but", "or", "nor", "for", "so", "yet",
                "in", "on", "at", "by", "from", "to", "with", "about", "above", "below",
                "of", "s", "t", "not", "this", "that", "these", "those", "i", "you", "he", "she",
                "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
                "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "also",
                "as", "if", "then", "than", "such", "other", "which", "what", "when", "where",
                "who", "whom", "why", "how", "all", "any", "both", "each", "few", "more",
                "most", "no", "some", "many"
            ])
            words = re.findall(r'\b\w+\b', text.lower()) # Find all words
            # Filter out stop words and very short words (e.g., less than 3 chars)
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            if not filtered_words:
                return []

            from collections import Counter
            word_counts = Counter(filtered_words)
            key_terms = [term for term, count in word_counts.most_common(num_terms)]
            return key_terms
        except Exception as e:
            logging.warning(f"Simple key term extraction failed: {e}")
            return []

    def _extract_entities(self, text):
        """Placeholder for entity extraction. Requires an NLP library like spaCy or NLTK."""
        # Example with spaCy (if you were to integrate it):
        # if self.nlp_model:
        #     doc = self.nlp_model(text)
        #     entities = list(set([ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE"]]))
        #     return entities
        return ["entity_extraction_placeholder"] # Placeholder

    def generate_metadata(self, summary_text, source_path, mode,
                          doc_type=None, # e.g., 40-F, 10-K
                          section_id=None, section_title=None, 
                          parent_id=None, relationship_ids=None,
                          theme_type=None, # For cross-sectional
                          **kwargs): # For any other context-specific info
        """
        Generates a dictionary of metadata for the summary.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary_id = str(uuid.uuid4())
        word_count = len(summary_text.split())
        char_count = len(summary_text)

        # Correct file_id to be workspace-relative
        # Assumes this script (metadata.py) is in model_doc_agent/src/
        # WORKSPACE_ROOT is then two levels above the directory of this file.
        try:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
            absolute_source_path = os.path.abspath(os.path.join(workspace_root, "model_doc_agent", source_path)) # model_doc_agent is the cwd for source_path
            # If source_path is already absolute or correctly relative from workspace_root, this might simplify
            # Given source_path like '../TestData/...' from model_doc_agent cwd:
            model_doc_agent_dir = os.path.join(workspace_root, "model_doc_agent")
            absolute_source_path = os.path.abspath(os.path.join(model_doc_agent_dir, source_path))
            
            corrected_file_id = os.path.relpath(absolute_source_path, workspace_root)
            # Normalize to use forward slashes for consistency
            corrected_file_id = corrected_file_id.replace(os.sep, '/')

        except Exception as e:
            logging.warning(f"Could not correct source_path to workspace-relative: {e}. Using original: {source_path}")
            corrected_file_id = source_path

        # Determine category based on doc_type
        category = "General" # Default category
        if doc_type:
            if doc_type.lower() == "news":
                category = "News"
            elif doc_type.lower() not in ["default", "unknown", "unknown_sec_filing"]:
                # Assume it's an SEC filing type if not explicitly news or a known non-SEC default
                category = "SEC Filings"
        
        meta = {
            "summary_id": summary_id,
            "parent_id": parent_id, # ID of a parent summary (e.g., for node under a file)
            "file_id": corrected_file_id, # Original source document path/identifier
            "category": category, 
            "domain": kwargs.get("domain", "Financial Information"), # Changed default domain
            "doc_type": doc_type, # e.g. 40-F, 10-K, extracted from input or filename
            "summary_type": mode, # file-level, node-level, master-level, cross-sectional
            "relationship_ids": relationship_ids if relationship_ids else [], # Related summaries for aggregation
            "generated_timestamp": timestamp,
            "word_count": word_count,
            "character_count": char_count, # Additional potentially useful stat
            "key_terms": self._extract_key_terms(summary_text), # Basic keyword extraction
            "entities_mentioned": self._extract_entities(summary_text), # Placeholder
            "llm_model_used": kwargs.get("llm_model_used", "unknown"), # Should be passed from summarizer
            "prompt_template_id": kwargs.get("prompt_template_id", "unknown") # Identifier for the prompt used
        }

        if section_id:
            meta["section_id"] = section_id
        if section_title:
            meta["section_title"] = section_title
        if theme_type:
            meta["theme_type"] = theme_type
        
        # Add any other kwargs that might have been passed
        meta.update(kwargs.get("additional_fields", {}))

        logging.debug(f"Generated metadata for summary_id: {summary_id}")
        return meta

def extract_metadata(
    data: Dict[str, Any], 
    original_filename: str, 
    file_path: str, 
    summary_mode: str, 
    effective_filing_type: Optional[str] = None, 
    document_title: Optional[str] = None,
    # num_chunks: Optional[int] = None, # Consider if this is needed here or added later
    additional_context: Optional[Dict[str, Any]] = None # For future expansion, e.g. themes
) -> Dict[str, Any]:
    """
    Extracts and standardizes metadata from the input data object and processing parameters.

    Args:
        data (Dict[str, Any]): The raw data from which to extract metadata (e.g., parsed JSON content).
                               For news, this is the direct JSON content of the news article.
        original_filename (str): The original name of the file being processed.
        file_path (str): The full path to the file being processed.
        summary_mode (str): The mode of summarization (e.g., 'file', 'node', 'master', 'cross').
        effective_filing_type (Optional[str]): The determined type of the document (e.g., '10-K', 'news').
        document_title (Optional[str]): An explicit title for the document, if available (e.g. from news JSON).
        additional_context (Optional[Dict[str, Any]]): Optional dictionary for themes or other context.

    Returns:
        Dict[str, Any]: A dictionary containing standardized metadata.
    """
    metadata_dict = {}
    # Basic file info
    metadata_dict["original_filename"] = original_filename
    # metadata_dict["file_size_bytes"] = os.path.getsize(file_path) if os.path.exists(file_path) else None # This might be heavy to do for every call
    metadata_dict["processed_timestamp"] = datetime.datetime.now().isoformat() # Changed to datetime.now()

    # Information about the summarization task itself
    metadata_dict["summary_mode"] = summary_mode
    if additional_context:
        metadata_dict.update(additional_context) # Merge themes or other context

    try:
        # Attempt to get document-specific fields if data is a dictionary
        if isinstance(data, dict):
            metadata_dict.update(data) # Copy all original data into the metadata_dict first

            # Generic fields that might exist in SEC or other structured docs (can overwrite if needed)
            metadata_dict["document_header"] = data.get("header")
            metadata_dict["document_id"] = data.get("id") # e.g. news id, or some internal ID

            # If an effective_filing_type is provided by orchestrator, use it
            if effective_filing_type:
                metadata_dict["effective_filing_type"] = effective_filing_type
            else: # Basic fallback if not provided (less likely with new orchestrator)
                metadata_dict["effective_filing_type"] = data.get("form_type", data.get("filing_type", "unknown"))

            # For news articles, extract specific fields if available
            if metadata_dict.get("effective_filing_type") == "news":
                # news_title already handled by document_title argument, which orchestrator should pass
                # metadata_dict["news_title"] = document_title if document_title else data.get("title", original_filename)
                metadata_dict["source_type"] = data.get("source_type")
                metadata_dict["source_name"] = data.get("source_name")
                metadata_dict["date"] = data.get("date") 
                metadata_dict["url"] = data.get("url")
                # The 'content' is handled by the chunker/summarizer, not stored as direct metadata here
                # Retain original news ID if present
                if data.get("id"):
                     metadata_dict["original_news_id"] = data.get("id")

        # TODO: Add more sophisticated extraction logic for other filing types if needed
        # For example, parsing dates, extracting specific company names, CIKs, etc.

    except Exception as e:
        logger.error(f"Error extracting metadata from data object for {original_filename}: {e}", exc_info=True)
        metadata_dict["metadata_extraction_error"] = str(e)

    # Construct the final metadata structure, which might be slightly different from internal metadata_dict
    # This is the structure that gets saved or passed on, embedding the extracted fields.
    final_metadata = {
        "original_filename": original_filename,
        "processed_timestamp": metadata_dict["processed_timestamp"], # Reuse timestamp from above
        "file_path": file_path, 
        "summary_type": summary_mode, 
        # "num_chunks": num_chunks, # This might be added later by orchestrator if it has this info
        "document_metadata": metadata_dict # Embeds all the extracted fields
    }

    # Determine document_title with clear fallback order
    title_to_use = original_filename # Default to original_filename as the last resort
    if metadata_dict.get("document_header"): # Check if 'document_header' exists and is not None/empty
        title_to_use = metadata_dict["document_header"]
    if metadata_dict.get("news_title"): # Check if 'news_title' exists (more specific for news)
        title_to_use = metadata_dict["news_title"]
    if document_title: # Explicit document_title argument from orchestrator takes highest precedence
        title_to_use = document_title
    final_metadata["document_title"] = title_to_use
    
    # Clean up redundant title in document_metadata if it's same as final_metadata["document_title"]
    if "news_title" in metadata_dict and metadata_dict["news_title"] == final_metadata["document_title"]:
        pass # Keep it for now, as it's under news specific section
    if "document_header" in metadata_dict and metadata_dict["document_header"] == final_metadata["document_title"]:
        del metadata_dict["document_header"] # Avoid redundancy if used as main title


    return final_metadata

# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    md_generator = MetadataGenerator()
    
    sample_summary = "This is a test summary about financial performance of ACME Corp in Q4 2023. Key factors include revenue growth and market expansion. ACME Corp plans new investments."
    
    metadata_file = md_generator.generate_metadata(
        summary_text=sample_summary,
        source_path="/path/to/sec_filing_001.json",
        mode="file",
        doc_type="10-K",
        llm_model_used="gpt-3.5-turbo-0125",
        prompt_template_id="file_summary_v1.txt",
        additional_fields={"CIK": "000123456"}
    )
    print("--- File Mode Metadata ---")
    print(json.dumps(metadata_file, indent=2))

    metadata_node = md_generator.generate_metadata(
        summary_text="This section discusses liquidity risk.",
        source_path="/path/to/sec_filing_001.json",
        mode="node",
        doc_type="10-K",
        section_id="item7a",
        section_title="Quantitative and Qualitative Disclosures About Market Risk",
        parent_id=metadata_file["summary_id"] # Link to the file-level summary
    )
    print("\n--- Node Mode Metadata ---")
    print(json.dumps(metadata_node, indent=2))

    # Add logic for news articles
    effective_filing_type = "news"
    data = {
        "title": "News Title",
        "source_type": "News Source Type",
        "source_name": "News Source Name",
        "date": "2023-04-01",
        "url": "https://example.com",
        "id": "news_id_123"
    }
    original_filename = "news_article.pdf"
    file_path = "/path/to/news_article.pdf"
    document_title = "News Article Title"

    metadata_dict = {
        "original_filename": original_filename,
        "processed_timestamp": datetime.datetime.now().isoformat(),
        "file_path": file_path,
        "summary_type": "file",
        "document_metadata": {
            "summary_id": "news_id_123",
            "parent_id": None,
            "file_id": file_path,
            "category": "SEC_Filings",
            "domain": "Financial_Risk",
            "doc_type": "news",
            "summary_type": "file",
            "relationship_ids": [],
            "generated_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "word_count": 0,
            "character_count": 0,
            "key_terms": [],
            "entities_mentioned": ["News Source Name"],
            "llm_model_used": "unknown",
            "prompt_template_id": "unknown",
            "news_title": "News Title",
            "news_source_type": "News Source Type",
            "news_source_name": "News Source Name",
            "news_date": "2023-04-01",
            "news_url": "https://example.com",
            "news_id": "news_id_123"
        }
    }
    if document_title:
        metadata_dict["document_title"] = document_title

    final_metadata = {
        "original_filename": original_filename,
        "processed_timestamp": datetime.datetime.now().isoformat(),
        "file_path": file_path,
        "summary_type": "file",
        "document_metadata": metadata_dict
    }
    if document_title:
        final_metadata["document_title"] = document_title

    print("\n--- News Article Metadata ---")
    print(json.dumps(final_metadata, indent=2)) 