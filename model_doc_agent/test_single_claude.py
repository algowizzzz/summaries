import json
import logging
import os
import re # For stripping HTML
import quopri # For decoding quoted-printable
from dotenv import load_dotenv

# Assuming this script is in model_doc_agent, adjust paths as needed
from src.summarization import LLMSummarizer
from src.chunker import get_sections_from_json, strip_html # strip_html is now in chunker

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def strip_html(html_string):
    """Simple HTML stripping using regex, also handles quoted-printable decoding."""
    if not html_string:
        return ""
    
    # Decode quoted-printable first
    try:
        # quopri.decodestring expects bytes
        decoded_bytes = quopri.decodestring(html_string.encode('utf-8', 'ignore'))
        text_content = decoded_bytes.decode('utf-8', 'ignore')
    except Exception as e:
        logging.warning(f"Quoted-printable decoding failed: {e}. Proceeding with original string for HTML stripping.")
        text_content = html_string # Fallback to original if decoding fails

    # Remove script and style elements
    clean_text = re.sub(r'<script[^>]*?>.*?</script>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'<style[^>]*?>.*?</style>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    # Remove all other HTML tags
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    # Replace common HTML entities
    clean_text = clean_text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    # Remove excessive newlines and whitespace, and join lines properly
    lines = [line.strip() for line in clean_text.splitlines()]
    # Filter out empty lines that might result from stripping, then join with a single space
    # to avoid losing sentence continuity if newlines were meaningful for sentence breaks in plain text context.
    # However, for LLM, distinct separate lines that were originally paragraphs might be better kept as separate lines.
    # Let's try joining with newline first, then a more aggressive space join if needed.
    clean_text = '\n'.join([line for line in lines if line]) # Keep non-empty lines
    
    # Remove non-printable ASCII characters (allow space to tilde, and newline)
    clean_text = re.sub(r'[^ -~\n]+', '', clean_text)
    # Consolidate multiple whitespace characters (including newlines if they become multiple) into a single space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def main():
    # Path to the single test file
    # test_file_path = "../TestData/news/reg/cfpb/2025-01/article_cfpb_1.json"
    test_file_path = "../TestData/Banks/JPMorgan_Chase/jpm_document_1.json" # Changed to JPM 8-K
    
    prompt_set_config_path = "sec_prompts_v1.json"

    from dotenv import load_dotenv
    load_dotenv()

    logging.info(f"Attempting to summarize file: {test_file_path}")

    try:
        summarizer = LLMSummarizer(prompt_set_path=prompt_set_config_path)
        logging.info(f"LLMSummarizer initialized with model: {summarizer.model_name}")
    except Exception as e:
        logging.error(f"Failed to initialize LLMSummarizer: {e}", exc_info=True)
        return

    try:
        with open(test_file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded data from {test_file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {test_file_path}: {e}", exc_info=True)
        return

    try:
        # For an 8-K, get_sections_from_json might stringify it or find specific items.
        # The content might be the full JSON string if no other structure is found.
        sections = get_sections_from_json(data)
        if not sections:
            logging.warning("No sections extracted from the document. Trying to use entire JSON as content.")
            document_content_str = json.dumps(data, indent=2)
            document_title = os.path.basename(test_file_path) 
            # We need to ensure this content is cleaned if it was just a JSON dump
            # strip_html will handle basic cleaning even if not strictly HTML
            cleaned_content_str = strip_html(document_content_str) 
            if not cleaned_content_str.strip():
                logging.error("Document content is empty after stringifying and cleaning. Cannot summarize.")
                return
            sections = [(document_title, cleaned_content_str)]
        logging.info(f"Extracted {len(sections)} section(s).")
    except Exception as e:
        logging.error(f"Failed to get sections from data: {e}", exc_info=True)
        return
        
    if not sections:
        logging.error("Still no sections available after fallback. Exiting.")
        return

    first_section_title, first_section_content = sections[0]
    
    # Determine effective_filing_type
    filename_lower = os.path.basename(test_file_path).lower()
    if "8-k" in filename_lower or (isinstance(data,dict) and data.get('type','').upper() == '8-K'):
        effective_filing_type = "8-K"
    elif "10-k" in filename_lower  or (isinstance(data,dict) and data.get('type','').upper() == '10-K'):
        effective_filing_type = "10-K"
    elif "13f-hr" in filename_lower or "13F-HR" in os.path.basename(test_file_path)  or (isinstance(data,dict) and data.get('type','').upper() == '13F-HR'):
         effective_filing_type = "13F-HR" # For our boa_document_1.json if we switch back
    # Add more specific types or use a generic SEC type if no match
    elif "news" in test_file_path or ".html.json" in filename_lower or "article" in filename_lower:
        effective_filing_type = "news" # Should not be hit for JPM 8-K
    else:
        effective_filing_type = data.get('form_type', data.get('type', "SEC_FILING")).upper() # Generic SEC filing type from data or default
        if not effective_filing_type:
             effective_filing_type = "UNKNOWN_SEC_FILING" # Ultimate fallback

    # Prepare args for the generic SEC file summary template
    # Variables: {effective_filing_type}, {source_filename}, {document_title}, {current_date}, {content}
    summarizer_args = {
        "effective_filing_type": effective_filing_type,
        "source_filename": os.path.basename(test_file_path),
        "document_title": first_section_title if first_section_title else os.path.basename(test_file_path),
        "current_date": "N/A for this test", # Placeholder, orchestrator would provide actual date
        "content": first_section_content 
    }
    
    # The strip_html is now expected to be handled by get_sections_from_json if content is from news/html.
    # For SEC filings that are stringified JSONs, the content might be raw JSON string.
    # The LLM will have to parse this. The generic prompt asks it to summarize the "content".

    content_to_log = summarizer_args.get("content", "")
    if not content_to_log.strip():
        logging.error("Final content to be summarized is empty or whitespace. Aborting LLM call.")
        return

    logging.info(f"Using summarizer.summarize() with mode='file', filing_type='{effective_filing_type}'")
    try:
        summary = summarizer.summarize(
            mode="file", 
            filing_type=effective_filing_type, # This ensures the correct top-level key in sec_prompts_v1.json is used
            **summarizer_args
        )
        
        logging.info("\n--- Generated Summary (summarizer.summarize) ---")
        print(summary)
        logging.info("-------------------------------------------------")

    except Exception as e:
        logging.error(f"Error during summarizer.summarize(): {e}", exc_info=True)

if __name__ == "__main__":
    main() 