import json
from typing import List, Dict, Any, Tuple, Optional, Union
import re
import logging
import tiktoken # For token counting, if specific token-based chunking is desired
import quopri # Added for HTML/Quoted-Printable decoding

# Configure logging
logger = logging.getLogger(__name__)

# Helper function for HTML and Quoted-Printable cleaning (moved from test_single_claude.py)
def strip_html(html_string):
    """Simple HTML stripping using regex, also handles quoted-printable decoding."""
    if not isinstance(html_string, str):
        return html_string # Return as-is if not a string
    if not html_string:
        return ""
    
    text_content = html_string
    # Decode quoted-printable first if it seems present or likely
    # A simple check for common QP patterns like "=" followed by newline or hex
    if "=\n" in text_content or re.search(r'=[0-9A-Fa-f]{2}', text_content):
        try:
            decoded_bytes = quopri.decodestring(text_content.encode('utf-8', 'ignore'))
            text_content = decoded_bytes.decode('utf-8', 'ignore')
        except Exception as e:
            logging.warning(f"Quoted-printable decoding failed during strip_html: {e}. Proceeding with original for HTML stripping.")
            # text_content remains original html_string if quopri fails

    # Remove script and style elements
    clean_text = re.sub(r'<script[^>]*?>.*?</script>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'<style[^>]*?>.*?</style>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    # Remove all other HTML tags
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    # Replace common HTML entities
    clean_text = clean_text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    
    lines = [line.strip() for line in clean_text.splitlines()]
    clean_text = '\n'.join([line for line in lines if line])
    
    # Remove non-printable ASCII characters (allow space to tilde, and newline)
    clean_text = re.sub(r'[^ -~\n]+', '', clean_text) 
    # Consolidate multiple whitespace characters into a single space, then strip ends
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def get_sections_from_json(data: Union[Dict, list, str], default_title_prefix="Section", min_length: int = 50) -> List[Tuple[str, str]]:
    """
    Extracts sections from parsed JSON data.
    Prioritizes a root-level 'content' key. If not found, looks for a 'sections' key.
    Each section is a tuple of (name, content).
    Filters out sections with content shorter than min_length.
    """
    sections = []
    
    # Case 1: Root-level 'content' key (typical for news articles)
    if "content" in data and isinstance(data["content"], str):
        content = data["content"]
        if len(content) >= min_length:
            # Use 'title' from data if available, otherwise None (orchestrator might pass a title separately)
            sections.append((data.get("title"), content))
        return sections

    # Case 2: 'sections' key with a dictionary of section_name: section_content
    if "sections" in data and isinstance(data["sections"], dict):
        for section_name, section_content in data["sections"].items():
            if isinstance(section_content, str) and len(section_content) >= min_length:
                sections.append((section_name, section_content))
            elif isinstance(section_content, list): # Handle list of strings if applicable
                temp_content = "\n".join(str(item) for item in section_content if isinstance(item, str))
                if len(temp_content) >= min_length:
                    sections.append((section_name, temp_content))
        if sections: # If we found sections here, return them
            return sections

    # Case 3: The JSON root is a list of objects (e.g., list of articles or items)
    # This might be less common for single document processing by this agent but included for robustness.
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                title = item.get("title", item.get("header", f"item_{i+1}"))
                content_str = ""
                if "content" in item and isinstance(item["content"], str):
                    content_str = item["content"]
                elif "text" in item and isinstance(item["text"], str): # common alternative
                    content_str = item["text"]
                else: # fallback to stringifying the item
                    content_str = json.dumps(item)
                
                if len(content_str) >= min_length:
                    sections.append((title, content_str))
        if sections:
            return sections

    # Case 4: Fallback for flat JSON (no 'content' at root, no 'sections', not a list)
    # Concatenate string values, or stringified complex values.
    # Use 'title' key for section name if present.
    if isinstance(data, dict) and not sections: 
        combined_content = ""
        section_title = data.get("title", "document_content") # Use 'title' or default
        
        # Create content string, excluding the title if it was used for section_title
        # Only exclude if section_title was actually derived from data.get("title")
        temp_data_for_content = { 
            k:v for k,v in data.items() 
            if not (k == "title" and data.get("title") == section_title) 
        }

        if not temp_data_for_content and data.get("title") == section_title : # If data only contained a title used as section_title
             if isinstance(section_title, str) and len(section_title) >= min_length:
                 sections.append((None, section_title)) # Treat title as content, with no separate section name
                 return sections
             else: # No other content to extract
                 return []

        for key, value in temp_data_for_content.items():
            if isinstance(value, str):
                combined_content += f"{key}: {value}\n"
            elif isinstance(value, (list, dict)):
                combined_content += f"{key}: {json.dumps(value)}\n"
            else:
                combined_content += f"{key}: {str(value)}\n" # stringify other types
        
        stripped_content = combined_content.strip()
        if stripped_content and len(stripped_content) >= min_length:
            sections.append((section_title, stripped_content))
        # If after all this, the only meaningful thing was the title itself, and it wasn't used as section_title 
        # (e.g. section_title defaulted to "document_content" but data["title"] exists)
        # Or if content was empty but title is meaningful.
        elif not stripped_content and data.get("title") and data.get("title") != section_title and len(data["title"]) >= min_length:
             sections.append((None, data["title"])) # Treat title as content

        if sections:
            return sections
            
    # Case 5: If still no sections, and it's a dict, treat the whole stringified JSON as content.
    # This is the ultimate fallback.
    if not sections and isinstance(data, dict):
        content_str = json.dumps(data)
        if len(content_str) >= min_length:
            sections.append((data.get("title", "full_document_content"), content_str))
            
    # Try to extract news article structure (if applicable)
    if isinstance(data, dict) and "content" in data: # Simplified check for news-like structure
        title = data.get("title", "News Article") 
        content = data.get("content", "")
        
        if isinstance(content, str):
            # Always attempt to clean content for news, especially if HTML is indicated or suspected
            # The raw_content_type field can provide a hint.
            if data.get("raw_content_type") == "html" or "<" in content and ">" in content: # Basic check for HTML tags
                logging.debug(f"HTML detected or indicated for news title '{title}'. Applying strip_html.")
                content = strip_html(content)
            else: # If not explicitly HTML, still good to ensure it's clean text for LLM
                content = strip_html(content) # Apply strip_html to clean potential QP and normalize whitespace
        elif content is not None: # If content is not a string but exists (e.g. number, boolean)
             content = str(content) # Convert to string, then strip_html will normalize
             content = strip_html(content)
        else:
            content = "" # Ensure content is an empty string if None
            
        if title and content.strip(): # Ensure content is not just whitespace after stripping
            return [(str(title), content)]
        elif title: # If content became empty but title exists, return title with placeholder
            logging.warning(f"Content for news article '{title}' is empty after cleaning. Returning title only.")
            return [(str(title), "[Content not available or empty after cleaning]")]

    return sections

def get_full_text(data: Union[Dict, List, str]) -> str:
    """Converts the input data (parsed JSON) into a single string for summarization."""
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, indent=2)
    except Exception as e:
        logger.error(f"Failed to serialize data to JSON string in get_full_text: {e}")
        return str(data) # Fallback to generic string conversion

class TextChunker:
    def __init__(self, max_chunk_size: int = 2000, overlap: int = 100, token_model: str = "gpt-4"):
        """
        Initializes the TextChunker.

        Args:
            max_chunk_size (int): The maximum size of each chunk (in characters or tokens, depending on strategy).
                                  For simplicity, this implementation uses character count.
            overlap (int): The number of characters of overlap between consecutive chunks.
            token_model (str): The name of the tokenizer model to use if token-based chunking is implemented.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        # self.tokenizer = tiktoken.encoding_for_model(token_model) # Uncomment for token-based chunking
        logger.info(f"TextChunker initialized with max_chunk_size={max_chunk_size}, overlap={overlap}")

    def chunk_text(self, text: str, section_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Splits a long text into smaller chunks with overlap.

        Args:
            text (str): The text content to be chunked.
            section_name (Optional[str]): The name of the section this text belongs to.

        Returns:
            List[Dict[str, Any]]: A list of chunks, where each chunk is a dictionary 
                                  with 'content', 'section_name', and 'chunk_index'.
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.max_chunk_size
            current_chunk_content = text[start:end]
            
            chunks.append({
                "content": current_chunk_content,
                "section_name": section_name,
                "chunk_index": chunk_index
            })
            
            chunk_index += 1
            # Move start, ensuring it doesn't go beyond text_len if max_chunk_size is huge
            if self.max_chunk_size - self.overlap <= 0: # Avoid infinite loop if overlap >= max_chunk_size
                 start = end 
            else:
                 start += self.max_chunk_size - self.overlap

            # If the remaining part is smaller than overlap, just take the rest (avoid tiny last chunk due to overlap rule)
            if text_len - start < self.overlap and start < text_len:
                current_chunk_content = text[start:]
                if chunks[-1]["content"].endswith(current_chunk_content[:self.overlap]): # Avoid duplicate if fully overlapped
                    if len(current_chunk_content) > self.overlap: # if there is new content
                         chunks.append({"content": text[start+self.overlap:], "section_name": section_name, "chunk_index": chunk_index})
                elif len(current_chunk_content) > 0 : 
                     chunks.append({"content": current_chunk_content, "section_name": section_name, "chunk_index": chunk_index})
                break # Exit loop as we've processed the remainder

        # Ensure the very last character is included if loop finishes due to start >= text_len
        # This logic might be redundant with the refinement above, but good as a safeguard.
        if start < text_len and end < text_len and text[end:]:
            # This case should ideally be handled by the loop logic if the remaining part is substantial enough
            # or by the small remainder handling. If text[end:] is very small, it might be skipped.
            pass 

        logger.debug(f"Chunked text from section '{section_name}' into {len(chunks)} chunks.")
        return chunks

    def chunk_document(self, file_path: str, sections: List[Tuple[Optional[str], str]]) -> List[Dict[str, Any]]:
        """
        Processes a document, extracts text from sections, and chunks it.

        Args:
            file_path (str): Path to the document file (for logging/metadata purposes).
            sections (List[Tuple[Optional[str], str]]): A list of (section_name, section_content) tuples.

        Returns:
            List[Dict[str, Any]]: A list of all chunks from the document.
        """
        all_chunks = []
        total_section_count = len(sections)
        logger.info(f"Starting chunking for document: {file_path}, {total_section_count} sections found.")

        for i, (section_name, section_content) in enumerate(sections):
            section_identifier = section_name if section_name else f"OriginalDocTitleInSection{i+1}" # More descriptive default
            logger.debug(f"Processing section {i+1}/{total_section_count}: '{section_identifier}' (length: {len(section_content)})")
            
            if not section_content or not section_content.strip():
                logger.warning(f"Skipping empty or whitespace-only section: '{section_identifier}' in {file_path}")
                continue

            text_chunks = self.chunk_text(section_content, section_name=section_identifier)
            all_chunks.extend(text_chunks)
            logger.debug(f"Section '{section_identifier}' produced {len(text_chunks)} chunks.")
        
        logger.info(f"Document {file_path} processed into {len(all_chunks)} total chunks.")
        return all_chunks

# Example usage (for testing)
if __name__ == '__main__':
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO)

    # Test get_sections_from_json
    print("\n--- Testing get_sections_from_json ---")
    
    news_data_content_only = {"title": "Big News", "content": "This is the main content of the news article. It's quite interesting."}
    print(f"News (content only): {get_sections_from_json(news_data_content_only)}")

    news_data_full = {
        "id": "news123",
        "title": "Another Story",
        "date": "2024-01-01",
        "source_type": "web",
        "source_name": "News Site",
        "url": "http://example.com/news",
        "content": "Detailed content about the story. It unfolds over several paragraphs..."
    }
    print(f"News (full article): {get_sections_from_json(news_data_full)}")

    sec_data_sections = {
        "header": "SEC Filing XYZ",
        "sections": {
            "Item 1": "This is item 1 content.",
            "Item 2": "This is item 2 content, a bit longer.",
            "Item 1A": "Risk factors are detailed here."
        }
    }
    print(f"SEC (with sections): {get_sections_from_json(sec_data_sections)}")

    flat_json_data = {"field1": "value1", "field2": "value2", "notes": "Some notes here."}
    print(f"Flat JSON (no specific keys): {get_sections_from_json(flat_json_data)}")
    
    flat_json_with_title = {"title": "My Document", "fieldA": "Data A", "fieldB": "Data B"}
    print(f"Flat JSON (with title): {get_sections_from_json(flat_json_with_title)}")

    data_only_title = {"title": "This title is the actual content."}
    print(f"Flat JSON (only title as content): {get_sections_from_json(data_only_title)}")

    json_list_data = [
        {"title": "Item A", "content": "Content of item A."},
        {"header": "Item B", "text": "Text for item B."}
    ]
    print(f"JSON List: {get_sections_from_json(json_list_data)}")

    empty_data = {}
    print(f"Empty JSON: {get_sections_from_json(empty_data)}")

    minimal_content = {"content": "short"}
    print(f"Minimal content (short): {get_sections_from_json(minimal_content, min_length=10)}")
    print(f"Minimal content (ok): {get_sections_from_json(minimal_content, min_length=3)}")


    # Test TextChunker
    print("\n--- Testing TextChunker ---")
    chunker = TextChunker(max_chunk_size=150, overlap=30)
    
    # Test with sections from get_sections_from_json output
    print("\nTesting chunker with news_data_full:")
    example_sections = get_sections_from_json(news_data_full)
    if example_sections:
        doc_chunks = chunker.chunk_document("dummy_news_path.json", example_sections)
        print(f"Chunks from news_data_full ({len(doc_chunks)} chunks):")
        for i, chunk_data in enumerate(doc_chunks):
            print(f"  Chunk {i}: Section='{chunk_data['section_name']}', Index={chunk_data['chunk_index']}")
            print(f"    Content: '{chunk_data['content'][:100].replace('\n', ' ')}...'")

    else:
        print("No sections extracted from news_data_full for chunker test.")

    print("\nTesting chunker with sec_data_sections:")
    example_sections_sec = get_sections_from_json(sec_data_sections)
    if example_sections_sec:
        doc_chunks_sec = chunker.chunk_document("dummy_sec_path.json", example_sections_sec)
        print(f"Chunks from sec_data_sections ({len(doc_chunks_sec)} chunks):")
        for i, chunk_data in enumerate(doc_chunks_sec):
            print(f"  Chunk {i}: Section='{chunk_data['section_name']}', Index={chunk_data['chunk_index']}")
            print(f"    Content: '{chunk_data['content'][:100].replace('\n', ' ')}...'")
    else:
        print("No sections extracted from sec_data_sections for chunker test.")

    print("\nTesting chunker with flat_json_with_title:")
    example_sections_flat = get_sections_from_json(flat_json_with_title)
    if example_sections_flat:
        doc_chunks_flat = chunker.chunk_document("dummy_flat_path.json", example_sections_flat)
        print(f"Chunks from flat_json_with_title ({len(doc_chunks_flat)} chunks):")
        for i, chunk_data in enumerate(doc_chunks_flat):
            print(f"  Chunk {i}: Section='{chunk_data['section_name']}', Index={chunk_data['chunk_index']}")
            print(f"    Content: '{chunk_data['content'][:100].replace('\n', ' ')}...'")
    else:
        print("No sections extracted from flat_json_with_title for chunker test.")

    print("\nTesting chunker with data_only_title:")
    example_sections_title_only = get_sections_from_json(data_only_title)
    if example_sections_title_only:
        doc_chunks_title_only = chunker.chunk_document("dummy_title_only.json", example_sections_title_only)
        print(f"Chunks from data_only_title ({len(doc_chunks_title_only)} chunks):")
        for i, chunk_data in enumerate(doc_chunks_title_only):
            print(f"  Chunk {i}: Section='{chunk_data['section_name']}', Index={chunk_data['chunk_index']}")
            print(f"    Content: '{chunk_data['content'][:100].replace('\n', ' ')}...'")
    else:
        print("No sections extracted from data_only_title for chunker test.")

    # Test chunk_text directly for edge cases
    print("\n--- Testing chunk_text directly ---")
    small_text = "This is a small text."
    chunks_small = chunker.chunk_text(small_text, "SmallTextSection")
    print(f"Small text ('{small_text}') chunks: {len(chunks_small)}")
    # for ch in chunks_small: print(ch)

    exact_size_text = "a" * 150
    chunks_exact = chunker.chunk_text(exact_size_text, "ExactSizeSection")
    print(f"Exact size text chunks: {len(chunks_exact)}")
    # for ch in chunks_exact: print(ch['content'])

    overlap_test_text = "a" * 200 # max_chunk_size=150, overlap=30. Chunk1: 0-150. Chunk2: 120-200 (80 chars)
    chunks_overlap = chunker.chunk_text(overlap_test_text, "OverlapTestSection")
    print(f"Overlap test text chunks: {len(chunks_overlap)}")
    # for ch in chunks_overlap: print(f"Index {ch['chunk_index']}: len={len(ch['content'])}, content='{ch['content'][:20]}...'")

    very_small_text_than_overlap = "abc" # overlap is 30
    chunks_tiny = chunker.chunk_text(very_small_text_than_overlap, "TinySection")
    print(f"Tiny text ('{very_small_text_than_overlap}') chunks: {len(chunks_tiny)}")
    # for ch in chunks_tiny: print(ch) 