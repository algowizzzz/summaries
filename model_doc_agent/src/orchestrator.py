import os
import json
import logging
from datetime import datetime
import re
import math # Added for ceil/floor

from model_doc_agent.src.summarization import LLMSummarizer # Added for token estimation

def run_summarization(mode, input_dir, output_dir, max_words,
                      cache_manager, chunker, summarizer, meta_generator, output_writer, prompt_set_path=None, args=None):
    """
    Coordinate the summarization process for all files in input_dir according to the specified mode.
    The prompt_set_path is passed to the summarizer if provided, otherwise summarizer uses its default.
    `args` can be the full argparse object if needed for specific flags like --theme.
    """
    logging.info(f"Starting summarization mode '{mode}' for input directory: {input_dir}")

    files_to_process = []
    # Adjusted file discovery
    # Check if input_dir points directly to a YYYY-MM news folder
    # or if it's a higher level directory like Data/Banks or Data/news/reg/cfpb
    potential_news_year_month_folder = os.path.basename(input_dir) # e.g. "2025-01"
    parent_of_potential_news_folder = os.path.basename(os.path.dirname(input_dir))

    # Heuristic to check if input_dir is a specific YYYY-MM news folder
    is_specific_news_month_folder = False
    if re.match(r"^\d{4}-\d{2}$", potential_news_year_month_folder) and \
       any(news_cat in input_dir for news_cat in ["/news/CA Banks/", "/news/reg/", "/news/usbank/"]):
        is_specific_news_month_folder = True

    if is_specific_news_month_folder:
        # Process only this specific YYYY-MM news folder
        for fname in os.listdir(input_dir):
            if fname.lower().endswith(".json") and fname.lower() != "index.json":
                files_to_process.append(os.path.join(input_dir, fname))
    else:
        # General walk for Data/Banks, Data/mutual_fund_filings, or higher-level Data/news paths
        for root, _, files in os.walk(input_dir):
            # Filter to include only relevant subdirectories for news, or all for banks/filings
            if "/news/" in root or "\\news\\" in root: # If processing under a news directory
                # Only process if the current root is a YYYY-MM folder
                if not re.search(r"/\d{4}-\d{2}$", root) and not re.search(r"\\\d{4}-\d{2}$", root):
                    continue # Skip if not a YYYY-MM folder within news hierarchy
            
            for fname in files:
                if fname.lower().endswith(".json") and fname.lower() != "index.json":
                    files_to_process.append(os.path.join(root, fname))
    
    if not files_to_process:
        logging.warning(f"No processable JSON files (excluding index.json) found in {input_dir} or its relevant subdirectories.")
        return
    
    logging.info(f"Found {len(files_to_process)} files to process.")

    for filepath in files_to_process:
        logging.info(f"Processing file: {filepath}")
        is_news_article = "/news/" in filepath or "\\news\\" in filepath
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from {filepath}: {e}")
            continue
        except Exception as e:
            logging.error(f"Failed to read file {filepath}: {e}")
            continue
        
        # Correct rel_path for output structure, relative to the initial input_dir given by user
        # This ensures that if input_dir was Data/Banks, rel_path starts from Bank_NameA/filing.json
        # If input_dir was Data/news/reg/cfpb, rel_path starts from 2025-01/article.json
        # If input_dir was Data/, rel_path starts from Banks/Bank_NameA/filing.json
        
        # To make output structure consistent like <output_dir>/<original_data_subpath>/
        # we need to make rel_path relative to the *root* of categories (Banks, mutual_fund_filings, news)
        # This requires identifying the base of these categories from the input_dir.

        # Simplification: output path construction will use a subfolder for the mode, then mirror from `input_dir`.
        # The user-provided `input_dir` becomes the base for mirroring in `output_dir`.
        # Example: if input_dir = Data/Banks/Bank_X, output will be output_dir/file-level/Bank_X/...
        # Example: if input_dir = Data/news/reg/cfpb/2025-01, output will be output_dir/file-level/2025-01/...

        rel_path_from_actual_input = os.path.relpath(filepath, input_dir) # Path relative to the CLI --input-dir
        base_name_ext = os.path.basename(filepath)
        base_name = os.path.splitext(base_name_ext)[0]
        
        output_mode_dir = os.path.join(output_dir, f"{mode}-level") # e.g. out/file-level
        # current_out_dir will be like: out/file-level/Bank_of_Montreal OR out/file-level/2025-01 (if input_dir was specific month)
        current_out_dir = os.path.join(output_mode_dir, os.path.dirname(rel_path_from_actual_input))
        os.makedirs(current_out_dir, exist_ok=True)

        # Determine filing_type for prompts (e.g., 40-F, 6-K, or "news")
        effective_filing_type = "default" # Fallback
        if is_news_article:
            effective_filing_type = "news"
            # For news, specific type might come from data itself, e.g. data.get("article_type")
            # but for prompt selection, "news" as a category is likely enough.
        else: # It's a filing
            # Attempt to extract filing type from filename like 6-K_...json or 40-F_...json
            fn_parts = base_name_ext.split('_')
            if len(fn_parts) > 1:
                potential_form_type = fn_parts[0]
                if re.match(r"^[A-Z0-9\-]+", potential_form_type) and len(potential_form_type) < 10: # Match common SEC form patterns
                    effective_filing_type = potential_form_type
                    logging.debug(f"Determined filing type from filename: {effective_filing_type}")
            
            # If filename parsing didn't yield a specific type (still "default"), try from JSON content
            if effective_filing_type == "default" and isinstance(data, dict):
                json_form_type = data.get("form_type", data.get("type"))
                if json_form_type and isinstance(json_form_type, str) and len(json_form_type) < 15: # Basic sanity check
                    effective_filing_type = json_form_type.upper()
                    logging.debug(f"Determined filing type from JSON content ('form_type' or 'type'): {effective_filing_type}")

        if mode == "file" or (mode == "news" and is_news_article): # Consolidate file and news single-item summarization
            actual_mode_for_summarizer = "file" # Ensures "file" mode is passed to summarizer, filing_type distinguishes content
            
            text_to_summarize = ""
            summarizer_kwargs = {}

            if is_news_article:
                text_to_summarize = data.get("content", "")
                summarizer_kwargs['title'] = data.get("title", base_name)
                
                # A1: Token limit handling for large news articles
                estimated_tokens = LLMSummarizer.estimate_tokens(text_to_summarize)
                # Model's actual max prompt tokens (e.g., Haiku is 200k context window, but prompt itself is part of it)
                # Let's use a practical limit for the *content* part of the prompt.
                # The prompt template itself takes some tokens.
                content_token_limit = 190000 # Max tokens for the content part of the prompt
                chunking_method_for_meta = "none"

                if estimated_tokens > content_token_limit:
                    logging.warning(f"News article {filepath} estimated at {estimated_tokens} tokens, exceeding content limit of {content_token_limit}. Dynamic chunking...")
                    chunking_method_for_meta = "sequential_summaries_dynamic_char_chunks"
                    
                    # Dynamically calculate chunk size
                    safe_tokens_per_chunk = 50000 # Target tokens for each chunk to be summarized (well within LLM's capacity for a single call)
                    num_chunks_ideal = math.ceil(estimated_tokens / safe_tokens_per_chunk)
                    
                    # Ensure at least 2 chunks if we decided to chunk.
                    num_chunks_ideal = max(2, num_chunks_ideal)

                    # Calculate character chunk size based on ideal number of chunks
                    # Use a more conservative char_per_token (e.g., 3) for calculating char chunk size from target tokens
                    # to ensure resulting chunks are small enough token-wise.
                    # However, TextChunker uses character count for its chunk_size.
                    # So, we divide total chars by num_chunks_ideal.
                    
                    total_chars = len(text_to_summarize)
                    char_chunk_size_target = math.floor(total_chars / num_chunks_ideal)
                    
                    # Ensure a minimum character chunk size to avoid overly tiny chunks if num_chunks_ideal is very high.
                    # And a maximum to respect overall limits if total_chars is astronomical.
                    # Practical char size for a 50k token chunk (at ~3-4 chars/token) would be 150k-200k chars.
                    # Let's set chunk_size for the splitter to be this target, but ensure it's not too small.
                    min_char_chunk_for_splitter = 20000 # Min characters for a meaningful summary chunk
                    char_chunk_size_for_splitter = max(min_char_chunk_for_splitter, char_chunk_size_target)
                    
                    # The overlap should be smaller than the chunk size
                    char_chunk_overlap = max(1000, int(char_chunk_size_for_splitter * 0.1))
                    
                    logging.info(f"Dynamic chunking: ideal_chunks={num_chunks_ideal}, target_char_size_per_chunk={char_chunk_size_target}, splitter_chunk_size={char_chunk_size_for_splitter}, overlap={char_chunk_overlap}")

                    news_chunks = chunker.chunk_text(
                        text_to_summarize, 
                        chunk_size=char_chunk_size_for_splitter, 
                        chunk_overlap=char_chunk_overlap
                    )
                    
                    if not news_chunks: # Should not happen if text_to_summarize is not empty
                        logging.error(f"Chunking resulted in no news_chunks for {filepath}. Skipping summarization.")
                        text_to_summarize = "Error: Chunking failed to produce any text segments."
                        chunking_method_for_meta = "error_chunking_failed"

                    chunk_summaries = []
                    for i, chunk_text in enumerate(news_chunks):
                        if not chunk_text.strip():
                            logging.debug(f"Skipping empty chunk {i+1} for {filepath}")
                            continue
                        logging.info(f"Summarizing chunk {i+1}/{len(news_chunks)} for {filepath} (length: {len(chunk_text)} chars)...")
                        
                        # Create new kwargs for this chunk summary
                        chunk_summarizer_kwargs = summarizer_kwargs.copy() # Inherit title
                        chunk_summarizer_kwargs['content'] = chunk_text

                        chunk_summary = summarizer.summarize(mode=actual_mode_for_summarizer,
                                                             filing_type=effective_filing_type,
                                                             **chunk_summarizer_kwargs)
                        if not chunk_summary.startswith("Error:"):
                            chunk_summaries.append(chunk_summary)
                        else:
                            logging.error(f"Error summarizing chunk {i+1} for {filepath}: {chunk_summary}")
                            chunk_summaries.append(f"[Error summarizing chunk {i+1}: {chunk_summary}]") # Include error in combined

                    if chunk_summaries:
                        text_to_summarize = "\\n\\n---\\n\\n".join(chunk_summaries)
                        logging.info(f"Combined {len(chunk_summaries)} chunk summaries for {filepath}.")
                    elif not text_to_summarize.startswith("Error:"): # If not already set to an error by chunking failure
                        logging.error(f"No valid chunk summaries generated for {filepath}, though chunking produced segments. Proceeding with error message.")
                        text_to_summarize = "Error: Failed to generate summary from chunks (no valid chunk summaries)."
                # End of A1 chunking logic
            else: # Filing
                text_to_summarize = chunker.get_full_text(data)
                company_name = data.get("company_name", "Unknown Company")
                doc_title_str = f'{company_name} - {effective_filing_type}' if effective_filing_type != "default" else company_name
                
                summarizer_kwargs['source_filename'] = base_name_ext
                summarizer_kwargs['effective_filing_type'] = effective_filing_type
                summarizer_kwargs['document_title'] = doc_title_str
                summarizer_kwargs['current_date'] = datetime.now().strftime("%Y-%m-%d")
            
            if not text_to_summarize.strip():
                logging.warning(f"No content to summarize for {filepath}. Skipping.")
                continue

            content_hash = cache_manager.hash_content(text_to_summarize + json.dumps(summarizer_kwargs)) # Hash content + key context

            if cache_manager.is_cached(content_hash):
                logging.info(f"Item {filepath} (mode: {actual_mode_for_summarizer}) is already cached. Skipping.")
                continue
            
            summarizer_kwargs['content'] = text_to_summarize
            summary_text = summarizer.summarize(mode=actual_mode_for_summarizer, 
                                              filing_type=effective_filing_type, 
                                              **summarizer_kwargs)
            cache_manager.mark_cached(content_hash)
            
            output_file_base = base_name
            if is_news_article:
                 output_file_base = data.get("id", base_name) # Use news ID for filename if available
                 output_file_base = re.sub(r'[^\w\-_.]', '_', output_file_base) # Sanitize ID

            output_path_md = os.path.join(current_out_dir, f"{output_file_base}_summary.md")
            output_path_json = os.path.join(current_out_dir, f"{output_file_base}_meta.json")
            
            # Pass the original JSON data for metadata generation for news
            metadata_kwargs = data if is_news_article else {"doc_type": effective_filing_type}
            if not is_news_article:
                 metadata_kwargs["form_type"] = effective_filing_type # From README for filings
            
            meta = meta_generator.generate_metadata(summary_text=summary_text, 
                                                   source_path=filepath, 
                                                   mode=actual_mode_for_summarizer, 
                                                   is_news=is_news_article,
                                                   news_data=data if is_news_article else None,
                                                   doc_type=effective_filing_type if not is_news_article else data.get("article_type"),
                                                   llm_model_used=summarizer.model_name, # Pass LLM model
                                                   chunking_method=chunking_method_for_meta if is_news_article else "none" # Add chunking info
                                                   )

            output_writer.write_summary(summary_text, output_path_md)
            output_writer.write_metadata(meta, output_path_json)
            logging.info(f"{actual_mode_for_summarizer.capitalize()}-level summary for {base_name_ext} written to {output_path_md}")

        elif mode == "node" and not is_news_article: # Node mode is for filings
            sections = chunker.get_sections_from_json(data)
            if not sections:
                logging.warning(f"No sections found in {filepath} for node mode. Skipping.")
                continue

            for i, (sec_title, sec_text) in enumerate(sections):
                sec_id = f"section_{i+1}" # Generate a unique section ID
                actual_sec_title = sec_title if sec_title else f"Untitled Section {sec_id}"

                if not sec_text.strip():
                    logging.debug(f"Skipping empty section {sec_id} ('{actual_sec_title}') in {filepath}")
                    continue
                
                section_unique_content = f"{filepath}:{sec_id}:{sec_text}" # sec_id is now defined
                content_hash = cache_manager.hash_content(section_unique_content)

                if cache_manager.is_cached(content_hash):
                    logging.info(f"Node {sec_id} ('{actual_sec_title}') in {filepath} is already cached. Skipping.")
                    continue
                
                # Prepare kwargs for node summary template
                # Refined document_title for node mode
                company_name_node = data.get("company_name", "Unknown Company")
                doc_title_for_node = f'{company_name_node} - {effective_filing_type}' if effective_filing_type != "default" else company_name_node

                summary_text = summarizer.summarize(mode="node", 
                                                  filing_type=effective_filing_type,
                                                  content=sec_text, 
                                                  section_title=actual_sec_title, 
                                                  document_title=doc_title_for_node, # Use refined document_title
                                                  effective_filing_type=effective_filing_type
                                                  )
                cache_manager.mark_cached(content_hash)
                
                output_path_md = os.path.join(current_out_dir, f"{base_name}_{sec_id}_summary.md")
                output_path_json = os.path.join(current_out_dir, f"{base_name}_{sec_id}_meta.json")
                
                meta = meta_generator.generate_metadata(summary_text=summary_text, 
                                                       source_path=filepath, mode="node", 
                                                       section_id=sec_id, section_title=actual_sec_title, 
                                                       doc_type=effective_filing_type,
                                                       llm_model_used=summarizer.model_name,
                                                       # Pass document_title and effective_filing_type for consistency if needed by metadata, though already in prompt
                                                       document_title_for_node=doc_title_for_node # Example of adding more context to meta if desired
                                                       )
                output_writer.write_summary(summary_text, output_path_md)
                output_writer.write_metadata(meta, output_path_json)
                logging.info(f"Node-level summary for section {sec_id} ('{actual_sec_title}') written to {output_path_md}")

        elif mode == "master" and not is_news_article: # Master mode is for filings
            logging.debug(f"Attempting master summary for {filepath}")

            # Path to where node-level summaries for this file *should* be
            # current_out_dir is ALREADY the mode-specific output dir for the CURRENT file being processed,
            # e.g., output_dir/master-level/path/to/doc/
            # So, node summaries would be in a parallel dir: output_dir/node-level/path/to/doc/
            
            # Correct base for node summaries using rel_path_from_actual_input
            # rel_path_from_actual_input is like: Bank_X/doc.json or 2025-01/article.json
            # os.path.dirname(rel_path_from_actual_input) gives: Bank_X or 2025-01
            node_summary_base_dir = os.path.join(output_dir, "node-level", os.path.dirname(rel_path_from_actual_input))
            
            node_summaries_content = []
            
            # Heuristic: check for section files.
            # base_name is the filename without extension, e.g., "ishares_document_1"
            i = 1
            while True:
                expected_node_summary_file = os.path.join(node_summary_base_dir, f"{base_name}_section_{i}_summary.md")
                if os.path.exists(expected_node_summary_file):
                    try:
                        with open(expected_node_summary_file, 'r', encoding='utf-8') as nf:
                            node_summaries_content.append(nf.read())
                        logging.debug(f"Read node summary: {expected_node_summary_file}")
                    except Exception as e:
                        logging.error(f"Error reading node summary {expected_node_summary_file}: {e}")
                        # If a node summary is corrupt, we might choose to skip this master summary
                        break # Exit loop on error reading a node summary
                else:
                    logging.debug(f"No more node summaries found for {base_name} (checked for section {i} at {expected_node_summary_file}).")
                    break # No more section files with this naming convention
                i += 1

            if not node_summaries_content:
                logging.warning(f"No node-level summaries found for {filepath} in {node_summary_base_dir}. Cannot generate master summary. Skipping.")
                continue

            combined_node_summaries = "\n\n---\n\n".join(node_summaries_content) # Separator for clarity
            
            # Content hash for master summary will be based on combined node summaries + identifying info
            # Using filepath and "master_from_nodes" to differentiate from a master summary made from full text
            content_hash = cache_manager.hash_content(f"{filepath}:master_from_nodes:{combined_node_summaries}")

            if cache_manager.is_cached(content_hash):
                logging.info(f"Master summary for {filepath} (from nodes) is already cached. Skipping.")
                continue

            # Refined document_title for master mode - ensure 'data' (original JSON) is loaded if not already
            # For safety, re-check if 'data' needs to be loaded or if it's assumed to be in scope.
            # Assuming 'data' (from the original file json.load) is still in scope here.
            company_name_master = data.get("company_name", "Unknown Company")
            doc_title_for_master = f'{company_name_master} - {effective_filing_type}' if effective_filing_type != "default" else company_name_master
            
            # Current date for the template if needed (though master template doesn't explicitly list it, good to have)
            current_date_str = datetime.now().strftime("%Y-%m-%d")

            summarizer_kwargs = {
                'content': combined_node_summaries, 
                'document_title': doc_title_for_master, 
                'effective_filing_type': effective_filing_type,
                'current_date': current_date_str # Adding for completeness, though not in current master template
            }

            summary_text = summarizer.summarize(mode="master",
                                              filing_type=effective_filing_type,
                                              **summarizer_kwargs)
            cache_manager.mark_cached(content_hash)

            # Output paths remain similar, current_out_dir is already output_dir/master-level/original_sub_path/
            output_path_md = os.path.join(current_out_dir, f"{base_name}_master_summary.md")
            output_path_json = os.path.join(current_out_dir, f"{base_name}_master_meta.json")
            
            meta = meta_generator.generate_metadata(summary_text=summary_text, 
                                                   source_path=filepath, mode="master", 
                                                   doc_type=effective_filing_type,
                                                   llm_model_used=summarizer.model_name,
                                                   # Potentially add list of node_summary_ids as parent_ids if we had them
                                                   )
            output_writer.write_summary(summary_text, output_path_md)
            output_writer.write_metadata(meta, output_path_json)
            logging.info(f"Master-level summary (from nodes) for {base_name_ext} written to {output_path_md}")

        elif mode == "cross":
            # Cross-sectional mode needs a collection of inputs, not single file processing.
            # This requires a different invocation strategy or pre-gathered inputs.
            # For now, this will be a placeholder if called on a single file.
            # A real implementation would likely process files gathered *outside* this loop.
            logging.warning(f"Cross-sectional mode is not designed to run on individual files like {filepath} in this loop. A collection of inputs is expected.")
            # Example: if args and args.theme are available for a theme-based cross summary
            theme = args.theme if args and hasattr(args, 'theme') else "general_cross_summary"
            
            # The content for cross-summary would typically be a collection of other summaries passed in.
            # For this example, let's assume we are (incorrectly) making a cross-summary of a single doc.
            text_to_summarize = chunker.get_full_text(data)
            if not text_to_summarize.strip(): continue

            content_hash = cache_manager.hash_content(f"{filepath}:cross:{theme}:{text_to_summarize}")
            if cache_manager.is_cached(content_hash): 
                logging.info(f"Cross summary (placeholder) for {filepath}, theme '{theme}' is cached. Skipping.")
                continue

            summary_text = summarizer.summarize(mode="cross", 
                                              filing_type=effective_filing_type, # Might be general or specific if all inputs are same type
                                              content=text_to_summarize, # This should be combined content
                                              theme=theme, 
                                              document_title=base_name_ext)
            cache_manager.mark_cached(content_hash)

            output_path_md = os.path.join(current_out_dir, f"{base_name}_{theme}_cross_summary.md")
            output_path_json = os.path.join(current_out_dir, f"{base_name}_{theme}_cross_meta.json")
            meta = meta_generator.generate_metadata(summary_text=summary_text, 
                                                   source_path=filepath, # This would be a list of paths in real scenario
                                                   mode="cross", 
                                                   theme_type=theme,
                                                   doc_type=effective_filing_type,
                                                   llm_model_used=summarizer.model_name)
            output_writer.write_summary(summary_text, output_path_md)
            output_writer.write_metadata(meta, output_path_json)
            logging.info(f"Cross-sectional (placeholder) summary for {base_name_ext} written to {output_path_md}")
        elif mode != "file" and mode != "news" and mode != "node" and mode != "master": # Handles cases where mode is valid but not for this item type
            logging.warning(f"Mode '{mode}' not applicable to this item type or not implemented for it: {filepath}")

    logging.info("Summarization process completed.")

import re # Add re for orchestrator's regex use 