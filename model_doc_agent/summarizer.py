#!/usr/bin/env python3
import argparse
import logging
import os
from src import orchestrator, cache, chunker, summarization, metadata, writer

def main():
    parser = argparse.ArgumentParser(
        prog="summarizer",
        description="Hierarchical SEC Filing Summarization Agent"
    )
    parser.add_argument("--mode", choices=["file", "node", "master", "cross", "sub_folder_content", "sub_folder_cross_sectional"], required=True,
                        help="Summarization mode: file, node, master, cross, sub_folder_content, or sub_folder_cross_sectional")
    parser.add_argument("--input_path", required=True, help="Path to input JSON file, directory of filings, or a parent directory for sub-folder modes (e.g., Data/Banks/Bank_of_Montreal, Data/news/reg/cfpb/2025-01, or Data/Banks for sub-folder modes)")
    parser.add_argument("--output_dir", required=True, help="Directory to write output summaries and metadata")
    parser.add_argument("--prompt_set", default="sec_prompts_v1.json",
                        help="Path to prompt-set JSON file defining prompts for each mode (default: sec_prompts_v1.json in agent root)")
    parser.add_argument("--max_words", type=int, default=2000,
                        help="Maximum words per chunk (approximate chunk size limit)")
    parser.add_argument("--retries", type=int, default=3, help="Max retry attempts for LLM calls on failure")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching of processed content")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize components
    cache_manager = cache.CacheManager(cache_dir=".cache", enabled=not args.no_cache)
    summarizer = summarization.LLMSummarizer(prompt_set_path=args.prompt_set,
                                            max_retries=args.retries)
    meta_generator = metadata.MetadataGenerator()
    output_writer = writer.OutputWriter()

    # Run the orchestrator for the specified mode
    orchestrator.run_summarization(mode=args.mode,
                                   input_dir=args.input_path,
                                   output_dir=args.output_dir,
                                   max_words=args.max_words,
                                   cache_manager=cache_manager,
                                   chunker=chunker,
                                   summarizer=summarizer,
                                   meta_generator=meta_generator,
                                   output_writer=output_writer)
    
if __name__ == "__main__":
    main() 