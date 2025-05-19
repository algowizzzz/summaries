import pytest
import os
import tempfile
import logging
import shutil

# Configure logging to capture DEBUG messages
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to PYTHONPATH to resolve imports
import sys
# Assuming the script is in model_doc_agent/tests/
# Adjust if necessary, or ensure PYTHONPATH is set externally
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_doc_agent.src.orchestrator import run_summarization
# Corrected import: only TextChunker class is needed for instantiation.
# The module-level functions get_full_text and get_sections_from_json are in chunker.py
# but for the news file-mode dynamic chunking path, only chunker.chunk_text is called on the passed object.
from model_doc_agent.src.chunker import TextChunker 
from model_doc_agent.src.summarization import LLMSummarizer
from model_doc_agent.src.metadata import MetadataGenerator
from model_doc_agent.src.writer import OutputWriter
from model_doc_agent.src.cache import CacheManager

# To get the ANTHROPIC_API_KEY from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env')) # Load .env from project root

# pytest_addoption and file_path fixture are now in conftest.py

def test_summarize_single_file_claude(file_path): # file_path fixture is injected by pytest
    """
    Tests summarization of a single file using the orchestrator with real components,
    specifically targeting Claude Haiku.
    Output is written to a temporary directory.
    """
    if not file_path:
        pytest.skip("No file_path provided, skipping test. Use --file_path option.")
        return

    # Resolve file_path relative to project_root
    absolute_file_path = os.path.join(project_root, file_path)

    if not os.path.exists(absolute_file_path):
        pytest.fail(f"Provided file_path does not exist: {absolute_file_path} (resolved from {file_path})")
        return
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env.")
        return

    logging.info(f"Starting test_summarize_single_file_claude for: {absolute_file_path}")

    # Create a temporary directory for output
    # temp_output_dir = tempfile.mkdtemp(prefix="claude_test_out_")
    # For easier inspection, let's create it in a known location and clean it up if it exists.
    # This also helps if the test needs to be re-run and we want to see previous output before it's auto-deleted.
    
    # Output directory within the workspace for easier access during debugging
    # Placed inside model_doc_agent/output/ for consistency with potential manual runs
    # Unique name to avoid clashes with other test runs or manual output
    test_output_base_dir = os.path.join(project_root, "model_doc_agent", "output", "test_single_claude_runs")
    # Sanitize original file_path (relative) for directory naming to keep it shorter and predictable
    sanitized_file_name = os.path.basename(file_path).replace('.', '_') 
    temp_output_dir = os.path.join(test_output_base_dir, sanitized_file_name)

    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir) # Clean up previous run for this file
    os.makedirs(temp_output_dir, exist_ok=True)
    
    logging.info(f"Output will be written to: {temp_output_dir}")

    # Initialize real components
    # Prompt set path - assuming sec_prompts_v1.json is in the project root or accessible path
    # For tests, usually it's relative to where tests are run or part of package data
    # The LLMSummarizer default is "sec_prompts_v1.json" (in model_doc_agent root)
    prompt_set_path = os.path.join(project_root, "model_doc_agent", "sec_prompts_v1.json")
    if not os.path.exists(prompt_set_path):
         pytest.fail(f"Prompt set file not found: {prompt_set_path}")

    llm_summarizer = LLMSummarizer(prompt_set_path=prompt_set_path, model_name="claude-3-haiku-20240307")
    
    # Instantiate TextChunker, as orchestrator expects an object with a .chunk_text() method
    # for the dynamic news chunking path.
    # The orchestrator will pass its own chunk_size and chunk_overlap to the method call.
    text_chunker_instance = TextChunker(max_chunk_size=200000, overlap=2000) # Defaults for instantiation

    cache_manager = CacheManager(cache_dir=os.path.join(temp_output_dir, ".cache"))
    metadata_generator = MetadataGenerator()
    output_writer = OutputWriter()

    # input_dir for orchestrator should be the directory containing the absolute_file_path
    input_file_dir = os.path.dirname(absolute_file_path)
    
    # Mock args for orchestrator if it expects them for other things (like theme)
    # For this test, we are primarily focused on the summarization flow.
    class MockArgs:
        def __init__(self):
            self.theme = None # Default theme
            self.max_words = 1000 # Default, might not be used for news file mode directly

    mock_cli_args = MockArgs()

    try:
        run_summarization(
            mode="file",  # Test 'file' mode, which handles news and SEC filings
            input_dir=input_file_dir, # Orchestrator will find the single file here
            output_dir=temp_output_dir,
            max_words=1000, # Placeholder, actual chunking for news is different
            cache_manager=cache_manager,
            chunker=text_chunker_instance, # Pass the TextChunker instance
            summarizer=llm_summarizer,
            meta_generator=metadata_generator,
            output_writer=output_writer,
            prompt_set_path=prompt_set_path, # Already passed to summarizer, but orchestrator might use it.
            args=mock_cli_args
        )
        logging.info(f"run_summarization completed for {absolute_file_path}")
        # Add assertions here if needed, e.g., check for output files
        # For now, focus is on observing logs for token limits.
        output_files = os.listdir(temp_output_dir)
        # Find the -level directory, e.g., "file-level"
        mode_level_dir_name = "file-level" # As 'mode' is "file"
        mode_level_path = os.path.join(temp_output_dir, mode_level_dir_name)
        
        assert os.path.exists(mode_level_path), f"Mode level output directory not created: {mode_level_path}"
        
        # The orchestrator creates subdirectories under mode_level_path based on rel_path_from_actual_input
        # For a single file test where input_dir = os.path.dirname(file_path),
        # rel_path_from_actual_input will be just os.path.basename(file_path).
        # So current_out_dir = os.path.join(mode_level_path, os.path.dirname(os.path.basename(file_path)))
        # Which simplifies to mode_level_path because dirname of a filename is empty.
        # Output files should be directly in mode_level_path.
        
        output_items_in_mode_dir = os.listdir(mode_level_path)
        logging.info(f"Files in output mode directory ({mode_level_path}): {output_items_in_mode_dir}")

        # Expect at least a .md and a .json file
        md_files_found = [f for f in output_items_in_mode_dir if f.endswith("_summary.md")]
        json_files_found = [f for f in output_items_in_mode_dir if f.endswith("_meta.json")]

        assert len(md_files_found) > 0, f"No summary MD file found in {mode_level_path}"
        assert len(json_files_found) > 0, f"No metadata JSON file found in {mode_level_path}"
        
        logging.info(f"Test passed for {absolute_file_path}. Output at {temp_output_dir}")

    except Exception as e:
        logging.error(f"Exception during test_summarize_single_file_claude for {absolute_file_path}: {e}", exc_info=True)
        pytest.fail(f"Summarization run failed for {absolute_file_path}: {e}")
    # finally:
        # Optional: Clean up the temporary directory
        # shutil.rmtree(temp_output_dir)
        # logging.info(f"Cleaned up temporary output directory: {temp_output_dir}")
        # For debugging, it's often better to leave it. 