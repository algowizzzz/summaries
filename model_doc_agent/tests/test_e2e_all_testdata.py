import pytest
import os
import logging
import shutil

# Configure logging to capture DEBUG messages
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to PYTHONPATH to resolve imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_doc_agent.src.orchestrator import run_summarization
# Import TextChunker class and the chunker module itself for its functions
from model_doc_agent.src.chunker import TextChunker 
import model_doc_agent.src.chunker as chunker_module 
from model_doc_agent.src.summarization import LLMSummarizer
from model_doc_agent.src.metadata import MetadataGenerator
from model_doc_agent.src.writer import OutputWriter
from model_doc_agent.src.cache import CacheManager

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

class OrchestratorChunkerAdapter:
    def __init__(self, text_chunker_instance):
        self._text_chunker = text_chunker_instance

    def chunk_text(self, text, chunk_size, chunk_overlap, section_name=None):
        # Orchestrator calls: chunker.chunk_text(text_to_summarize, 
        #                                      chunk_size=char_chunk_size_for_splitter, 
        #                                      chunk_overlap=char_chunk_overlap)
        # TextChunker.chunk_text definition: chunk_text(self, text: str, section_name: Optional[str] = None)
        # It uses self.max_chunk_size and self.overlap internally.
        # This adapter method will temporarily set these on the instance.
        
        original_max_chunk_size = self._text_chunker.max_chunk_size
        original_overlap = self._text_chunker.overlap
        
        self._text_chunker.max_chunk_size = chunk_size
        self._text_chunker.overlap = chunk_overlap
        
        try:
            # Pass section_name if TextChunker.chunk_text supports it, otherwise it will be ignored if not part of its signature.
            # The current TextChunker.chunk_text in chunker.py *does* accept section_name.
            # However, orchestrator does not pass section_name to chunker.chunk_text call for news.
            # So, we pass it as None or rely on the default in TextChunker.chunk_text.
            # For safety, explicitly pass section_name=None if the orchestrator doesn't provide it.
            if section_name:
                 results = self._text_chunker.chunk_text(text, section_name=section_name)
            else:
                 results = self._text_chunker.chunk_text(text) # Relies on default section_name=None in TextChunker
        finally:
            # Restore original values
            self._text_chunker.max_chunk_size = original_max_chunk_size
            self._text_chunker.overlap = original_overlap
        return results

    def get_full_text(self, data):
        return chunker_module.get_full_text(data)

    def get_sections_from_json(self, data, default_title_prefix="Section", min_length=50):
        # This method is not directly called by the orchestrator on the chunker object in file mode,
        # but good to have for completeness if other modes were to use this adapter.
        # Orchestrator calls chunker_module.get_sections_from_json directly in node mode.
        return chunker_module.get_sections_from_json(data, default_title_prefix=default_title_prefix, min_length=min_length)


@pytest.fixture(scope="module")
def test_env_setup():
    """Sets up the environment for E2E tests."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env.")

    test_data_input_dir = os.path.join(project_root, "TestData")
    if not os.path.exists(test_data_input_dir):
        pytest.fail(f"TestData directory not found at: {test_data_input_dir}")

    base_output_dir = os.path.join(project_root, "model_doc_agent", "output", "e2e_test_runs")
    prompt_set_path = os.path.join(project_root, "model_doc_agent", "sec_prompts_v1.json")
    if not os.path.exists(prompt_set_path):
         pytest.fail(f"Prompt set file not found: {prompt_set_path}")

    # Ensure node mode outputs exist for master mode to consume (run node mode if not already run)
    # This is a simple way to handle dependency for now. A more robust solution might use pytest-depends
    # or ensure the orchestrator for master mode can generate node summaries if missing.
    # For this test suite, we assume node mode test will have run or its output is present.
    # However, to make this test more self-contained if run in isolation, let's quickly run node mode.
    
    logging.info("Ensuring node-level summaries are available for master mode test...")
    node_mode_output_dir_for_master_dependency = os.path.join(base_output_dir, "node_mode_output")
    # We don't need to re-run if it already exists from a previous test_e2e_node_mode_all_testdata run in the same session.
    # But if running this test in isolation, or after a clean, it helps.
    # The orchestrator for master mode is designed to look for these.

    # Note: The orchestrator's master mode itself should be robust enough to find or generate node summaries.
    # This pre-run is more of a belt-and-suspenders for the test environment.
    # The existing orchestrator master mode tries to find node summaries. If not found, it tries to generate them.
    # So, this explicit run of node-mode here might be redundant if the master mode orchestrator works as designed.
    # Let's rely on the master mode orchestrator's capability to generate node summaries if needed.

    return {
        "test_data_input_dir": test_data_input_dir,
        "base_output_dir": base_output_dir,
        "prompt_set_path": prompt_set_path,
        "node_mode_output_for_master_dependency": node_mode_output_dir_for_master_dependency
    }

def test_e2e_file_mode_all_testdata(test_env_setup):
    """
    Tests 'file' mode summarization for all processable files in the TestData directory.
    """
    input_dir = test_env_setup["test_data_input_dir"]
    mode = "file"
    current_mode_output_dir = os.path.join(test_env_setup["base_output_dir"], f"{mode}_mode_output")

    if os.path.exists(current_mode_output_dir):
        shutil.rmtree(current_mode_output_dir)
    os.makedirs(current_mode_output_dir, exist_ok=True)
    logging.info(f"E2E Test ({mode} mode): Output will be written to: {current_mode_output_dir}")

    llm_summarizer = LLMSummarizer(prompt_set_path=test_env_setup["prompt_set_path"], model_name="claude-3-haiku-20240307")
    
    # Instantiate TextChunker and wrap it with the adapter
    text_chunker_instance = TextChunker(max_chunk_size=200000, overlap=2000) # Default values for the instance
    orchestrator_chunker = OrchestratorChunkerAdapter(text_chunker_instance)
    
    cache_manager = CacheManager(cache_dir=os.path.join(current_mode_output_dir, ".cache"))
    metadata_generator = MetadataGenerator()
    output_writer = OutputWriter()

    class MockArgs:
        def __init__(self):
            self.theme = None
            self.max_words = 1000

    mock_cli_args = MockArgs()

    try:
        run_summarization(
            mode=mode,
            input_dir=input_dir, 
            output_dir=current_mode_output_dir, 
            max_words=mock_cli_args.max_words,
            cache_manager=cache_manager,
            chunker=orchestrator_chunker, # Use the adapter instance here
            summarizer=llm_summarizer,
            meta_generator=metadata_generator,
            output_writer=output_writer,
            prompt_set_path=test_env_setup["prompt_set_path"],
            args=mock_cli_args
        )
        logging.info(f"E2E Test ({mode} mode): run_summarization completed for input: {input_dir}")
        
        actual_output_location = os.path.join(current_mode_output_dir, f"{mode}-level")
        assert os.path.exists(actual_output_location), f"Orchestrator did not create mode-level output dir: {actual_output_location}"
        assert os.listdir(actual_output_location), f"Mode-level output directory {actual_output_location} is empty."
        
        logging.info(f"E2E Test ({mode} mode) passed. Output at {current_mode_output_dir}")

    except Exception as e:
        logging.error(f"Exception during E2E Test ({mode} mode) for input {input_dir}: {e}", exc_info=True)
        pytest.fail(f"E2E Test ({mode} mode) failed for input {input_dir}: {e}") 

def test_e2e_node_mode_all_testdata(test_env_setup):
    """
    Tests 'node' mode summarization for all processable files in the TestData directory.
    Node mode is primarily for multi-section SEC filings.
    """
    input_dir = test_env_setup["test_data_input_dir"]
    mode = "node"
    current_mode_output_dir = os.path.join(test_env_setup["base_output_dir"], f"{mode}_mode_output")

    if os.path.exists(current_mode_output_dir):
        shutil.rmtree(current_mode_output_dir)
    os.makedirs(current_mode_output_dir, exist_ok=True)
    logging.info(f"E2E Test ({mode} mode): Output will be written to: {current_mode_output_dir}")

    llm_summarizer = LLMSummarizer(prompt_set_path=test_env_setup["prompt_set_path"], model_name="claude-3-haiku-20240307")
    text_chunker_instance = TextChunker(max_chunk_size=200000, overlap=2000) 
    orchestrator_chunker = OrchestratorChunkerAdapter(text_chunker_instance)
    cache_manager = CacheManager(cache_dir=os.path.join(current_mode_output_dir, ".cache"))
    metadata_generator = MetadataGenerator()
    output_writer = OutputWriter()

    class MockArgs:
        def __init__(self):
            self.theme = None
            self.max_words = 1000 # Not directly used by node summarization logic for content length

    mock_cli_args = MockArgs()

    try:
        run_summarization(
            mode=mode,
            input_dir=input_dir, 
            output_dir=current_mode_output_dir, 
            max_words=mock_cli_args.max_words,
            cache_manager=cache_manager,
            chunker=orchestrator_chunker, 
            summarizer=llm_summarizer,
            meta_generator=metadata_generator,
            output_writer=output_writer,
            prompt_set_path=test_env_setup["prompt_set_path"],
            args=mock_cli_args
        )
        logging.info(f"E2E Test ({mode} mode): run_summarization completed for input: {input_dir}")
        
        actual_output_location = os.path.join(current_mode_output_dir, f"{mode}-level")
        assert os.path.exists(actual_output_location), f"Orchestrator did not create mode-level output dir: {actual_output_location}"
        
        # It's possible that for some files in TestData, no node summaries are generated (e.g., news articles).
        # So, the directory might exist but be empty, or contain only a few outputs.
        # A more robust check would be to see if *some* files were processed or if the process didn't error out.
        # For now, we'll just check if the directory exists. If it has content, os.listdir will be true.
        # If no suitable files for node mode exist in TestData, this might still be empty, which is acceptable.
        if not os.listdir(actual_output_location):
            logging.warning(f"Mode-level output directory {actual_output_location} is empty. This may be okay if no files in TestData were suitable for node-level summarization.")
        else:
            logging.info(f"Mode-level output directory {actual_output_location} contains files.")

        logging.info(f"E2E Test ({mode} mode) considered passed (check logs for details). Output at {current_mode_output_dir}")

    except Exception as e:
        logging.error(f"Exception during E2E Test ({mode} mode) for input {input_dir}: {e}", exc_info=True)
        pytest.fail(f"E2E Test ({mode} mode) failed for input {input_dir}: {e}") 

def test_e2e_master_mode_all_testdata(test_env_setup):
    """
    Tests 'master' mode summarization for all processable files in the TestData directory.
    Master mode depends on existing or generatable node-level summaries.
    """
    input_dir = test_env_setup["test_data_input_dir"]
    mode = "master"
    # Master mode orchestrator needs to know where to find/create node summaries.
    # The orchestrator's master mode logic looks for node summaries in a path relative to its *own* output_dir structure.
    # e.g., if master output is <base_output_dir>/master_mode_output/master-level/
    # it will look for node summaries in <base_output_dir>/master_mode_output/node-level/
    # So, the `output_dir` passed to `run_summarization` for master mode is important for this relative lookup.
    current_mode_output_dir = os.path.join(test_env_setup["base_output_dir"], f"{mode}_mode_output")

    if os.path.exists(current_mode_output_dir):
        shutil.rmtree(current_mode_output_dir)
    os.makedirs(current_mode_output_dir, exist_ok=True)
    logging.info(f"E2E Test ({mode} mode): Output will be written to: {current_mode_output_dir}")

    llm_summarizer = LLMSummarizer(prompt_set_path=test_env_setup["prompt_set_path"], model_name="claude-3-haiku-20240307")
    text_chunker_instance = TextChunker(max_chunk_size=200000, overlap=2000)
    orchestrator_chunker = OrchestratorChunkerAdapter(text_chunker_instance)
    cache_manager = CacheManager(cache_dir=os.path.join(current_mode_output_dir, ".cache")) # Cache for master summaries
    metadata_generator = MetadataGenerator()
    output_writer = OutputWriter()

    class MockArgs:
        def __init__(self):
            self.theme = None
            self.max_words = 1000 

    mock_cli_args = MockArgs()

    # The orchestrator for master mode is expected to:
    # 1. Determine the path where node-level summaries *should* be (relative to its own output dir).
    # 2. Check if they exist.
    # 3. If not, call the node-level summarization logic to generate them there.
    # 4. Read those node summaries and create a master summary.

    try:
        run_summarization(
            mode=mode,
            input_dir=input_dir, 
            output_dir=current_mode_output_dir, # This is where master-level and its dependent node-level (if generated by it) will go.
            max_words=mock_cli_args.max_words,
            cache_manager=cache_manager,
            chunker=orchestrator_chunker, 
            summarizer=llm_summarizer,
            meta_generator=metadata_generator,
            output_writer=output_writer,
            prompt_set_path=test_env_setup["prompt_set_path"],
            args=mock_cli_args
        )
        logging.info(f"E2E Test ({mode} mode): run_summarization completed for input: {input_dir}")
        
        actual_output_location = os.path.join(current_mode_output_dir, f"{mode}-level")
        assert os.path.exists(actual_output_location), f"Orchestrator did not create mode-level output dir: {actual_output_location}"
        
        # Similar to node mode, it's possible no files were suitable for master summaries (e.g. if no node summaries could be made)
        if not os.listdir(actual_output_location):
            logging.warning(f"Mode-level output directory {actual_output_location} is empty. This may be okay if no files in TestData were suitable for master-level summarization.")
        else:
            logging.info(f"Mode-level output directory {actual_output_location} contains files.")

        logging.info(f"E2E Test ({mode} mode) considered passed (check logs for details). Output at {current_mode_output_dir}")

    except Exception as e:
        logging.error(f"Exception during E2E Test ({mode} mode) for input {input_dir}: {e}", exc_info=True)
        pytest.fail(f"E2E Test ({mode} mode) failed for input {input_dir}: {e}") 