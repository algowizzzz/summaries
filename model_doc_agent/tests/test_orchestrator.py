# tests/test_orchestrator.py
import os
import json
import tempfile
import pytest
from model_doc_agent.src.orchestrator import run_summarization
from unittest.mock import MagicMock, patch, call, ANY
from model_doc_agent.src import orchestrator
from model_doc_agent.src.cache import CacheManager # Real CacheManager for some tests
import re

class DummyCache:
    def __init__(self):
        self.seen = set()
    def hash_content(self, c): return c
    def is_cached(self, h): return False
    def mark_cached(self, h): self.seen.add(h)

class DummyChunker:
    def __init__(self):
        pass
    def chunk_json(self, data, max_words):
        return [("1", data["text"], {})]
    def extract_sections(self, data):
        return [("1", "Title", data["text"])]
    def get_full_text(self, data):
        return data["text"]

class DummySumm:
    def __init__(self):
        self.calls = []
        self.model_name = "dummy_model"
    def summarize(self, mode, **kwargs):
        self.calls.append((mode, kwargs))
        return f"{mode.upper()}_SUMMARY"

class DummyMeta:
    def generate_metadata(self, *args, **kwargs):
        return {"meta": "ok"}

class DummyWriter:
    def __init__(self):
        self.summaries = []
        self.metadata = []
    def write_summary(self, text, path):
        self.summaries.append((path, text))
    def write_metadata(self, meta, path):
        self.metadata.append((path, meta))

@pytest.mark.parametrize("mode,expected_suffix", [
    ("file", "_summary.md"),
    ("node", "_1_summary.md"),
    ("master", "_master_summary.md"),
])
def test_orchestrator_basic_modes(tmp_path, mode, expected_suffix):
    # Setup input JSON file
    inp = tmp_path/"in"
    out = tmp_path/"out"
    inp.mkdir()
    out.mkdir()
    data = {"text": "hello world"}
    src_file = inp/"test.json"
    src_file.write_text(json.dumps(data))
    # Dummy components
    cache = DummyCache()
    chunker = DummyChunker()
    summ = DummySumm()
    meta = DummyMeta()
    writer = DummyWriter()
    # Run
    run_summarization(mode, str(inp), str(out), max_words=10,
                      cache_manager=cache,
                      chunker=chunker,
                      summarizer=summ,
                      meta_generator=meta,
                      output_writer=writer,
                      prompt_set_path=None) # No need for prompt set with dummy summarizer

    # Assertions (example, expand as needed)
    assert len(writer.summaries) > 0
    assert writer.summaries[0][1] == f"{mode.upper()}_SUMMARY"
    # Check output path construction based on mode
    # This is a simplified check; real paths are more complex
    # Example: if mode == 'file', path might be 'out/test/test_1_summary.md'
    # if mode == 'node', path might be 'out/test/node_summary.md'
    # Placeholder for more specific path assertions
    assert expected_suffix in writer.summaries[0][0] 

# --- Fixtures --- #

@pytest.fixture
def mock_args(tmp_path):
    "Fixture for mock command line arguments."
    args = MagicMock()
    args.mode = "file"
    args.input_dir = str(tmp_path / "input")
    args.output_dir = str(tmp_path / "output")
    args.max_words = 1000
    args.no_cache = False
    args.prompt_set_path = "dummy_prompts.json"
    args.theme = None # Default, can be overridden in tests
    args.verbose = False
    # Create dummy input and output dirs for the mock args
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    return args

@pytest.fixture
def mock_cache_manager():
    "Mock CacheManager."
    manager = MagicMock(spec=CacheManager)
    manager.is_cached.return_value = False # Default: not cached
    manager.hash_content.side_effect = lambda x: f"hash_of_{x[:10]}" # Simple hash mock
    return manager

@pytest.fixture
def mock_chunker_module():
    "Mock the entire chunker module passed to orchestrator."
    # The orchestrator uses chunker.get_sections_from_json and chunker.TextChunker
    # Let's mock these specific functions/classes if orchestrator calls them directly as module.function
    # Based on previous orchestrator, it seems to expect chunker objects or module-level functions.
    # For this test, assuming orchestrator expects a module-like object or direct function calls.
    
    mock_module = MagicMock()
    # Mocking get_sections_from_json (used by updated orchestrator for news/flat JSONs)
    mock_module.get_sections_from_json.return_value = [("section1_title", "Section 1 content")] 
    # Mocking TextChunker class behavior if orchestrator instantiates it.
    # If orchestrator expects an already instantiated TextChunker, this fixture needs adjustment.
    # The current orchestrator.py doesn't directly instantiate TextChunker, but uses get_sections_from_json from it.
    # And it passes the `chunker` module itself to `run_summarization` which then calls `chunker.get_full_text` etc.
    # So we need to mock those functions on the module object.

    mock_module.get_full_text.return_value = "Full document text."
    mock_module.extract_sections.return_value = [("s1", "Section 1 Title", "Section 1 content.")]

    # Mock TextChunker instance that might be created by orchestrator or passed in
    mock_text_chunker_instance = MagicMock()
    mock_text_chunker_instance.chunk_document.return_value = [
        {"content": "Chunk 1 content", "section_name": "Section 1", "chunk_index": 0}
    ]
    mock_module.TextChunker.return_value = mock_text_chunker_instance
    return mock_module

@pytest.fixture
def mock_summarizer():
    "Mock LLMSummarizer or the new Summarizer."
    summarizer = MagicMock()
    # Based on Summarizer.summarize_document and Summarizer.summarize_chunk
    summarizer.summarize.return_value = "Mocked document summary."
    # summarizer.summarize_chunk.return_value = "Mocked chunk summary." # Not strictly needed if summarize covers all
    summarizer.model_name = "mock_gpt_test_fixture"
    return summarizer

@pytest.fixture
def mock_metadata_generator():
    "Mock MetadataGenerator or the new extract_metadata function wrapper."
    # The new orchestrator uses extract_metadata directly.
    # So we should patch 'model_doc_agent.src.orchestrator.extract_metadata' 
    # or pass a mock function if orchestrator takes it as an argument.
    # For simplicity, let's assume orchestrator might take a generator object with a method.
    # Or, if it calls a module function, that needs to be patched.
    # The current orchestrator.py uses `meta_generator.generate_metadata` (class based)
    # but the user has been moving towards functional components. Let's assume the orchestrator
    # has been updated to use the new `extract_metadata` function. 
    # No, the provided orchestrator still has `meta_generator.generate_metadata` from `codebase.md` structure.
    # And `summarizer.py` (CLI) instantiates `metadata.MetadataGenerator()`
    # So we mock the class instance's method.
    meta_gen = MagicMock()
    meta_gen.generate_metadata.return_value = {"summary_id": "meta123", "processed_timestamp": "sometime"}
    return meta_gen

@pytest.fixture
def mock_output_writer():
    "Mock OutputWriter."
    writer = MagicMock()
    return writer

# --- Helper Function to Create Dummy Files --- #

def create_dummy_json_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)

# --- Orchestrator Tests --- #

SAMPLE_NEWS_CONTENT_FOR_ORCH = {
    "id": "orch-news-001", "title": "Orchestrator News Test", "content": "News content for orchestrator.",
    "date": "2024-05-23", "source_type": "test", "source_name": "pytest channel"
}
SAMPLE_SEC_CONTENT_FOR_ORCH = {"form_type": "10-K", "sections": {"item1": "SEC content here."}}

def test_orchestrator_run_summarization_news_file_mode(
    mock_args, mock_cache_manager, mock_chunker_module, mock_summarizer, mock_metadata_generator, mock_output_writer, tmp_path):
    
    # Setup: Create a dummy news file in a nested structure
    input_dir = tmp_path / "input_data" / "news" / "test_source" / "2024-05"
    input_dir.mkdir(parents=True, exist_ok=True)
    news_file_path = input_dir / "test_news_article.json"
    create_dummy_json_file(news_file_path, SAMPLE_NEWS_CONTENT_FOR_ORCH)
    
    mock_args.input_dir = str(input_dir) # Ensure input_dir in args is the specific one for this test
    # mock_args.mode is already "file" by default from fixture

    # Configure mocks
    # mock_summarizer is already an instance from the fixture
    mock_summarizer.summarize.return_value = "News summary by orchestrator." # Configure the summarize method
    # mock_summarizer.model_name is already set by its fixture if needed by meta_generator

    # mock_metadata_generator is already an instance from the fixture.
    # MockExtractMetadata is for when 'extract_metadata' is called directly.
    # The orchestrator uses meta_generator.generate_metadata.
    mock_metadata_generator.generate_metadata.return_value = {"summary_id": "news_meta_123", "title": "Orchestrator News Test"}
    
    # Mock the chunker module's get_full_text method for news processing (used if data.get("content") is empty)
    # and get_sections_from_json which is used for news to extract title/content
    mock_chunker_module.get_sections_from_json.return_value = [ (SAMPLE_NEWS_CONTENT_FOR_ORCH["title"], SAMPLE_NEWS_CONTENT_FOR_ORCH["content"]) ]
    mock_chunker_module.get_full_text.return_value = SAMPLE_NEWS_CONTENT_FOR_ORCH["content"]


    # Action
    orchestrator.run_summarization(
        mode=mock_args.mode,
        input_dir=mock_args.input_dir,
        output_dir=mock_args.output_dir,
        max_words=mock_args.max_words,
        cache_manager=mock_cache_manager,
        chunker=mock_chunker_module,  # Pass the mock chunker module
        summarizer=mock_summarizer,    # Pass the mock summarizer instance
        meta_generator=mock_metadata_generator, # Pass the mock metadata_generator instance
        output_writer=mock_output_writer,
        prompt_set_path=mock_args.prompt_set_path,
        args=mock_args # Pass the full args object for other params like theme
    )

    # Assertions
    # 1. Cache check and content hashing for news content
    # Orchestrator gets content (data.get("content")) and title (data.get("title")) for news.
    # It hashes text_to_summarize + json.dumps(summarizer_kwargs)
    # summarizer_kwargs for news will be {'title': news_title} at the point of hashing.
    # The 'content' key is added to summarizer_kwargs *after* this hash is computed.
    
    expected_text_to_summarize = SAMPLE_NEWS_CONTENT_FOR_ORCH["content"]
    expected_summarizer_kwargs_for_hash = {'title': SAMPLE_NEWS_CONTENT_FOR_ORCH["title"]}
    
    expected_cache_input = expected_text_to_summarize + json.dumps(expected_summarizer_kwargs_for_hash)
    mock_cache_manager.hash_content.assert_any_call(expected_cache_input)
    mock_cache_manager.is_cached.assert_any_call(f"hash_of_{expected_cache_input[:10]}")

    # 2. Summarizer called correctly for news
    # For news, orchestrator calls summarizer.summarize(mode="news", filing_type="news", title=..., content=...)
    mock_summarizer.summarize.assert_called_once_with(
        mode="news", # actual_mode_for_summarizer becomes "news"
        filing_type="news",
        title=SAMPLE_NEWS_CONTENT_FOR_ORCH["title"],
        content=SAMPLE_NEWS_CONTENT_FOR_ORCH["content"]
    )

    # 3. Metadata generation for news
    # meta_generator.generate_metadata is called
    mock_metadata_generator.generate_metadata.assert_called_once_with(
        summary_text="News summary by orchestrator.",
        source_path=str(news_file_path),
        mode="news", # actual_mode_for_summarizer
        is_news=True,
        news_data=SAMPLE_NEWS_CONTENT_FOR_ORCH,
        doc_type=SAMPLE_NEWS_CONTENT_FOR_ORCH.get("article_type"), # or "news" if not present
        llm_model_used=mock_summarizer.model_name
    )

    # 4. Output writer called for MD and JSON
    # sanitized_output_base = re.sub(r'[^\\w\\-_.]', '_', SAMPLE_NEWS_CONTENT_FOR_ORCH["id"]) # Re-bypassing, dynamic version causes mock issues
    hardcoded_sanitized_id = "orch-news-001" # Orchestrator's actual behavior matches this.
    
    # Construct expected output paths based on orchestrator logic
    expected_md_path = os.path.join(mock_args.output_dir, "file-level", f"{hardcoded_sanitized_id}_summary.md")
    expected_json_path = os.path.join(mock_args.output_dir, "file-level", f"{hardcoded_sanitized_id}_meta.json")
    
    mock_output_writer.write_summary.assert_called_with("News summary by orchestrator.", expected_md_path)
    mock_output_writer.write_metadata.assert_called_with(mock_metadata_generator.generate_metadata.return_value, expected_json_path)

# TODO: Add more tests for SEC filings, different modes (node, master), caching behavior, error handling etc.
# This initial test focuses on the news file processing path within the orchestrator.

# if __name__ == "__main__":
#     pytest.main(["-v", __file__]) # Commenting out to avoid execution if file is run directly during testing