Below are Pytest-style test scripts covering each core module. Place them under a `tests/` directory alongside your `src/` package.

---

```python
# tests/test_cache.py
import os
import tempfile
import pytest
from src.cache import CacheManager

def test_hash_and_cache(tmp_path):
    cm = CacheManager(cache_dir=str(tmp_path/"cache"), enabled=True)
    content = "some important text"
    h = cm.hash_content(content)
    # Initially not cached
    assert not cm.is_cached(h)
    # Mark cached and check
    cm.mark_cached(h)
    assert cm.is_cached(h)
    # Cache file exists
    assert os.path.exists(tmp_path/"cache"/f"{h}.cache")

def test_no_cache_mode(tmp_path):
    cm = CacheManager(cache_dir=str(tmp_path/"cache"), enabled=False)
    content = "text"
    h = cm.hash_content(content)
    # In no-cache mode, is_cached always False and mark_cached does nothing
    assert not cm.is_cached(h)
    cm.mark_cached(h)
    assert not cm.is_cached(h)
    # Cache directory shouldn’t be created
    assert not os.path.exists(tmp_path/"cache")
```

---

```python
# tests/test_chunker.py
import json
import pytest
from src.chunker import chunk_json, extract_sections, get_full_text

SAMPLE_JSON = {
    "sections": [
        {"title": "Sec A", "content": "word " * 10},
        {"title": "Sec B", "content": "word " * 3000},  # will be split
    ]
}

def test_extract_sections_and_full_text():
    secs = extract_sections(SAMPLE_JSON)
    assert len(secs) == 2
    ids = [s[0] for s in secs]
    titles = [s[1] for s in secs]
    assert ids == ["1", "2"]
    assert titles == ["Sec A", "Sec B"]
    full = get_full_text(SAMPLE_JSON)
    # Full text contains both sections
    assert "Sec A" not in full  # sections() only returns content
    assert len(full) > 0

def test_chunk_json_respects_max_words(tmp_path):
    # For a small max_words, the second section will be split
    chunks = chunk_json(SAMPLE_JSON, max_words=50)
    # First section under limit → 1 chunk; second section splits into multiple
    assert chunks[0][0] == "1"  # first chunk id
    assert all(len(c[1].split()) <= 50 for c in chunks)
    # Metadata includes section_title
    assert all("section_title" in c[2] for c in chunks)

def test_chunk_json_fallback_non_structured():
    simple = {"key": "value"}
    chunks = chunk_json(simple, max_words=5)
    # Entire JSON is split by word count
    assert all(isinstance(c[0], str) for c in chunks)
    assert sum(len(c[1].split()) for c in chunks) >= 1
```

---

```python
# tests/test_summarization.py
import json
import pytest
from src.summarization import LLMSummarizer

def test_load_prompt_config_and_invalid_mode(tmp_path):
    cfg = {
        "test": {
            "template": "Hello, {name}!",
            "input_variables": ["name"]
        }
    }
    cfg_path = tmp_path/"pset.json"
    cfg_path.write_text(json.dumps(cfg))
    summarizer = LLMSummarizer(prompt_set_path=str(cfg_path), max_retries=1)
    # Unknown mode should raise
    with pytest.raises(ValueError):
        summarizer.summarize("invalid", name="World")

def test_prompt_template_run(monkeypatch, tmp_path):
    # Create a prompt-set that echos name
    cfg = {
        "echo": {
            "template": "{name}",
            "input_variables": ["name"]
        }
    }
    cfg_path = tmp_path/"pset.json"
    cfg_path.write_text(json.dumps(cfg))
    summarizer = LLMSummarizer(prompt_set_path=str(cfg_path), max_retries=1)
    # Monkey-patch the LLMChain.run to just return the template filled
    class DummyChain:
        def __init__(self, **kwargs): pass
        def run(self, inputs):
            return inputs["name"].upper()
    monkeypatch.setattr("src.summarization.PromptTemplate", lambda input_variables, template: template)
    monkeypatch.setattr("src.summarization.LLMChain", lambda llm, prompt: DummyChain())
    result = summarizer.summarize("echo", name="world")
    assert result == "WORLD"
```

---

```python
# tests/test_metadata.py
import re
import pytest
from src.metadata import MetadataGenerator

def test_generate_metadata_basic():
    mg = MetadataGenerator()
    text = "OpenAI builds powerful AI models."
    meta = mg.generate_metadata(summary_text=text,
                                source_path="/path/doc.json",
                                mode="node",
                                section_id="1",
                                section_title="Intro")
    # Check presence and types
    assert re.match(r"[0-9a-f\-]{36}", meta["summary_id"])
    assert meta["source_path"].endswith("doc.json")
    assert meta["mode"] == "node"
    assert meta["section_id"] == "1"
    assert meta["section_title"] == "Intro"
    assert "timestamp" in meta
    assert isinstance(meta["word_count"], int) and meta["word_count"] > 0
    assert isinstance(meta["key_terms"], list)
    assert isinstance(meta["entities"], list)
```

---

```python
# tests/test_writer.py
import os
import json
import tempfile
from src.writer import OutputWriter

def test_write_summary_and_metadata(tmp_path):
    writer = OutputWriter()
    # Prepare paths
    md_path = tmp_path/"out"/"sum.md"
    meta_path = tmp_path/"out"/"sum_meta.json"
    summary_text = "This is a test summary."
    metadata = {"foo": "bar"}
    # Write
    writer.write_summary(summary_text, str(md_path))
    writer.write_metadata(metadata, str(meta_path))
    # Verify summary file
    assert md_path.exists()
    assert md_path.read_text() == summary_text
    # Verify metadata file
    assert meta_path.exists()
    loaded = json.loads(meta_path.read_text())
    assert loaded == metadata
```

---

```python
# tests/test_orchestrator.py
import os
import json
import tempfile
import pytest
from src.orchestrator import run_summarization

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
                      output_writer=writer)
    # Verify writer called
    assert writer.summaries, "No summary written"
    # Path ends with expected suffix in output dir
    path, text = writer.summaries[0]
    assert path.startswith(str(out))
    assert path.endswith(expected_suffix)
    assert "SUMMARY" in text
    # Metadata also written
    meta_path, metadata = writer.metadata[0]
    assert meta_path.endswith(expected_suffix.replace(".md", "_meta.json"))
    assert metadata == {"meta": "ok"}
```

---

**Run all tests** with:

```bash
pytest --maxfail=1 --disable-warnings -q
```

These tests cover:

* **CacheManager**: hashing, caching behavior, no-cache mode
* **Chunker**: section extraction, chunk-splitting by size, full-text fallback
* **Summarizer**: prompt loading, invalid mode error, basic template execution (with monkeypatch)
* **MetadataGenerator**: presence and basic structure of metadata fields
* **OutputWriter**: file creation and content correctness
* **Orchestrator**: integration flow for `file`, `node`, and `master` modes with dummy components

You can extend similar patterns to test **cross** mode, CLI argument parsing, and edge cases as needed.
