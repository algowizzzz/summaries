# tests/test_writer.py
import os
import json
import tempfile
from model_doc_agent.src.writer import OutputWriter

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