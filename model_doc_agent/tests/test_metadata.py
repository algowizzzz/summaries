import pytest
import json
import os
from datetime import datetime
from model_doc_agent.src.metadata import extract_metadata # Updated import

# Sample data for testing
SAMPLE_NEWS_DATA = {
    "id": "news-sample-001",
    "title": "Test News Article Title",
    "date": "2024-05-21",
    "source_type": "web",
    "source_name": "Test News Online",
    "url": "http://example.com/test-news",
    "content": "This is the content of the test news article. It mentions several important things."
}

SAMPLE_SEC_DATA = {
    "header": "Form 10-K Example Company",
    "form_type": "10-K",
    "sections": {
        "item1": "Business description here.",
        "item1a": "Risk factors detailed here."
    }
}

@pytest.fixture
def sample_news_json_data():
    return SAMPLE_NEWS_DATA.copy()

@pytest.fixture
def sample_sec_json_data():
    return SAMPLE_SEC_DATA.copy()

# Test for News Article Metadata Extraction
def test_extract_metadata_news_article(sample_news_json_data):
    original_filename = "sample_news.json"
    file_path = f"/test/data/news/{original_filename}"
    summary_mode = "file" # or 'news' mode if that's how orchestrator sets it
    effective_filing_type = "news"
    document_title_from_data = sample_news_json_data["title"]

    metadata = extract_metadata(
        data=sample_news_json_data,
        original_filename=original_filename,
        file_path=file_path,
        summary_mode=summary_mode,
        effective_filing_type=effective_filing_type,
        document_title=document_title_from_data
    )

    assert metadata["original_filename"] == original_filename
    assert metadata["file_path"] == file_path
    assert metadata["summary_type"] == summary_mode
    assert metadata["document_title"] == document_title_from_data
    
    doc_meta = metadata["document_metadata"]
    assert doc_meta["effective_filing_type"] == "news"
    assert doc_meta["source_type"] == sample_news_json_data["source_type"]
    assert doc_meta["source_name"] == sample_news_json_data["source_name"]
    assert doc_meta["date"] == sample_news_json_data["date"]
    assert doc_meta["url"] == sample_news_json_data["url"]
    assert doc_meta["original_news_id"] == sample_news_json_data["id"]
    assert "processed_timestamp" in doc_meta
    # Ensure content (and other original fields) are now part of document_metadata
    assert "content" in doc_meta
    assert doc_meta["content"] == sample_news_json_data["content"]
    # Check another original field to be sure
    assert "title" in doc_meta # The original title from the news data itself
    assert doc_meta["title"] == sample_news_json_data["title"]

# Test for SEC Filing Metadata Extraction
def test_extract_metadata_sec_filing(sample_sec_json_data):
    original_filename = "test_10k.json"
    file_path = f"/test/data/sec/{original_filename}"
    summary_mode = "file"
    effective_filing_type = "10-K"
    document_title_from_header = sample_sec_json_data["header"]

    metadata = extract_metadata(
        data=sample_sec_json_data,
        original_filename=original_filename,
        file_path=file_path,
        summary_mode=summary_mode,
        effective_filing_type=effective_filing_type,
        document_title=document_title_from_header
    )

    assert metadata["original_filename"] == original_filename
    assert metadata["file_path"] == file_path
    assert metadata["summary_type"] == summary_mode
    assert metadata["document_title"] == document_title_from_header
    
    doc_meta = metadata["document_metadata"]
    assert doc_meta["effective_filing_type"] == "10-K"
    # document_header might be removed if same as document_title, so check presence or value
    assert doc_meta.get("document_header") is None or doc_meta.get("document_header") == document_title_from_header
    assert "processed_timestamp" in doc_meta
    # Check that news-specific fields are not present
    assert "source_type" not in doc_meta
    assert "news_url" not in doc_meta # Check for a field that would have a prefix if it was old style
    assert "url" not in doc_meta # Check new style news field name shouldn't be here

# Test with minimal data and fallbacks
def test_extract_metadata_minimal_data(tmp_path):
    expected_basename = "minimal.json"
    minimal_file_path = tmp_path / expected_basename
    minimal_file_path.write_text(json.dumps({"some_key": "some_value"}))
    
    input_original_filename = os.path.basename(minimal_file_path)
    assert input_original_filename == expected_basename # Verify the input to the function

    extracted_metadata = extract_metadata(
        data=json.loads(minimal_file_path.read_text()), # Ensure data is read consistently
        original_filename=input_original_filename,
        file_path=str(minimal_file_path),
        summary_mode="file"
    )
    
    assert "original_filename" in extracted_metadata, "'original_filename' key missing from metadata"
    assert extracted_metadata["original_filename"] == expected_basename, \
        f"Expected original_filename to be '{expected_basename}' but got '{extracted_metadata["original_filename"]}'"
    
    assert extracted_metadata["file_path"] == str(minimal_file_path)
    assert extracted_metadata["summary_type"] == "file"
    assert "document_metadata" in extracted_metadata
    assert extracted_metadata["document_metadata"]["some_key"] == "some_value"
    assert extracted_metadata["document_title"] == expected_basename # Default title fallback

# Test additional context being passed through
def test_extract_metadata_with_additional_context(sample_news_json_data):
    theme_context = {"theme": "Market Analysis"}
    metadata = extract_metadata(
        data=sample_news_json_data,
        original_filename="news_with_theme.json",
        file_path="/path/to/news_with_theme.json",
        summary_mode="file",
        effective_filing_type="news",
        document_title=sample_news_json_data["title"],
        additional_context=theme_context
    )
    doc_meta = metadata["document_metadata"]
    assert doc_meta["theme"] == "Market Analysis"

if __name__ == "__main__":
    pytest.main() 