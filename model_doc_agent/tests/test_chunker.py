import pytest
import json
import os
from model_doc_agent.src.chunker import TextChunker, get_sections_from_json # Assuming this is the location

# Sample data for testing get_sections_from_json
NEWS_ARTICLE_DATA = {
    "id": "news001",
    "title": "Big Tech Announces Breakthrough",
    "date": "2024-01-15",
    "source_name": "Tech Times",
    "content": "San Francisco, CA - Today, a major tech company, InnovateCorp, announced a significant breakthrough in quantum computing. Their new processor, the 'Quasar', is said to be 1000 times faster than current supercomputers. This could revolutionize fields like medicine, materials science, and financial modeling. The announcement was made by CEO Dr. Aris Thorne at a packed press conference. Details about commercial availability are expected next quarter. This is a single paragraph for simplicity."
}

SEC_FILING_DATA_SECTIONS = {
    "form_type": "10-K",
    "company_name": "Example Corp",
    "sections": {
        "Item 1": "This is the business description.",
        "Item 1A": "These are the risk factors. There are many risks.",
        "Item 7": "Management discussion and analysis. We performed well."
    }
}

FLAT_JSON_DATA = {
    "report_title": "Quarterly Sales Report",
    "region": "North America",
    "sales_total": 500000,
    "notes": "Strong performance in Q3 despite market challenges."
}

JSON_LIST_DATA = [
    {"item_id": "A1", "description": "This is item A1 from the list.", "status": "active"},
    {"item_id": "B2", "description": "Item B2 details are here.", "status": "pending"}
]

@pytest.fixture
def sample_news_data():
    return NEWS_ARTICLE_DATA.copy()

@pytest.fixture
def sample_sec_data_sections():
    return SEC_FILING_DATA_SECTIONS.copy()

@pytest.fixture
def sample_flat_json_data():
    return FLAT_JSON_DATA.copy()

@pytest.fixture
def sample_json_list_data():
    return JSON_LIST_DATA.copy()

@pytest.fixture
def text_chunker_default():
    return TextChunker(max_chunk_size=100, overlap=20)


# Tests for get_sections_from_json

def test_get_sections_news_article(sample_news_data):
    sections = get_sections_from_json(sample_news_data)
    assert len(sections) == 1
    title, content = sections[0]
    assert title == sample_news_data["title"]
    assert content == sample_news_data["content"]

def test_get_sections_sec_filing(sample_sec_data_sections):
    sections = get_sections_from_json(sample_sec_data_sections)
    assert len(sections) == 3
    expected_sections = list(sample_sec_data_sections["sections"].items())
    for i, (name, content) in enumerate(sections):
        assert name == expected_sections[i][0]
        assert content == expected_sections[i][1]

def test_get_sections_flat_json(sample_flat_json_data):
    sections = get_sections_from_json(sample_flat_json_data)
    assert len(sections) == 1
    # Chunker defaults to "document_content" if no clear title key like "title" is found.
    assert sections[0][0] == "document_content" 
    # Check if the content contains key elements from the flat JSON
    for key, value in sample_flat_json_data.items():
        assert f"{key}: {value}" in sections[0][1]

def test_get_sections_json_list(sample_json_list_data):
    sections = get_sections_from_json(sample_json_list_data)
    assert len(sections) == len(sample_json_list_data)
    
    for i, item_data in enumerate(sample_json_list_data):
        expected_title = f"item_{i+1}" # Expecting generic titles like "item_1", "item_2"
        assert sections[i][0] == expected_title
        # Ensure all fields from the original list item are in its stringified content
        # sections[i][1] should be a JSON string representation of item_data
        parsed_content_dict = json.loads(sections[i][1])
        assert parsed_content_dict == item_data # Direct dictionary comparison

def test_get_sections_empty_json():
    sections = get_sections_from_json({})
    assert len(sections) == 0

def test_get_sections_min_length(sample_news_data):
    short_content_data = sample_news_data.copy()
    short_content_data["content"] = "Too short."
    sections = get_sections_from_json(short_content_data, min_length=50)
    assert len(sections) == 0
    sections_ok = get_sections_from_json(short_content_data, min_length=5)
    assert len(sections_ok) == 1

# Tests for TextChunker class

TEST_DOCUMENT_PATH = "dummy_test_file.json" # For chunk_document path argument

def test_chunk_text_simple(text_chunker_default):
    text = "This is a test sentence. This is another test sentence for chunking."
    chunks = text_chunker_default.chunk_text(text, section_name="TestSection1")
    assert len(chunks) > 0
    assert chunks[0]["content"] is not None
    assert chunks[0]["section_name"] == "TestSection1"
    assert chunks[0]["chunk_index"] == 0

def test_chunk_text_overlap(text_chunker_default):
    text = "a" * 150 # chunk_size=100, overlap=20
    # Expected: 
    # Chunk 0: text[0:100]
    # Chunk 1: text[80:180] -> text[80:150]
    chunks = text_chunker_default.chunk_text(text, section_name="OverlapTest")
    assert len(chunks) == 2
    assert chunks[0]["content"] == "a" * 100
    assert chunks[1]["content"] == "a" * 70 # 150 - 80 = 70
    assert chunks[0]["content"][80:] == chunks[1]["content"][:20] # Check overlap content

def test_chunk_document_news(text_chunker_default, sample_news_data):
    # get_sections_from_json will extract title and content for news
    sections = get_sections_from_json(sample_news_data)
    assert len(sections) == 1
    # section_name from get_sections will be the news article's title
    assert sections[0][0] == sample_news_data["title"]

    all_chunks = text_chunker_default.chunk_document(TEST_DOCUMENT_PATH, sections)
    assert len(all_chunks) > 0
    for chunk in all_chunks:
        assert chunk["section_name"] == sample_news_data["title"] # News title as section name
        assert sample_news_data["content"].startswith(chunk["content"][:50]) or chunk["content"] in sample_news_data["content"]

def test_chunk_document_sec_filing(text_chunker_default, sample_sec_data_sections):
    sections = get_sections_from_json(sample_sec_data_sections)
    assert len(sections) == 3
    all_chunks = text_chunker_default.chunk_document(TEST_DOCUMENT_PATH, sections)
    
    total_expected_chunks = 0
    for sec_name, sec_content in sample_sec_data_sections["sections"].items():
        # Manual calculation based on chunker settings
        text_len = len(sec_content)
        cs = text_chunker_default.max_chunk_size #100
        ov = text_chunker_default.overlap #20
        if text_len == 0: continue
        num_chunks_for_sec = 1
        remaining_len = text_len - cs
        while remaining_len > 0:
            num_chunks_for_sec +=1
            remaining_len -= (cs - ov)
        total_expected_chunks += num_chunks_for_sec
        
        # Check if chunks from this section exist
        assert any(c["section_name"] == sec_name for c in all_chunks)

    # This assertion might be fragile if chunking logic has subtle edge cases not perfectly mirrored here.
    # The main point is that chunks are generated for all sections.
    # For this test data, with max_chunk_size=100:
    # Item 1 (30 chars) -> 1 chunk
    # Item 1A (48 chars) -> 1 chunk
    # Item 7 (47 chars) -> 1 chunk
    # Total = 3 chunks
    assert len(all_chunks) == 3 


if __name__ == "__main__":
    pytest.main() 