# tests/test_summarization.py
import pytest
import os
import json
from unittest.mock import patch, MagicMock
from model_doc_agent.src.summarization import LLMSummarizer as Summarizer, PromptLoader # Updated import path

@pytest.fixture
def summarizer_instance(setup_dummy_templates_and_config):
    config_path, templates_dir = setup_dummy_templates_and_config
    # Patch LLMChain.run to avoid actual API calls during tests
    with patch('langchain.chains.LLMChain.run') as mock_llm_run: 
        def side_effect_llm_run(inputs):
            # Consolidate title extraction for mock clarity
            title_from_inputs = inputs.get('title', 
                                    inputs.get('document_title', 
                                    inputs.get('section_name', 
                                    inputs.get('source_filename', 'UnknownTitle'))))
            content_from_inputs = str(inputs.get('text', inputs.get('content', '')))[:20]
            user_prompt_simulation = f"User prompt with title: '{title_from_inputs}' and content: '{content_from_inputs}...'"
            return f"Mock Summary for {title_from_inputs}. {user_prompt_simulation}"
        mock_llm_run.side_effect = side_effect_llm_run
        
        summarizer = Summarizer(prompt_config_path=config_path, templates_dir=templates_dir)
        summarizer.mock_llm_call = mock_llm_run # Attach mock for assertions
        yield summarizer

# --- Tests for Summarizer --- #

def test_summarize_document_news(summarizer_instance):
    content = "This is a news article about a recent event."
    title = "Big News Today"
    # The LLMSummarizer.summarize method calls self.get_prompt_template, then chain.run(kwargs)
    # kwargs sent to chain.run will be what the prompt template expects.
    # For default_user_file.txt: "User: Summarize document '{title}': {text}"
    # For news_user_file.txt: "User: Summarize news '{title}': {text}"
    # LLMSummarizer.summarize passes 'content' as 'text' to the prompt if mode is 'file' or 'node'.
    # So, inputs to mock_llm_run will be like {'title': 'Big News Today', 'text': 'This is a news...'}
    summary = summarizer_instance.summarize(mode="file", filing_type="news", title=title, content=content) # Changed from summarize_document
    assert "Mock Summary for Big News Today" in summary 
    assert "User prompt with title: 'Big News Today'" in summary
    assert "content: 'This is a news artic..." in summary # Check content part of mock
    summarizer_instance.mock_llm_call.assert_called_once_with({'title': title, 'text': content})

// ... existing code ...