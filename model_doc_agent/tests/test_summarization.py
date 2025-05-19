# tests/test_summarization.py
import pytest
import os
import json
from unittest.mock import patch, MagicMock
from model_doc_agent.src.summarization import LLMSummarizer as Summarizer, PromptLoader

# Helper to create dummy template files and config for tests
@pytest.fixture(scope="module")
def setup_dummy_templates_and_config(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("summarizer_tests")
    templates_dir = base_dir 
    config_path = base_dir / "test_prompts.json"

    dummy_template_files = {
        "default_file.md": "Default File Summary: {title} - {text}",
        "default_node.md": "Default Node Summary for {document_title} section {section_name}: {text}",
        "default_cross.md": "Default Cross Analysis for {title} using questions {questions_text}: {text}",
        "news_file.md": "News File Summary: {title} - {text}",
        "news_node.md": "News Node Summary for {document_title} section {section_name}: {text}",
        "sec_10k_file.md": "SEC 10-K File Summary: {title} - {text}"
    }
    for fname, content in dummy_template_files.items():
        with open(templates_dir / fname, 'w') as f:
            f.write(content)

    dummy_config = {
        "default": {
            "file": str(templates_dir / "default_file.md"),
            "node": str(templates_dir / "default_node.md"),
            "cross": str(templates_dir / "default_cross.md")
        },
        "news": {
            "file": str(templates_dir / "news_file.md"),
            "node": str(templates_dir / "news_node.md")
        },
        "10-K": {
            "file": str(templates_dir / "sec_10k_file.md")
        }
    }
    with open(config_path, 'w') as f:
        json.dump(dummy_config, f, indent=2)
    
    return str(config_path), str(templates_dir) # templates_dir returned for PromptLoader test in original code, keep for now

@pytest.fixture
def summarizer_instance(setup_dummy_templates_and_config):
    config_path, _ = setup_dummy_templates_and_config # templates_dir not needed for LLMSummarizer init
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy_key"}):
        with patch('langchain.chains.LLMChain.run') as mock_llm_run:
            def side_effect_llm_run(inputs_dict): 
                actual_inputs = inputs_dict if isinstance(inputs_dict, dict) else {}
                # Prioritize section_name, then title, then document_title for the mock's display title
                title_for_mock = actual_inputs.get('section_name', 
                                   actual_inputs.get('title',
                                   actual_inputs.get('document_title', 
                                   actual_inputs.get('source_filename', 'UnknownTitle'))))
                prompt_text_content = actual_inputs.get('text', '') 
                return f"Mock Summary for '{title_for_mock}'. Input Text: '{str(prompt_text_content)[:30]}...'"
            mock_llm_run.side_effect = side_effect_llm_run
            
            summarizer = Summarizer(prompt_set_path=config_path) 
            summarizer.mock_llm_call = mock_llm_run 
            yield summarizer

# --- Tests for Summarizer (LLMSummarizer) --- #

def test_summarize_news_file_mode(summarizer_instance):
    content = "This is a news article about a recent event."
    title = "Big News Today"
    summary = summarizer_instance.summarize(mode="file", filing_type="news", title=title, content=content)
    assert f"Mock Summary for '{title}'" in summary # For file mode, title should be from 'title' kwarg
    assert f"Input Text: '{content[:30]}...'" in summary
    summarizer_instance.mock_llm_call.assert_called_once_with({'title': title, 'text': content})

def test_summarize_sec_file_mode(summarizer_instance):
    content = "This is an SEC 10-K filing content."
    title = "Example Corp 10-K"
    summary = summarizer_instance.summarize(mode="file", filing_type="10-K", title=title, content=content)
    assert f"Mock Summary for '{title}'" in summary
    assert f"Input Text: '{content[:30]}...'" in summary
    summarizer_instance.mock_llm_call.assert_called_once_with({'title': title, 'text': content})

def test_summarize_default_file_mode(summarizer_instance):
    content = "Some other document type."
    title = "Other Doc Title"
    summary = summarizer_instance.summarize(mode="file", filing_type="OTHER_TYPE", title=title, content=content)
    assert f"Mock Summary for '{title}'" in summary
    assert f"Input Text: '{content[:30]}...'" in summary
    summarizer_instance.mock_llm_call.assert_called_once_with({'title': title, 'text': content})

def test_summarize_news_node_mode(summarizer_instance):
    content = "A small chunk of a news article."
    doc_title = "Main News Story"
    section_name = "Update Section"
    summary = summarizer_instance.summarize(mode="node", filing_type="news", document_title=doc_title, section_name=section_name, content=content)
    assert f"Mock Summary for '{section_name}'" in summary # For node mode, title should be from 'section_name'
    assert f"Input Text: '{content[:30]}...'" in summary
    summarizer_instance.mock_llm_call.assert_called_once_with({'document_title': doc_title, 'section_name': section_name, 'text': content})

def test_summarize_default_node_mode(summarizer_instance):
    content = "A small chunk of a generic document."
    doc_title = "Generic Report"
    section_name = "Part 1"
    summary = summarizer_instance.summarize(mode="node", filing_type="default", document_title=doc_title, section_name=section_name, content=content)
    assert f"Mock Summary for '{section_name}'" in summary
    assert f"Input Text: '{content[:30]}...'" in summary
    summarizer_instance.mock_llm_call.assert_called_once_with({'document_title': doc_title, 'section_name': section_name, 'text': content})

def test_summarize_cross_mode(summarizer_instance):
    full_text_content = "Summary A. Summary B."
    title = "Cross Analysis"
    questions = ["What is A?", "Compare A and B."]
    questions_text = "\n".join(questions)
    summary = summarizer_instance.summarize(
        mode="cross", filing_type="default", title=title, 
        questions_text=questions_text, text=full_text_content
    )
    assert f"Mock Summary for '{title}'" in summary # For cross mode, title should be from 'title' kwarg
    assert f"Input Text: '{full_text_content[:30]}...'" in summary
    summarizer_instance.mock_llm_call.assert_called_once_with({'title': title, 'questions_text': questions_text, 'text': full_text_content})

def test_missing_template_file_in_config(tmp_path_factory, setup_dummy_templates_and_config):
    # This test was originally for PromptLoader, now adapted for LLMSummarizer
    config_path_base, _ = setup_dummy_templates_and_config
    
    base_dir = tmp_path_factory.mktemp("missing_template_test_specific") 
    specific_config_path = base_dir / "bad_prompts.json"

    bad_config = {"default": {"file": str(base_dir / "non_existent_template.md")}}
    with open(specific_config_path, 'w') as f: json.dump(bad_config, f)
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy_key"}):
        summarizer_bad_config = Summarizer(prompt_set_path=str(specific_config_path))
    
    with pytest.raises(ValueError, match="Missing prompt template for mode 'file'"):
        summarizer_bad_config.get_prompt_template(mode="file", filing_type="default")

    # The summarize method should also raise this ValueError when a template is missing
    with pytest.raises(ValueError, match=r"Missing prompt template for mode 'file'(?: \(filing_type: 'default'\))?"):
        summarizer_bad_config.summarize(mode="file", filing_type="default", title="Test", content="Test content")


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 