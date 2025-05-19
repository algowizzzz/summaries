import json
import logging
import time
import os
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
# from langchain_openai import ChatOpenAI # Old import
# from langchain_community.callbacks.manager import get_openai_callback # OpenAI specific
from typing import Optional, Union, Dict, Any
from dotenv import load_dotenv # Add this import

load_dotenv() # Add this line to load the .env file

# Ensure OPENAI_API_KEY is set in the environment
# from dotenv import load_dotenv
# load_dotenv() # Uncomment if you use a .env file for API keys

class LLMSummarizer:
    """
    Handles LLM-based summarization using LangChain.
    It loads prompt templates from a JSON configuration file and uses an Anthropic model.
    """
    def __init__(self, prompt_set_path, model_name="claude-3-haiku-20240307", temperature=0.2, max_retries=2):
        self.prompt_set_path = prompt_set_path
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompts = self._load_prompt_set(prompt_set_path)
        
        # Initialize the LLM. ANTHROPIC_API_KEY should be in env.
        # Updated to ChatAnthropic
        self.llm = ChatAnthropic(model_name=self.model_name, temperature=self.temperature)
        logging.info(f"LLMSummarizer initialized with model: {self.model_name}, temp: {self.temperature}")

    @staticmethod
    def estimate_tokens(text: str, char_per_token: float = 3.5) -> int:
        """Estimates token count based on character length."""
        if not text:
            return 0
        return int(len(text) / char_per_token)

    def _load_prompt_set(self, path):
        """Loads prompt configurations from a JSON file."""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            # The config structure in README.md is: { "filing_type": { "mode": "template_path.md" } ... }
            # The codebase.md example has: { "mode": { "template": "string", "input_variables": [] } }
            # Let's adapt to the README.md structure, which is more flexible for different filing types.
            # We will load the *content* of the template files specified.
            
            loaded_prompts = {}
            for filing_type, modes in config.items():
                loaded_prompts[filing_type] = {}
                if isinstance(modes, dict): # Expected structure
                    for mode, template_file_path in modes.items():
                        try:
                            with open(template_file_path, 'r', encoding='utf-8') as tf:
                                template_string = tf.read()
                            loaded_prompts[filing_type][mode] = PromptTemplate.from_template(template_string)
                        except FileNotFoundError:
                            loaded_prompts[filing_type][mode] = None # Mark as missing
                        except Exception as e: # GENERIC EXCEPTION
                            loaded_prompts[filing_type][mode] = None
                else:
                    logging.warning(f"Invalid structure for filing_type '{filing_type}' in prompt set. Expected a dict of modes.")
            
            logging.info(f"Prompt set loaded successfully from {path}")
            return loaded_prompts
        except FileNotFoundError:
            logging.error(f"Prompt-set file not found: {path}. Summarization will likely fail.")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from prompt-set file {path}: {e}")
            return {}
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading prompt set from {path}: {e}")
            return {}

    def get_prompt_template(self, mode, filing_type="default"):
        """Gets a compiled LangChain PromptTemplate for the given mode and filing type."""

        if filing_type in self.prompts and mode in self.prompts[filing_type]:
            pt_obj = self.prompts[filing_type][mode]
            if pt_obj:
                return pt_obj
            logging.warning(f"Template object for {filing_type}-{mode} is None or evaluates to False. Falling back to default.")
        
        # Fallback to default filing_type if specific one not found or its mode is missing
        if "default" in self.prompts and mode in self.prompts["default"]:
            default_template = self.prompts["default"].get(mode)
            if default_template: 
                return default_template
        
        logging.error(f"No suitable prompt template found for mode '{mode}' (filing_type: '{filing_type}' or default). Cannot summarize.")
        raise ValueError(f"Missing prompt template for mode '{mode}' (filing_type: '{filing_type}')")

    def summarize(self, mode, filing_type="default", **kwargs):
        """
        Generates a summary for the given content using the appropriate prompt template.
        kwargs should contain all necessary variables for the prompt template (e.g., content, section_title).
        """
        prompt_template = self.get_prompt_template(mode, filing_type)
        if not prompt_template:
            # Error already logged by get_prompt_template
            return "Error: Prompt template not found."
        
        # Prioritize 'content' kwarg for 'text' template variable if 'text' is expected
        if 'content' in kwargs and 'text' in prompt_template.input_variables:
            kwargs['text'] = kwargs.pop('content')

        # Log the input variables expected by the template vs. provided
        expected_vars = set(prompt_template.input_variables)
        provided_vars = set(kwargs.keys())
        if not expected_vars.issubset(provided_vars):
            missing = expected_vars - provided_vars
            logging.warning(f"Missing input variables for prompt mode '{mode}', filing_type '{filing_type}': {missing}. Prompt may fail or be incomplete.")
            # You might want to raise an error here or provide default values
            for var in missing:
                 kwargs[var] = "[Information not provided]" # Provide a placeholder

        # Create the LLMChain for this specific call
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        retries = 0
        summary_text = ""
        while retries <= self.max_retries:
            try:
                # Added detailed logging for content length
                content_to_log = ""
                if 'content' in kwargs:
                    content_to_log = kwargs['content']
                elif 'text' in kwargs: # If 'content' was popped into 'text'
                    content_to_log = kwargs['text']
                
                logging.debug(f"Calling LLM for mode '{mode}', filing '{filing_type}'. Attempt {retries + 1}/{self.max_retries + 1}. Input keys: {list(kwargs.keys())}. Length of content/text: {len(content_to_log)} chars.")
                
                # Log the fully formatted prompt string that will be sent
                try:
                    formatted_prompt_for_log = prompt_template.format(**kwargs)
                    logging.debug(f"Formatted prompt for LLM (first 500 chars): {formatted_prompt_for_log[:500]}")
                    # Log length of formatted prompt to compare with Anthropic's token count
                    logging.debug(f"Length of formatted_prompt_for_log: {len(formatted_prompt_for_log)} chars.")
                except Exception as e_format:
                    logging.error(f"Error formatting prompt for logging: {e_format}")

                # Using get_openai_callback to track token usage for this specific call
                # with get_openai_callback() as cb: # This is OpenAI specific, commenting out
                # Langchain run method expects a dict if multiple input_variables, or single value if one.
                # If prompt expects `content`, and kwargs has `content`, it should work.
                # If prompt has multiple vars, e.g. `content` and `section_title`, kwargs must supply them.
                response = chain.run(kwargs) # Pass all kwargs
                summary_text = response.strip() if isinstance(response, str) else str(response) # Ensure string output
                # logging.debug(f"LLM call successful. Tokens used: {cb.total_tokens}, Cost (USD): ${cb.total_cost:.6f}") # OpenAI specific
                logging.debug(f"LLM call successful for model {self.model_name}.") # Generic success message
                break # Success, exit retry loop
            except Exception as e:
                logging.error(f"LLM API call failed (attempt {retries + 1}): {e}")
                retries += 1
                if retries > self.max_retries:
                    logging.error("Max retries reached. Failed to get summary.")
                    return f"Error: Failed to generate summary after {self.max_retries + 1} attempts. Last error: {e}"
                time.sleep(2 ** retries) # Exponential backoff
        
        return summary_text

class PromptLoader:
    """Placeholder for PromptLoader class."""
    def __init__(self, config_path: str, templates_dir: str):
        self.config_path = config_path
        self.templates_dir = templates_dir
        # Minimal initialization for testing purposes
        logging.info(f"[Placeholder] PromptLoader initialized with config: {config_path}, templates_dir: {templates_dir}")
        self.prompts_config = self._load_config()

    def _load_config(self):
        # Simplified config loading for placeholder
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"[Placeholder] Error loading prompt config {self.config_path}: {e}")
            return {}

    def get_prompt(self, filing_type: str, mode: str) -> Dict[str, str]:
        # Return dummy prompts to allow tests to proceed
        logging.debug(f"[Placeholder] get_prompt called for filing_type: {filing_type}, mode: {mode}")
        
        # Try to find specific prompt
        if filing_type in self.prompts_config and mode in self.prompts_config[filing_type]:
            template_paths = self.prompts_config[filing_type][mode]
            # This placeholder won't actually load template files, just return defaults
            # Real implementation would load template_paths['system_prompt_template'] etc.
            if (isinstance(template_paths, dict) and
                template_paths.get('system_prompt_template') and
                template_paths.get('user_prompt_template')):
                logging.info(f"[Placeholder] Using configured (but not loaded) templates for {filing_type} - {mode}")
                return {
                    "system": "System: Placeholder for configured system prompt.",
                    "user": "User: Placeholder for configured user prompt."
                }

        # Fallback to truly default prompts if specific or default config not found/valid
        default_system_prompt = "You are an expert financial analyst and skilled summarizer. Your task is to provide a concise and accurate summary of the given financial document or text segment."
        default_user_prompt = "Please summarize the following financial document segment accurately and concisely: {text_content}"
        
        logging.warning(f"[Placeholder] Using hardcoded default prompts for {filing_type} - {mode}.")
        return {
            "system": default_system_prompt,
            "user": default_user_prompt
        }

# Example Usage (Illustrative - needs actual prompt files and OpenAI API key)
if __name__ == '__main__':
    # Create dummy prompt files for testing
    os.makedirs("templates", exist_ok=True)
    os.makedirs("prompt_sets", exist_ok=True)
    
    # Dummy default prompt set JSON
    default_prompt_set_content = {
        "default": {
            "file": "templates/file_summary.txt",
            "node": "templates/node_summary.txt",
            "master": "templates/master_summary.txt"
        },
        "40-F": {
            "file": "templates/file_summary_40F.txt" 
        }
    }
    with open("prompt_sets/default_prompt_set.json", 'w') as f:
        json.dump(default_prompt_set_content, f)

    # Dummy template files
    with open("templates/file_summary.txt", "w") as f: f.write("Summarize this document: {content}")
    with open("templates/node_summary.txt", "w") as f: f.write("Summarize this section titled '{section_title}': {content}")
    with open("templates/master_summary.txt", "w") as f: f.write("Create a master summary for '{document_title}' based on: {content}")
    with open("templates/file_summary_40F.txt", "w") as f: f.write("40-F Specific Summary for {source_filename}: {content}")

    # Ensure OPENAI_API_KEY is set in your environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable to run this example.")
    else:
        logging.basicConfig(level=logging.DEBUG)
        summarizer = LLMSummarizer(prompt_set_path="prompt_sets/default_prompt_set.json")
        
        sample_content = "This is a long piece of text that needs to be summarized effectively by an advanced AI model."
        
        print("--- File Mode Summary (Default) ---")
        file_summary = summarizer.summarize(mode="file", content=sample_content, source_filename="doc1.pdf")
        print(f"Summary: {file_summary}")

        print("\n--- Node Mode Summary (Default) ---")
        node_summary = summarizer.summarize(mode="node", section_title="Introduction", content="This is the introduction text.", source_filename="doc1.pdf")
        print(f"Summary: {node_summary}")

        print("\n--- Master Mode Summary (Default) ---")
        master_summary = summarizer.summarize(mode="master", document_title="Comprehensive Report Q1", content="Summary of section 1... Summary of section 2...")
        print(f"Summary: {master_summary}")

        print("\n--- File Mode Summary (40-F Specific) ---")
        # Test with a filing_type that has a specific template
        file_summary_40f = summarizer.summarize(mode="file", filing_type="40-F", content=sample_content, source_filename="report_40F.json")
        print(f"Summary (40-F): {file_summary_40f}")

        print("\n--- Test Missing Template ---")
        try:
            summarizer.summarize(mode="non_existent_mode", content="test")
        except ValueError as e:
            print(f"Caught expected error: {e}")
        
        print("\n--- Test Missing Filing Type (should use default) ---")
        # Assumes 'default' has a 'file' mode template
        summary_missing_ftype = summarizer.summarize(mode="file", filing_type="XYZ-FORM", content=sample_content, source_filename="doc2.txt")
        print(f"Summary (Missing Filing Type, Default Used): {summary_missing_ftype}") 