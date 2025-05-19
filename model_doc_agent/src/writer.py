import os
import json
import logging

class OutputWriter:
    """
    Handles writing the generated summaries (Markdown) and metadata (JSON) to disk.
    Ensures the output directory structure is created as needed.
    """
    def __init__(self):
        logging.info("OutputWriter initialized.")

    def write_summary(self, summary_text, output_path_md):
        """Writes the summary text to a Markdown file."""
        try:
            os.makedirs(os.path.dirname(output_path_md), exist_ok=True)
            with open(output_path_md, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            logging.info(f"Summary successfully written to: {output_path_md}")
        except IOError as e:
            logging.error(f"Failed to write summary to {output_path_md}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while writing summary to {output_path_md}: {e}")

    def write_metadata(self, metadata_dict, output_path_json):
        """Writes the metadata dictionary to a JSON file."""
        try:
            os.makedirs(os.path.dirname(output_path_json), exist_ok=True)
            with open(output_path_json, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2) # Use indent for readability
            logging.info(f"Metadata successfully written to: {output_path_json}")
        except IOError as e:
            logging.error(f"Failed to write metadata to {output_path_json}: {e}")
        except TypeError as e:
            logging.error(f"Metadata dictionary is not JSON serializable for {output_path_json}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while writing metadata to {output_path_json}: {e}")

# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    writer = OutputWriter()
    
    sample_summary = "This is a test summary for demonstration."
    sample_metadata = {
        "summary_id": "uuid-12345",
        "source_file": "doc.txt",
        "mode": "file",
        "timestamp": "2023-01-01T12:00:00Z"
    }
    
    # Create a temporary directory for output for this example
    temp_output_dir = "temp_output_writer_test"
    os.makedirs(temp_output_dir, exist_ok=True)
    
    md_path = os.path.join(temp_output_dir, "example_summary.md")
    json_path = os.path.join(temp_output_dir, "example_summary_meta.json")
    
    print(f"Attempting to write summary to: {md_path}")
    writer.write_summary(sample_summary, md_path)
    
    print(f"Attempting to write metadata to: {json_path}")
    writer.write_metadata(sample_metadata, json_path)
    
    # Verify files (optional manual check)
    if os.path.exists(md_path):
        print(f"Markdown file created: {md_path}")
        with open(md_path, 'r') as f_md:
            print(f"Content: \n{f_md.read()}")
    else:
        print(f"Markdown file NOT created: {md_path}")
        
    if os.path.exists(json_path):
        print(f"JSON metadata file created: {json_path}")
        with open(json_path, 'r') as f_json:
            print(f"Content: \n{f_json.read()}")
    else:
        print(f"JSON metadata file NOT created: {json_path}")

    # Clean up the temporary directory (optional)
    # import shutil
    # shutil.rmtree(temp_output_dir)
    # print(f"Cleaned up temporary directory: {temp_output_dir}") 