import os
import hashlib
import logging
import json

class CacheManager:
    """
    Manages caching of processed content chunks to avoid redundant LLM calls.
    Cache entries are stored as files in a specified cache directory.
    The filename is the hash of the content, and the file can be empty or store metadata.
    """
    def __init__(self, cache_dir=".cache", enabled=True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
                logging.info(f"Cache directory created: {self.cache_dir}")
            except OSError as e:
                logging.error(f"Failed to create cache directory {self.cache_dir}: {e}")
                self.enabled = False # Disable caching if directory creation fails

    def hash_content(self, content):
        """Generate a SHA-256 hash for a piece of content (string)."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def is_cached(self, content_hash):
        """Check if a content hash exists in the cache."""
        if not self.enabled:
            return False
        cache_file_path = os.path.join(self.cache_dir, f"{content_hash}.cache")
        is_present = os.path.exists(cache_file_path)
        if is_present:
            logging.debug(f"Cache hit for hash: {content_hash}")
        else:
            logging.debug(f"Cache miss for hash: {content_hash}")
        return is_present

    def mark_cached(self, content_hash, metadata=None):
        """
        Mark a content hash as cached by creating a file. Optionally store metadata.
        If metadata is provided, it's saved as JSON in the cache file.
        """
        if not self.enabled:
            return
        cache_file_path = os.path.join(self.cache_dir, f"{content_hash}.cache")
        try:
            with open(cache_file_path, 'w') as f:
                if metadata:
                    json.dump(metadata, f)
                else:
                    f.write("") # Create an empty file to mark as cached
            logging.debug(f"Marked as cached: {content_hash}")
        except IOError as e:
            logging.error(f"Failed to write cache file {cache_file_path}: {e}")

    def get_cached_metadata(self, content_hash):
        """Retrieve metadata for a cached item, if it exists and is not empty."""
        if not self.is_cached(content_hash): # Relies on is_cached to check self.enabled
            return None
        cache_file_path = os.path.join(self.cache_dir, f"{content_hash}.cache")
        try:
            with open(cache_file_path, 'r') as f:
                content = f.read()
                if content:
                    return json.loads(content)
                return None # Empty file means no specific metadata, just presence
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Failed to read or parse cache metadata from {cache_file_path}: {e}")
            return None

    def clear_cache(self):
        """Remove all files from the cache directory."""
        if not self.enabled or not os.path.exists(self.cache_dir):
            logging.info("Cache is not enabled or directory does not exist. Nothing to clear.")
            return
        
        cleared_count = 0
        error_count = 0
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    cleared_count += 1
                # Optionally, could also remove subdirectories if your cache structure uses them
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}')
                error_count += 1
        
        if error_count == 0:
            logging.info(f"Cache cleared successfully. {cleared_count} items removed from {self.cache_dir}")
        else:
            logging.warning(f"Cache clearing partially failed. {cleared_count} items removed, {error_count} errors.") 