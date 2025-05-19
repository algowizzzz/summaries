# tests/test_cache.py
import os
import tempfile
import pytest
from model_doc_agent.src.cache import CacheManager

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
    # Cache directory shouldn't be created
    assert not os.path.exists(tmp_path/"cache") 