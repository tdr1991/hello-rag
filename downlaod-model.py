#coding:utf-8
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(repo_id="BAAI/bge-small-en-v1.5", ignore_patterns=[".safetensors"], local_dir="/mnt/d/work/models")