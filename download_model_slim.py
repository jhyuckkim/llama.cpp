import os
import json
import torch
import argparse
import chardet
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from pathlib import Path
from itertools import chain
from typing import Set, Dict
import shutil

def collect_unique_token_ids(corpus_dir: Path, tokenizer_name: str) -> Set[int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    unique_token_ids = set()
    for file_path in chain(corpus_dir.glob("*.txt"), corpus_dir.glob("*.json")):
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            text = json.dumps(json.load(f)) if file_path.suffix == ".json" else f.read()
            token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            unique_token_ids.update(token_ids)
    
    return unique_token_ids

def initialize_embeddings_with_subset(model, unique_token_ids: Set[int]) -> Dict[int, int]:
    special_tokens = {model.config.bos_token_id, model.config.eos_token_id, model.config.pad_token_id}
    unique_token_ids.update(tid for tid in special_tokens if tid is not None)

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    embedding_dim = input_embeddings.embedding_dim
    unique_token_ids = sorted(unique_token_ids)

    original_to_new_id = {orig_id: new_idx for new_idx, orig_id in enumerate(unique_token_ids)}
    new_input_embeddings = torch.stack([input_embeddings.weight[token_id] for token_id in unique_token_ids])
    new_output_embeddings = torch.stack([output_embeddings.weight[token_id] for token_id in unique_token_ids])

    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(new_input_embeddings, freeze=False))
    model.set_output_embeddings(torch.nn.Linear(embedding_dim, len(unique_token_ids), bias=False))
    model.get_output_embeddings().weight = torch.nn.Parameter(new_output_embeddings)
    model.config.vocab_size = len(unique_token_ids)

    return original_to_new_id

def save_modified_configs_and_model(model, save_dir, original_to_new_id):
    if hasattr(model.config, "_name_or_path"):
        del model.config._name_or_path
    model.save_pretrained(save_dir)

    slimmed_vocab = [str(k) for k, v in sorted(original_to_new_id.items(), key=lambda item: item[1])]
    with open(save_dir / "slimed_vocab.txt", 'w') as f:
        f.write(" ".join(slimmed_vocab))

    print(f"Model saved with updated vocabulary size: {model.config.vocab_size}")
    print(f"Slimmed vocabulary saved at: {save_dir / 'slimed_vocab.txt'}")

def main(model_id: str = None, model_path: str = None, corpus_dir: str = None):
    if model_id:
        save_dir = Path("models") / (model_id + "-slim-embd")
        model_dir = snapshot_download(repo_id=model_id, local_dir=save_dir, local_dir_use_symlinks=False, revision="main")
    elif model_path:
        model_dir = Path(model_path)
        save_dir = model_dir.parent / (model_dir.name + "-slim-embd")
        save_dir.mkdir(parents=True, exist_ok=True)
        for file in model_dir.glob("*"):
            if file.name.startswith('.') or file.is_dir() or file.suffix == '.gguf' or file.name in {'.cache', '.lock'}:
                continue
            shutil.copy2(file, save_dir)
    else:
        raise ValueError("Either model_id or model_path must be provided.")

    model = AutoModelForCausalLM.from_pretrained(save_dir)
    tokenizer_name = str(model_dir)
    corpus_path = Path(corpus_dir)

    unique_token_ids = collect_unique_token_ids(corpus_path, tokenizer_name)
    original_to_new_id = initialize_embeddings_with_subset(model, unique_token_ids)

    save_modified_configs_and_model(model, save_dir, original_to_new_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slim embedding layer of a model based on unique token IDs in a corpus.")
    parser.add_argument("--model_id", type=str, help="The Hugging Face model ID to download and modify.")
    parser.add_argument("--model_path", type=str, help="The local path to the model directory.")
    parser.add_argument("corpus_dir", type=str, help="Directory path to the corpus containing text or JSON files.")
    args = parser.parse_args()
    main(model_id=args.model_id, model_path=args.model_path, corpus_dir=args.corpus_dir)
