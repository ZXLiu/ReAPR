import json
import difflib
import faiss
import numpy as np 
from tqdm import tqdm
from transformers import AutoModel,AutoTokenizer
import torch
import transformers
transformers.logging.set_verbosity_error()


def main(bug_str, top_k):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_path", default="")
    parser.add_argument("--bug_str", type=str, default=bug_str)
    parser.add_argument("--top_k", type=int, default=top_k)
    parser.add_argument("--encoding_batch_size", type=int, default=32)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--num_docs", type=int, default=364063)
    parser.add_argument("--embedding_dir", default="")
    parser.add_argument("--pretrained_model_path", default="")
    args = parser.parse_args()

    # make faiss index
    embedding_dimension = 768
    index = faiss.IndexFlatIP(embedding_dimension)
    for idx in tqdm(range(args.num_shards), desc='building index from embedding...', position=0):
        data = np.load(f"{args.embedding_dir}/fixes_shard_{idx}.npy")
        index.add(data)

    # load bug-fix snippets
    bug_snippets = []
    fix_snippets = []
    with open(args.fix_path, 'r') as f:
        bugs = json.load(f)
        for bug in tqdm(bugs, total=args.num_docs, desc="loading bug-fix snippets...", position=0):
            bug_snippets.append(bug['buggy_function'])
            fix_snippets.append(bug['fixed_function'])

    # load bug encoder
    bug_encoder = AutoModel.from_pretrained(args.pretrained_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bug_encoder.to(device).eval()

    # embed bug_str
    with torch.no_grad():
        bug_embedding = bug_encoder(**tokenizer(args.bug_str, max_length=256, truncation=True, padding=True, return_tensors='pt').to(device))
        bug_embedding = bug_embedding.last_hidden_state[:, 0, :]
        bug_embedding = bug_embedding.cpu().detach().numpy()

    # retrieve top-k documents
    _, I = index.search(bug_embedding, args.top_k)

    retrieval_pairs_list = []
    for row in I:
        for id in row:
            buggy_code = bug_snippets[id]
            fixed_code = fix_snippets[id]
            temp_dict = {'bug_str': buggy_code, 'fix_str': fixed_code}
            retrieval_pairs_list.append(temp_dict)

    return retrieval_pairs_list
