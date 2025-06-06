import json
from tqdm import tqdm
import os
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    BertTokenizer, AutoTokenizer,
    BertModel, AutoModel,
    )
import torch
import numpy as np
from accelerate import PartialState


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_path",default="")
    parser.add_argument("--num_docs",type=int,default=364063)
    parser.add_argument("--encoding_batch_size",type=int,default=512)
    parser.add_argument("--pretrained_model_path", default="")
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    distributed_state = PartialState()
    device = distributed_state.device

    fix_encoder = AutoModel.from_pretrained(args.pretrained_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    fix_encoder.eval()
    fix_encoder.to(device)

    # load fixes
    progress_bar = tqdm(total=args.num_docs, disable=not distributed_state.is_main_process,ncols=100,desc='loading fixes_snippets...', position=0)
    fix_snippets = []
    with open(args.fix_path, 'r') as f:
        fixes = json.load(f)
        for fix in fixes:
            fix_snippets.append(fix['fixed_function'])
            progress_bar.update(1)

    with distributed_state.split_between_processes(fix_snippets) as sharded_fix_snippets:
        
        sharded_fix_snippets = [sharded_fix_snippets[idx:idx+args.encoding_batch_size] for idx in range(0,len(sharded_fix_snippets),args.encoding_batch_size)]
        encoding_progress_bar = tqdm(total=len(sharded_fix_snippets), disable=not distributed_state.is_main_process,ncols=100,desc='encoding fix_snippets...', position=0)
        fix_embeddings = []
        for fix_snippets in sharded_fix_snippets:
            fix_list = [fix for fix in fix_snippets]
            model_input = tokenizer(fix_list,max_length=256,padding=True,return_tensors='pt',truncation=True).to(device)
            with torch.no_grad():
                CLS_POS = 0
                output = fix_encoder(**model_input).last_hidden_state[:,CLS_POS,:].cpu().numpy()
            fix_embeddings.append(output)
            encoding_progress_bar.update(1)
        fix_embeddings = np.concatenate(fix_embeddings,axis=0)
        os.makedirs(args.output_dir,exist_ok=True)
        np.save(f'{args.output_dir}/fixes_shard_{distributed_state.process_index}.npy',fix_embeddings)
