import argparse

from tqdm import tqdm
import torch
import os
import json
import time

from model import RepairModel
from Dataset.parse_gitbugjava import clean_parse_gitbugjava
from Dataset.parse_d4j import clean_parse_d4j, get_unified_diff
from Dataset.validate_gitbugjava import validate_one_gitbugjava_patches
from Dataset.validate_d4j import validate_one_d4j_patches
from prompt import *
from BM25.search_bm25 import main as bm25_main
from DPR.search_dpr import main as dpr_main
from util import pick_smallest_example_fix, set_seed
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


def generateRetrievalPairs(pairs):
    retrieval_pairs = ""
    for pair in reversed(pairs):
        retrieval_pairs += "// Provide a fix for the buggy function\n\n"
        retrieval_pairs += f"// Buggy Function\n{pair['bug_str']}\n\n"
        retrieval_pairs += f"// Fixed Function\n{pair['fix_str']}\n\n"
    return retrieval_pairs[:-1]


def repair_loop(args, dataset, model, prompt, file_name, bug, t_chances):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    if not model.check_input(prompt, bug['buggy']):
        return 0, False, False, repair_result

    total_times = 0
    patch_count = 0
    while t_chances > 0:
        total_times += 1
        torch.cuda.empty_cache()
        print("Try :{}".format(total_times))
        well, length, outputs, entropies = model.model_predict(prompt, bug['buggy'], do_sample=True,
                                                               num_samples=t_chances)
        patch_count += args.batch_size
        t_chances -= args.batch_size
        if well:
            for index, output in enumerate(outputs):
                diff = get_unified_diff(bug['buggy'], output)
                if diff in p_diff:
                    repair_result[p_diff[diff]]['num'] += 1
                    continue
                p_diff[diff] = len(repair_result)
                eval_result = validate_one_d4j_patches(file_name, output) if dataset == 'defects4j' else validate_one_gitbugjava_patches(file_name, output)
                repair_result.append({'output': output,
                                      'diff': diff,
                                      'finish_reason': 'stop',
                                      'entropy': entropies[index],
                                      'valid': eval_result,
                                      'num': 1})

    end = time.time()

    print("{} Unique Patches Generated in {}s".format(len(repair_result), end - start))

    return len(repair_result), False, False, repair_result


def repair_defects4j(args, model, bugs, folder, retrival_way, used_prompt, chances, only_same=True):
    with open(folder + "/prompt.txt", "w") as f:
        f.write(used_prompt)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    t_plausible = 0

    start_t = time.time()
    for file_name, bug in tqdm(bugs.items(), desc='Processing bugs', unit='bug', position=0):
        if "Collections" in file_name:
            example_bug, example_fix = pick_smallest_example_fix(bugs, file_name, only_same=False)
        else:
            example_bug, example_fix = pick_smallest_example_fix(bugs, file_name, only_same=only_same)
        if retrival_way == 'BM25':
            retrieval_pairs = bm25_main(bug['buggy'].strip())
            prompt = used_prompt.format(retrieval_pairs=generateRetrievalPairs(retrieval_pairs),
                                        example_bug=example_bug, example_fix=example_fix, bug=bug['buggy'])
        elif retrival_way == 'DPR':
            retrieval_pairs = dpr_main(bug['buggy'].strip(), 1)
            prompt = used_prompt.format(retrieval_pairs=generateRetrievalPairs(retrieval_pairs),
                                        example_bug=example_bug, example_fix=example_fix, bug=bug['buggy'])
        else:
            prompt = used_prompt.format(example_bug=example_bug, example_fix=example_fix, bug=bug['buggy'])
        n_generated, valid, first_try, result[file_name] = repair_loop(args, 'defects4j', model, prompt, file_name, bug, chances)
        if n_generated >= 1:
            t_generated += chances
            t_unique += len(result[file_name])
            t_plausible = sum([1 for item in result[file_name] if item['valid'] == 'valid'])
    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total plausible: {}\n".format(t_plausible))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def repair_gitbugjava(args, model, bugs, folder, retrival_way, used_prompt, chances):
    if not os.path.exists(folder):
        os.makedirs(folder)

    result = {}
    t_generated = 0
    t_unique = 0
    t_plausible = 0

    start_t = time.time()
    for file_name, bug in tqdm(bugs.items(), desc='Processing bugs', unit='bug', position=0):
        if retrival_way == 'BM25':
            retrieval_pairs = bm25_main(bug['buggy'].strip())
            prompt = used_prompt.format(retrieval_pairs=generateRetrievalPairs(retrieval_pairs), bug=bug['buggy'])
        elif retrival_way == 'DPR':
            retrieval_pairs = dpr_main(bug['buggy'].strip(), 1)
            prompt = used_prompt.format(retrieval_pairs=generateRetrievalPairs(retrieval_pairs), bug=bug['buggy'])
        else:
            prompt = used_prompt.format(bug=bug['buggy'])
        n_generated, valid, first_try, result[file_name] = repair_loop(args, 'gitbug-java', model, prompt, file_name, bug, chances)
        if n_generated >= 1:
            t_generated += chances
            t_unique += len(result[file_name])
            t_plausible = sum([1 for item in result[file_name] if item['valid'] == 'valid'])
    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total plausible: {}\n".format(t_plausible))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="CodeLlama-7b")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="gitbug-java")
    parser.add_argument("--retrival_way", type=str, default="BM25")
    parser.add_argument("--chances", type=int, default=20)
    parser.add_argument("--folder", type=str, default="Results/test")
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()
    if args.dataset == "defects4j":
        dataset = clean_parse_d4j(folder="")
        if args.retrival_way == 'BM25' or args.retrival_way == 'DPR':
            prompt = JAVA_PROMPT_D4J_RETRIEVAL
        else:
            prompt = JAVA_PROMPT_D4J
        stop = "// Provide a fix for the buggy function"
        if args.single_line:
            stop = "\n"
        args.language = "java"
    elif args.dataset == "gitbug-java":
        dataset = clean_parse_gitbugjava(folder="")
        if args.retrival_way == 'BM25' or args.retrival_way == 'DPR':
            prompt = JAVA_PROMPT_GITBUG_RETRIEVAL
        else:
            prompt = JAVA_PROMPT_GITBUG
        stop = "// Provide a fix for the buggy function"
        if args.single_line:
            stop = "\n"
        args.language = "java"
    else:
        print("Unknown dataset: {}".format(args.dataset))
        return -1

    set_seed(args.seed)

    model = RepairModel(batch_size=args.batch_size, pretrained=args.model_name, stop=stop, weight=args.weight)

    if args.dataset == "defects4j":
        repair_defects4j(args, model, dataset, args.folder, args.retrival_way, prompt, args.chances, only_same=args.dataset.startswith("defects4j"))
    if args.dataset == "gitbug-java":
        repair_gitbugjava(args, model, dataset, args.folder, args.retrival_way, prompt, args.chances)


if __name__ == '__main__':
    main()
