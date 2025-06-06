import os
import argparse
import json
from tqdm import tqdm
import pickle
import random
import csv
import time
import difflib

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25


def build(corpus_file, temp_dir):
    # corpus and query file have to be plain text files
    print("Building bm25 corpus")
    with open(corpus_file, 'r') as f:
        datas = json.load(f)
    try:
        os.mkdir(temp_dir)
    except FileExistsError:
        pass

    f1 = open(os.path.join(temp_dir, "corpus.jsonl"), "w")
    for i, data in enumerate(datas):
        f1.write(json.dumps({"_id": str(i), "text": data['buggy_function'].strip()}) + "\n")
    f1.flush()  # 刷新缓冲区
    f1.close()  # 关闭文件

    f2 = open(os.path.join(temp_dir, "bug_fix_pairs.jsonl"), "w")
    for i, data in enumerate(datas):
        f2.write(json.dumps({"_id":str(i), "buggy_function":data['buggy_function'].strip(), "fixed_function":data['fixed_function'].strip()})+"\n")
    f2.flush()  # 刷新缓冲区
    f2.close()  # 关闭文件


def get_retrieval_results(temp_path, save_name, top_k):
    index_list = []
    with open(save_name, 'rb') as file:
        data = pickle.load(file)
        for key, values in data.items():
            sorted_dict = sorted(values.items(), key=lambda x: x[1], reverse=True)
            for index, scores in sorted_dict[:top_k]:
                index_list.append(index)

    top_k_code = []
    bug_fix_pairs_file = os.path.join(temp_path, "bug_fix_pairs.jsonl")
    with open(bug_fix_pairs_file, 'rb') as f:
        for line in f:
            data = json.loads(line)
            if data["_id"] in index_list:
                top_k_code.append(data)

    bug_fix_pairs = []
    for index in index_list:
        for code in top_k_code:
            if code["_id"] == index:
                bug_fix_pairs.append(code)
                break
    return bug_fix_pairs


def main(query_str):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--search_corpus', '-i', default="/home/lzx/PycharmProject/myAPRproject_3/Build_Corpus/corpus/BM25.json", help="search corpus file, json file")
    parser.add_argument('--query_str', '-q', default=query_str, help="queries file, string")
    parser.add_argument('--save_name', '-o', default="/home/lzx/PycharmProject/myAPRproject_3/BM25/BM25-result.json", help="same file name")
    parser.add_argument('--temp_path', '-t', default="/home/lzx/PycharmProject/myAPRproject_3/BM25/beir", help="temp dir to save beir-format data")
    args = parser.parse_args()

    if not os.path.exists(args.temp_path):
        build(args.search_corpus, args.temp_path)
        time.sleep(10)

    fq = open(os.path.join(args.temp_path, "query.jsonl"), "w")
    fq.write(json.dumps({"_id": str(0), "text": query_str}) + "\n")
    fq.flush()  # 刷新缓冲区
    fq.close()  # 关闭文件

    fr = open(os.path.join(args.temp_path, "res.tsv"), "w")
    csv_fr = csv.writer(fr, delimiter='\t')
    fr.write("q\td\t\s\n")
    csv_fr.writerow([str(0), str(0), 1])
    fr.flush()  # 刷新缓冲区
    fr.close()  # 关闭文件

    # current_path = os.path.abspath(__file__)
    # path = os.path.dirname(current_path) + '/' + args.temp_path
    path = args.temp_path
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=os.path.join(path, "corpus.jsonl"),
        query_file=os.path.join(path, "query.jsonl"),
        qrels_file=os.path.join(path, "res.tsv")
    ).load_custom()

    model = BM25(index_name="reapr", hostname="169.254.3.1:9201", initialize=True)
    retriever = EvaluateRetrieval(model, k_values=[10])

    results = retriever.retrieve(corpus, queries)
    pickle.dump(results, open(args.save_name, "wb"))

    bug_fix_pairs = get_retrieval_results(args.temp_path, args.save_name, 1)
    return bug_fix_pairs
