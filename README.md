# ReAPR
Here is the open-source code repository for the paper "ReAPR: Automatic Program Repair via Retrieval-Augmented Large Language Models."

It is structured as follows:
- [BM25](BM25) contains the specific implementation of the BM25 algorithm.
- [DPR](DPR) contains the specific implementation of the DPR algorithm, including model training, word vector embedding, and similarity retrieval.
- [Dataset](Dataset) contains the processing logic for two benchmarks as well as the patch validation logic on these two benchmarks.
- [Defects4j](Defects4j) contains the single-function bugs we extracted from Defects4J 2.0.
- [Gitbug-java](Gitbug-java) contains the single-function bugs we extracted from GitBug-Java.
- [reapir.py](repair.py) contains the entire repair process and serves as the main program entry point.
- [requirements.txt](requirements.txt) contains the dependencies that need to be installed for the project to run.

## Guide
### 1.Install Dependencies
To run this code, you first need to install the project dependencies and two benchmarks: [Defects4J](https://github.com/rjust/defects4j) and [GitBug-Java](https://github.com/gitbugactions/gitbug-java).<br>
To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```
### 2.BM25 Algorithm Implementation
After downloading the retrieval repository to a suitable local directory, run the following command to execute the BM25 algorithm.
```bash
cd BM25
python3 search_bm25.py --search_corpus your retrieval path --query_str the string to be retrieved --same_name results save location --temp_path temp path
```
### 3.DPR Algorithm Implementation
First, run the following command to train a dense retriever.
```bash
cd DPR
accelerate launch Train_dpr_retriever.py
```
Then, use the trained dense retriever to embed the fix content of each bug in the corpus into vectors, forming a vector retrieval database.
```bash
accelerate launch fix2embedding.py --fix_path retrieval corpus path --pretrained_model_path your dense retriever --output_dir embedding corpus path
```
Finally, run the following command to execute the DPR algorithm.
```bash
python3 search_dpr.py --fix_path retrieval corpus path --bug_str the string to be retrieved --top_k top_k --embedding_dir embedding corpus path --pretrained_model_path your dense retriever
```
### 4.ReAPR Workflow Implementation
Running the following command will execute the complete ReAPR workflow.
```bash
python3 repair.py --model_name generative model name --batch_size batch_size --dataset defects4j or githubjava --retrieval_way bm25 or dpr or no retrieval --chances beam search count
```
## Benchmarks

Before running the program, please make sure to configure Defects4J and GitBug-Java properly.<br>
- Defects4J: [https://github.com/rjust/defects4j](https://github.com/rjust/defects4j)<br>
- GitBug-Java: [https://github.com/gitbugactions/gitbug-java](https://github.com/gitbugactions/gitbug-java)

## Retrieval Corpus

The retrieval database supporting the results of this study is openly available in the HuggingFace repository. The dataset can be accessed at: [https://huggingface.co/datasets/zxliu/ReAPR-Automatic-Program-Repair-via-Retrieval-Augmented-Large-Language-Models](https://huggingface.co/datasets/zxliu/ReAPR-Automatic-Program-Repair-via-Retrieval-Augmented-Large-Language-Models)
