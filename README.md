# ReAPR
Here is the open-source code repository for the paper "ReAPR: Automatic Program Repair via Retrieval-Augmented Large Language Models."

It is structured as follows:
- [BM25](BM25) contains the specific implementation of the BM25 algorithm.
- [DPR](DPR) contains the specific implementation of the BM25 algorithm, including model training, word vector embedding, and similarity retrieval.
- [Dataset](Dataset) contains the processing logic for two benchmarks as well as the patch validation logic on these two benchmarks.
- [Defects4j](Defects4j) contains the single-function bugs we extracted from Defects4J 2.0.
- [Gitbug-java](Gitbug-java) contains the single-function bugs we extracted from GitBug-Java.
- [reapir.py](repair.py) contains the entire repair process and serves as the main program entry point.

## Benchmarks

Before running the program, please make sure to configure Defects4J and GitBug-Java properly.
Defects4J: [https://github.com/rjust/defects4j](https://github.com/rjust/defects4j)
GitBug-Java: [https://github.com/gitbugactions/gitbug-java](https://github.com/gitbugactions/gitbug-java)

## Retrieval Corpus

The retrieval database supporting the results of this study is openly available in the HuggingFace repository. The dataset can be accessed at: [https://huggingface.co/datasets/zxliu/ReAPR-Automatic-Program-Repair-via-Retrieval-Augmented-Large-Language-Models](https://huggingface.co/datasets/zxliu/ReAPR-Automatic-Program-Repair-via-Retrieval-Augmented-Large-Language-Models)
