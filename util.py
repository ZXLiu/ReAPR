import random
import torch
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_relevant_bugs(bugs, current_bug, only_same):
    potential_pairs = []
    project = current_bug.split("-")[0]
    for file_name, bug in bugs.items():
        if file_name == current_bug:
            continue
        if file_name.startswith(project + "-") and only_same:
            potential_pairs.append((len(bug['buggy']) + len(bug['fix']), file_name))
        elif not only_same:
            potential_pairs.append((len(bug['buggy']) + len(bug['fix']), file_name))
    # sort from smallest to largest
    potential_pairs.sort(key=lambda x: x[0])
    return potential_pairs


# picking an example fix pairs from a project
def pick_smallest_example_fix(bugs, current_bug, only_same=False):
    potential_pairs = _get_relevant_bugs(bugs, current_bug, only_same)
    return bugs[potential_pairs[0][1]]['buggy'], bugs[potential_pairs[0][1]]['fix']
