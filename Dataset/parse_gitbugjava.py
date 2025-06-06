import json


def clean_parse_gitbugjava(folder):
    result = []
    with open(folder + "Gitbug-java/gitbugjava.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            result.append(entry)

    cleaned_result = {}
    for item in result:
        lines = item['buggy_code'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[item["bug_id"] + ".java"] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
        lines = item['fixed_code'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[item["bug_id"] + ".java"]["fix"] = "\n".join([line[leading_white_space:] for line in lines])
    return cleaned_result
