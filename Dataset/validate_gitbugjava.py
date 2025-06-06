import json
import shutil
import subprocess
import logging
import tqdm
import re
import os
from unidiff import PatchSet
env = os.environ.copy()
cwd = os.getcwd()
env['PATH'] = f"/home/lzx/gitbug-java:/home/lzx/gitbug-java/bin:{env.get('PATH', '')}"


def read_jsonl(file_name):
    all_projects = []
    with open(file_name, "r") as file:
        for line in file:
            entry = json.loads(line)
            all_projects.append(entry)
    return all_projects


def save_jsonl(data_list, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
        for entry in data_list:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(json_line + "\n")


def remove_java_comments(source):
    try:
        # Define states
        NORMAL, SINGLE_COMMENT, MULTI_COMMENT, STRING_LITERAL, CHAR_LITERAL = range(5)

        state = NORMAL
        result = []
        i = 0

        while i < len(source):
            # Check the current state and process accordingly
            if state == NORMAL:
                if source[i : i + 2] == "//":
                    state = SINGLE_COMMENT
                    i += 2
                elif source[i : i + 2] == "/*":
                    state = MULTI_COMMENT
                    i += 2
                elif source[i] == '"':
                    state = STRING_LITERAL
                    result.append(source[i])
                    i += 1
                elif source[i] == "'":
                    state = CHAR_LITERAL
                    result.append(source[i])
                    i += 1
                else:
                    result.append(source[i])
                    i += 1
            elif state == SINGLE_COMMENT:
                if source[i] == "\n":
                    state = NORMAL
                    result.append(source[i])
                    i += 1
                else:
                    i += 1
            elif state == MULTI_COMMENT:
                if source[i : i + 2] == "*/":
                    state = NORMAL
                    i += 2
                else:
                    i += 1
            elif state == STRING_LITERAL:
                if source[i] == "\\":
                    result.append(source[i])
                    i += 1
                    result.append(source[i])
                    i += 1
                elif source[i] == '"':
                    state = NORMAL
                    result.append(source[i])
                    i += 1
                else:
                    result.append(source[i])
                    i += 1
            elif state == CHAR_LITERAL:
                if source[i] == "\\":
                    result.append(source[i])
                    i += 1
                    result.append(source[i])
                    i += 1
                elif source[i] == "'":
                    state = NORMAL
                    result.append(source[i])
                    i += 1
                else:
                    result.append(source[i])
                    i += 1

        return "".join(result)
    except Exception as e:
        logging.warning(
            f"Failed to remove_java_comments from\n```n{source}\n```\nwith error: {e}"
        )
        return None


def remove_empty_lines(source):
    """Remove all empty lines from Java source code."""
    return re.sub(r"^\s*$\n", "", source, flags=re.MULTILINE)


def get_projects_for_gitbug_java(file_name):
    all_projects = read_jsonl(file_name)
    for project in tqdm.tqdm(all_projects, "Loading GitBug-Java"):
        bug_id = project["bug_id"].strip()
        run = subprocess.run(f"/home/lzx/gitbug-java/gitbug-java info {bug_id}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = run.stdout.decode("utf-8")
        diff = stdout.split("### Bug Patch")[1].split("```diff")[1].split("```")[0]
        project["ground_truth"] = diff
    save_jsonl(all_projects, "../Gitbug-java/gitbugjava_processed.jsonl")


def validate_one_gitbugjava_patches(bug_id,patch):
    project = ""
    all_projects = read_jsonl("Gitbug-java/gitbugjava_processed.jsonl")
    for project_item in all_projects:
        if project_item["bug_id"] == bug_id:
            project = project_item
            break
    diff = PatchSet(project["ground_truth"])

    tmp_bug_id = "test_" + bug_id
    tmp_bug_file_path = f"/tmp/gitbugjava/{tmp_bug_id}"

    subprocess.run('rm -rf ' + tmp_bug_file_path, shell=True)

    checkout_result = subprocess.run(f"/home/lzx/gitbug-java/gitbug-java checkout {bug_id} {tmp_bug_file_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if checkout_result.returncode != 0: raise Exception("Error while checking out bug")

    buggy_file_path = os.path.join(tmp_bug_file_path,
                                   (diff[0].source_file[2:] if diff[0].source_file.startswith("a/") else diff[0].source_file))

    with open(buggy_file_path, "r", encoding="ISO-8859-1") as f:
        buggy_code = f.read()

    # remove comments
    buggy_code = remove_java_comments(buggy_code)
    # remove empty lines
    buggy_code = remove_empty_lines(buggy_code)

    # remove comments
    project["buggy_code"] = remove_java_comments(project["buggy_code"])
    # remove empty lines
    project["buggy_code"] = remove_empty_lines(project["buggy_code"])

    if project["buggy_code"] not in buggy_code:
        error_str = f"Could not find buggy code in {buggy_file_path} for {bug_id}"
        print(error_str)
        logging.error(error_str)
        return error_str

    candidate_code = buggy_code.replace(project["buggy_code"], patch)

    with open(buggy_file_path,"w",encoding="ISO-8859-1",errors="replace",) as f:
        f.write(candidate_code)

    result = subprocess.run(f"/home/lzx/gitbug-java/gitbug-java run {tmp_bug_file_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    print(f"result.returncode: {result.returncode}\nresult.stdout: {result.stdout}\n")
    m = re.search(r"Failing tests: ([0-9]+)", result.stdout.decode("utf-8"))
    if result.returncode == 0 and m != None and int(m.group(1)) == 0:
        result_str = "valid"
        print("The patch is valid")
    else:
        result_str = "invalid"
        print("The patch is invalid")

    subprocess.run('rm -rf ' + tmp_bug_file_path, shell=True)

    return result_str
