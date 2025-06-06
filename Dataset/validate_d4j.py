import json
import time
import javalang
import subprocess
import re
import os
import signal


def run_d4j_test(source, testmethods, bug_id):
    bugg = False
    compile_fail = False
    timed_out = False
    entire_bugg = False
    error_string = ""

    try:
        tokens = javalang.tokenizer.tokenize(source)
        parser = javalang.parser.Parser(tokens)
        parser.parse()
    except:
        print("Syntax Error")
        return compile_fail, timed_out, bugg, entire_bugg, True

    for t in testmethods:
        Returncode = ""
        error_file = open("stderr.txt", "wb")
        child = subprocess.Popen(
            ["/home/lzx/defects4j/framework/bin/defects4j", "test", "-w", '/tmp/defects4j/' + bug_id + '/', "-t", t.strip()],
            stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
            start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()  # child.stdout.read()
                print(b"".join(Returncode).decode('utf-8'))
                error_file.close()
                break
            elif Flag != 0 and Flag is not None:
                compile_fail = True
                error_file.close()
                with open("stderr.txt", "rb") as f:
                    r = f.readlines()
                f.close()
                for line in r:
                    if re.search(':\serror:\s', line.decode('utf-8')):
                        error_string = line.decode('utf-8')
                        break
                print(error_string)
                break
            elif time.time() - while_begin > 15:
                error_file.close()
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                timed_out = True
                break
            else:
                time.sleep(0.01)
        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            continue
        else:
            bugg = True
            break

    # Then we check if it passes all the tests, include the previously okay tests
    if not bugg:
        print('So you pass the basic tests, Check if it passes all the test, include the previously passing tests')
        Returncode = ""
        child = subprocess.Popen(
            ["/home/lzx/defects4j/framework/bin/defects4j", "test", "-w", '/tmp/defects4j/' + bug_id + '/'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1,
            start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()  # child.stdout.read()
                break
            elif Flag != 0 and Flag is not None:
                bugg = True
                break
            elif time.time() - while_begin > 180:
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                bugg = True
                break
            else:
                time.sleep(0.01)
        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            print('success')
        else:
            entire_bugg = True

    return compile_fail, timed_out, bugg, entire_bugg, False


def validate_one_d4j_patches(buggy_file,patch):
    with open("Defects4j/single_function_repair.json", "r") as f:
        bug_dict = json.load(f)

    print("Validating patch ... ")
    bug_id = buggy_file.split('.')[0].split('-')[-1]
    project = buggy_file.split('.')[0].split('-')[0]
    bug_name = project + "-" + bug_id
    start = bug_dict[bug_name]['start']
    end = bug_dict[bug_name]['end']
    tmp_bug_id = "test_" + project + "-" + bug_id

    subprocess.run('rm -rf ' + '/tmp/defects4j/' + tmp_bug_id, shell=True)
    subprocess.run(
        ["/home/lzx/defects4j/framework/bin/defects4j", "checkout", "-p", project, "-v",
         bug_id + 'b', "-w", '/tmp/defects4j/' + tmp_bug_id],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    testmethods_result = subprocess.run(["/home/lzx/defects4j/framework/bin/defects4j", "export", "-w", "/tmp/defects4j/" + tmp_bug_id, "-p", "tests.trigger"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    testmethods = testmethods_result.stdout.decode().splitlines()

    source_dir_result = subprocess.run(
        ["/home/lzx/defects4j/framework/bin/defects4j", "export", "-p", "dir.src.classes", "-w", "/tmp/defects4j/" + tmp_bug_id],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    source_dir = source_dir_result.stdout.decode().splitlines()[-1].strip()
    with open("Defects4j/location" + "/{}.buggy.lines".format(bug_name), "r") as f:
        locs = f.read()

    loc = set([x.split("#")[0] for x in locs.splitlines()])  # should only be one
    loc = loc.pop()

    try:
        with open("/tmp/defects4j/" + tmp_bug_id + "/" + source_dir + "/" + loc, 'r') as f:
            source = f.readlines()
    except:
        with open("/tmp/defects4j/" + tmp_bug_id + "/" + source_dir + "/" + loc, 'r', encoding='ISO-8859-1') as f:
            source = f.readlines()

    source = "".join(source[:start - 1] + patch + source[end:])
    try:
        with open("/tmp/defects4j/" + tmp_bug_id + "/" + source_dir + "/" + loc, 'w') as f:
            f.write(source)
    except:
        with open("/tmp/defects4j/" + tmp_bug_id + "/" + source_dir + "/" + loc, 'w', encoding='ISO-8859-1') as f:
            f.write(source)

    compile_fail, timed_out, bugg, entire_bugg, syntax_error = run_d4j_test(source, testmethods, tmp_bug_id)
    subprocess.run('rm -rf ' + '/tmp/defects4j/' + tmp_bug_id, shell=True)
    if not compile_fail and not timed_out and not bugg and not entire_bugg and not syntax_error:
        print("Patch is valid")
        return 'valid'
    else:
        print("Patch is invalid")
        return 'invalid'
