import json
import numpy as np
import re
import argparse

def get_word(words, objlist):
    if words in objlist:
        return words
    elif words[:-1] in objlist:
        return words[:-1]
    elif words[:-2] in objlist:
        return words[:-2]
    elif words[:-3] in objlist:
        return words[:-3]
    elif words[:-4] in objlist:
        return words[:-4]
    elif words+'s' in objlist:
        return words+'s'
    elif words+'es' in objlist:
        return words+'es'
    else:
        return -1
def split_words(text):
    pattern = r"[\w']+|[^a-zA-Z0-9\s]"
    words = re.findall(pattern, text)
    return words
def replace_words_with_idk(sentence, objlist, p_all, un):
    words = (sentence.replace('.', ' .').replace(',', ' ,')).split()
    num_words = len(words)
    num_replacement = int(num_words * 0.2)
    del_list=list()
    for i in range(num_words - num_replacement, num_words):
        if words[i] in objlist or words[i][:-1] in objlist or words[i][:-2] in objlist or words[i][:-3] in objlist \
                or words[i][:-4] in objlist or 'words[i]' + 's' in objlist or 'words[i]' + 'es' in objlist:
            words[i] = "[IDK]"
            if i-1>=0:
                if ~(words[i-1]==',' or  words[i-1]=='.'):
                    del_list.append(i-1)
            if i+1< len(words):
                if ~(words[i-1]==',' or  words[i-1]=='.'):
                    del_list.append(i + 1)
    del_list.sort(reverse=True)

    for del_index in del_list:
        if del_index<len(words):
            if words[del_index] != '.' and words[del_index] != ',' and words[del_index] != '[IDK]':
                del words[del_index]
    del_list=list()
    new_sentence = " ".join(words)
    words = new_sentence.split()
    for i in range(len(words)):
        obj_word = get_word(words[i], objlist)
        if obj_word == -1 or len(p_all[obj_word])==0: continue
        if -np.log((p_all[obj_word])[0])> un:
            words[i] = "[IDK]"
            if i-1>=0:
                if ~(words[i-1]==',' or  words[i-1]=='.'):
                    del_list.append(i-1)
            if i+1< len(words):
                if ~(words[i-1]==',' or  words[i-1]=='.'):
                    del_list.append(i + 1)

    del_list.sort(reverse=True)
    for del_index in del_list:
        if words[del_index]!='.' and words[del_index]!=',' and words[del_index] != '[IDK]':
            del words[del_index]
    new_sentence = " ".join(words)
    pattern = r"\[(IDK)\](?:\s*\[\1\])+"
    result = re.sub(pattern, r"[\1]", new_sentence)
    new_sentence = result.replace(' .', '.').replace(' ,', ',')
    return new_sentence


def parse_args():
    parser = argparse.ArgumentParser(description="Create IDK")
    parser.add_argument("--input", type=str)
    parser.add_argument("--un", default= 0.9, type = float)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    return args

args = parse_args()
input_file = args.input
onput_file = args.output
un = args.un

if __name__ == "__main__":
    with open(input_file, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                answer = replace_words_with_idk(data['caption'], data["objs"], data["p_all"], un)
                data['caption'] = answer
                with open(onput_file, 'a+') as file:
                    json.dump(data, file)
                    file.write('\n')
                    file.flush()
