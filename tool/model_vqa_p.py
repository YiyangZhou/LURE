import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import numpy as np
import nltk


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    input_dir = args.input_dir
    output_file = args.output_file
    with torch.no_grad():
        with open(output_file, "a+") as f:
            for filename in tqdm(os.listdir(input_dir)):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    if filename in open(output_file).read():continue
                    qs = 'Describe this image.'
                    cur_prompt = qs
                    if model.config.mm_use_im_start_end:
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                    image = Image.open(os.path.join(args.input_dir, filename))
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids, probs = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            do_sample=True,
                            temperature=args.temperature,
                            top_p= 1,
                            num_beams= 1,
                            # no_repeat_ngram_size=3, args.top_p args.num_beams
                            max_new_tokens=1024,
                            use_cache=True)


                    input_token_len = input_ids.shape[1]
                    output_token = (output_ids[0])[input_token_len:]
                    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    if n_diff_input_output > 0:
                        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()
                    tokens = nltk.word_tokenize(outputs)
                    pos_tags = nltk.pos_tag(tokens)
                    u_wordlist=list()
                    wordlist = list()
                    p_list= list()
                    p_all = {}
                    for word, pos in pos_tags:
                        inputs = tokenizer(word).input_ids
                        if word not in p_all.keys():
                            p_all[word] = list()
                        if word not in wordlist and pos.startswith('NN'):
                            wordlist.append(word)                
                        for i in range(len(inputs)):
                            token = inputs[i]
                            if  torch.where(output_token == token)[0].numel() != 0:
                                    toke_idx = torch.where(output_token == token)[0][0]
                                    p_all[word].append(probs[toke_idx,token].cpu().item())
                                    if -np.log(probs[toke_idx,token].cpu())>0.9:
                                        if word not in u_wordlist and pos.startswith('NN'):
                                            u_wordlist.append(word)
                                            p_list.append(probs[toke_idx,token])
                                            break
                    float_list = [tensor.item() for tensor in p_list]
                    result = {"id": filename, "question": cur_prompt, "answer": outputs,"uncertain_objs": u_wordlist, "plist": float_list,"p_all": p_all, "objs": wordlist, "model": "MiniGPT-4_13b"}
                    json.dump(result, f)
                    f.write('\n')
                    f.flush()
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LLaVA-13B-v1")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default="your image folder")
    parser.add_argument("--output_file", type=str, default="your output folder/xxxx.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
