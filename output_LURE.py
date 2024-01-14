import argparse
import os
import random
import sys
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION, Conversation, SeparatorStyle

from PIL import Image
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--input_caption", help="path to input caption.")
    parser.add_argument("--input_image", help="path to image file.")
    parser.add_argument("--output_file", help="path to output file.")
    parser.add_argument("--mode", default= "rewrite")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), mode = args.mode)


results = []
input_dir =  args.input_image
output_file = args.output_file 
caption_data = []
if args.mode == 'rewrite':
    input_caption = args.input_caption
    with open(input_caption, 'r', encoding='utf-8') as f:
        for line in f:
            caption_data.append(json.loads(line.strip()))
count = 0
with torch.no_grad():
    with open(output_file, "a+") as f:
        for filename in tqdm(os.listdir(input_dir)):
            count +=1
            if filename in open(output_file).read():continue
            if filename.endswith((".jpg", ".jpeg", ".png")):
                file_id = filename
                qs = ""
                if args.mode == 'rewrite':
                    for temp_caption_item in caption_data:
                        if temp_caption_item["id"] == file_id:
                            qs += temp_caption_item["caption"]
                    prompt = 'According to the picture, remove the information that does not exist in the following description: ' + qs
                elif args.mode == 'inference':
                    prompt = 'Describe this image.'

                this_question = prompt
                chat_state = Conversation(
                    system='Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.', 
                    roles=('Human', 'Assistant'), 
                    messages=[['Human', '<Img><ImageHere></Img> ' + prompt]], 
                    offset=2, 
                    sep_style=SeparatorStyle.SINGLE, 
                    sep='###', 
                    sep2=None, 
                    skip_next=False, 
                    conv_id=None
                )
                image_path = os.path.join(input_dir, filename)
                image = Image.open(image_path).convert('RGB')
                img_list = []
                image = chat.vis_processor(image).unsqueeze(0).to('cuda:{}'.format(args.gpu_id))
                image_emb, _, = chat.model.encode_img(image)
                img_list.append(image_emb)
                if args.mode == 'rewrite':
                    output,_, _, _, _, _, _ = chat.answer(chat_state, img_list)

                    result = {"id": filename, "question": this_question, "caption": output, "model": "LURE"}
                elif args.mode == 'inference':
                    output, _,  _, u_wordlist, wordlist, plist, p_all = chat.answer(chat_state, img_list)
                    float_list = [tensor.item() for tensor in plist]
                    result = {"id": filename, "question": this_question, "caption": output,"objs": wordlist, "plist": float_list, "p_all": p_all, "model": "MiniGPT-4_13b"}


                json.dump(result, f)
                f.write('\n')
                f.flush()
f.close()

    
