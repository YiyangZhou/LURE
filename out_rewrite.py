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
    parser.add_argument("--cfg-path", default="/nas-alinlp/yiyang.zyy/MiniGPT-4/eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
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
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))

# annotations = json.load(open("/nas-alinlp/zcl/MiniGPT-4-main/dataset_coco.json", "r"))
# annotations = annotations["images"]
# image_path = "/nas-alinlp/yiyang.zyy/data_shop/coco/val2014"
results = []
#/iris/u/huaxiu/ch/mgpt-4/rewrite_input/llava_cocotest_des_sample_context_idnt1.jsonl
input_dir = "/iris/u/huaxiu/ch/mgpt-4/coco5000_val/coco5000"
output_file = "/iris/u/huaxiu/ch/mgpt-4/output_json/minigpt4_rewrite_nocoo.jsonl" 
caption_data = []
with open("/iris/u/huaxiu/ch/mgpt-4/mgpt4_coco5k_des_sample_context_idnt1.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        caption_data.append(json.loads(line.strip()))
with torch.no_grad():
    with open(output_file, "a+") as f:
        for filename in tqdm(os.listdir(input_dir)):
            if filename in open(output_file).read():continue
            if filename.endswith((".jpg", ".jpeg", ".png")):
                file_id = filename
                # temp_caption = ""
                qs = ""
                for temp_caption_item in caption_data:
                    if temp_caption_item["id"] == file_id:
                        qs += temp_caption_item["answer"]
                # caption_cc = ""
                # for re_caption_item in re_caption:
                #     if re_caption_item["image_id"] == file_id:
                #         caption_cc += re_caption_item["caption"]
                # qs = 'Describe this image.'
                # qs = 'List the objects in this picture. Strictly follow the following format: object one, object two, etc.'
                # qs = f'''These objects exist in the picture: {temp_caption}. This is a reference description:{caption_cc}. Describe this image.'''
                this_question = qs
                # print(this_question)
                chat_state = Conversation(
                    system='Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.', 
                    roles=('Human', 'Assistant'), 
                    messages=[['Human', '<Img><ImageHere></Img> ' + 'According to the picture, remove the information that does not exist in the following description: ' + qs]], 
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
                output, _,  _, u_wordlist, wordlist, plist, p_all = chat.answer(chat_state, img_list)

                float_list = [tensor.item() for tensor in plist]
                result = {"id": filename, "question": this_question, "caption": output,"uncertain_objs": u_wordlist, "plist": float_list, "p_all": p_all, "model": "MiniGPT-4_13b"}
                json.dump(result, f)
                f.write('\n')
                f.flush()
f.close()

    