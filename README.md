# Analyzing and Mitigating Object Hallucination in Large Vision-Language Models


[Yiyang Zhou*](https://yiyangzhou.github.io/), [Chenhang Cui*](https://gzcch.github.io/), [Jaehong Yoon](https://jaehong31.github.io/), [Linjun Zhang](https://linjunz.github.io/), [Zhun Deng](https://www.zhundeng.org/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/), [Huaxiu Yao](https://www.huaxiuyao.io/)
<div align="center">
*Equal Contribution
</div>
<div align="center">
    <a href="https://arxiv.org/pdf/2310.00754.pdf"><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>

## News
* 🚀 [11.29] Our new benchmark, [Bingo](https://github.com/gzcch/Bingo), is now online!
* 🔥 [10.03] Our paper is online now: https://arxiv.org/pdf/2310.00754.pdf.

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and ativate it via the following command

```bash
git clone https://github.com/YiyangZhou/LURE.git
cd LURE
conda env create -f environment.yml
conda activate LURE
```


**2. Prepare the pretrained Vicuna weights**

The current version of MiniGPT-4 is built on the v0 versoin of Vicuna-13B.
Download the corresponding LLM weights from the following huggingface space via clone the repository using git-lfs.
|                                          Vicuna V0 13B                                           |
:------------------------------------------------------------------------------------------------:
 [Download](https://huggingface.co/Vision-CAIR/vicuna/tree/main) 

The final weights would be in a single folder in a structure similar to the following:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

Then, set the path to the vicuna weight in the model config file 
[here](minigpt4/configs/models/minigpt4.yaml#L16) at Line 16.

**3. Prepare the pretrained MiniGPT-4 checkpoint**

Download the pretrained checkpoints according to the Vicuna model from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). In our paper, the initial parameters we used are from MiniGPT-4's stage1.

|                                Checkpoint Aligned with Vicuna 13B (stage 1）                               |                                Checkpoint Aligned with Vicuna 13B (stage 2）                               |
:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
 [Download](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view) | [Download](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view)


Then, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L11) at Line 11. 

**4. How to train your own LURE?**

**(Step 1)** Prepare dataset

You can modify your data set path [here](minigpt4/configs/datasets/cc_sbu/align.yaml#L5) at Line 5.
The final dataset path would be organized in a single folder, following a structure similar to what's described below:

```
dataset_train
├── filter_cap.json
└── image
    ├── 2.jpg
    ├── 3.jpg
    ...   
```

The file *'filter_cap.json'* contains our prepared 5000 LURE training data entries. Each sample within includes three fields: *'image_id'* , which represents the name of the image in the training data; *'caption'*, which denotes the detailed description obtained from [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) corresponding to the image; and *'h_caption'*, which signifies the hallucinated description we constructed based on *'caption'* (this might include ambiguous objects and contributing objects).

The images can be directly downloaded from [coco2014 train](https://cocodataset.org/#download). As for *'filter_cap.json'*, we have already prepared a version containing data masked for uncertain objects, which can be found at [here](dataset_train/). We have also uploaded a dataset (*'hallucination5k_train.jsonl'*) without masks, which includes several fields: *'value'* represents the corresponding *'caption'* in *'filter_cap.json'*, while *'h_value'* represents the unmasked version of *'h_caption'* in *'filter_cap.json'*. Additionally, *'co_objects'* indicates the co-occurring objects extracted by GPT, and *'uncertain_objects'* represents the uncertain objects extracted by LVLMs during the image description process.

**(Step 2)** Training

To launch the second stage alignment, first specify the path to the initial checkpoint file in [train_configs/minigpt4_stage2_finetune.yaml](train_configs/minigpt4_stage2_finetune.yaml).
You can also specify the output path there. 
Then, run the following command. In our experiments, we use 1 A100 80G.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```



### Model Inference
Prepare model captions in the format similiar to the following (To get the following reasoning format please refer to the sample reasoning file named *'model_vqa_q.py'* we provided in the *'tool'* folder):

```
{"id": "image_path", "answer": "caption of LLVM",  "p_all": {"word1": [probs, ...], "word2": [probs,...], ...}, "objs": ["obj1", "obj2", ...]}
```

 For extracting objects from sentences, natural language processing (NLP) libraries can be used for part-of-speech tagging or named entity recognition, such as NLTK (Natural Language Toolkit) and SpaCy. 
To output probabilities, we modify the generation/utils.py file in the Transformers library to generate probabilities for each token. We store the probability of each word's first token in a dictionary named 'p_all'.

To get the masked caption of  prepared captions,  run the following command:

```bash
python generate_IDK.py   --input_file /path/to/caption_file.jsonl  --output_file /path/to/idk_caption_file.jsonl
```


Then, run the following command to obtain the rewriting response:
```bash
python output_LURE.py --cfg-path /path/to/config.yaml --gpu-id gpu-id --input_caption /path/to/idk_caption_file  --input_image /path/to/image_file --output_file /path/to/output.jsonl
```
### Other
**Output probabilities during inference**

If you want to output probabilities during inference, please replace *'your_env_environment/lib/python xx.xx/site-packages/transformers/generation/utils.py'* with the *'utils.py'* file in the *'tool'* folder. We made modifications at lines 2559-2620 in the *'utils.py'* file. 

Once you have prepared the above steps, you can save the probabilities during the inference process by using sample reasoning file named *'model_vqa_p.py'* provided in *'tool'* folder.

**How to calculate CHAIR from the description**

We calculated chair metrics based on this [github](https://github.com/LisaAnne/Hallucination). For convenience I've organized it into the following process:

**(Step 1)**: Cloning the repository and preparing annotations

```bash
git clone https://github.com/LisaAnne/Hallucination.git
cd Hallucination
mkdir annotations
```

Download the corresponding annotations from the [website](https://cocodataset.org/#download) (2014 Train/Val annotations) and extract them to the folder *'annotations'*.

**(Step 2)**: Prepare your reasoned results and convert them to a standardized format

You get the reasoning results well documented in the following format in jsonl (where the id and answer fields are required):

```
{"id": "COCO_train2014_000000157393.jpg", "question": xxx, "answer": xxx, "model": xxx}
```

Convert the result file to the standard format needed for inference according to *'to_chair.py'* provided in the *'tool'* folder. Line 15 [Here](tool/to_chair.py#L15) is adjusted according to the id field of your jsonl to ensure that the sample's id in the output json is as follows:

```
"image_id": 157393
```

**(Step 3)**: Calculate chair

```bash
cd Hallucination/utils/
```

Replace *'--annotation_path'* and *'--cap_file'* in *'chair.py'* with the folder where you store the annotation and the address of the json you got in the previous step, respectively.

```bash
python chair.py
```

### Checkpoint release

The ckpt we trained based on MiniGPT-4 7B as a baseline is available at [Hugingface](https://huggingface.co/YiyangAiLab/LURE).

## Related Projects

- [CHAIR](https://github.com/LisaAnne/Hallucination)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Bingo](https://github.com/gzcch/Bingo)

## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@article{zhou2023analyzing,
  title={Analyzing and mitigating object hallucination in large vision-language models},
  author={Zhou, Yiyang and Cui, Chenhang and Yoon, Jaehong and Zhang, Linjun and Deng, Zhun and Finn, Chelsea and Bansal, Mohit and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2310.00754},
  year={2023}
}

@article{cui2023holistic,
  title={Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges},
  author={Cui, Chenhang and Zhou, Yiyang and Yang, Xinyu and Wu, Shirley and Zhang, Linjun and Zou, James and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2311.03287},
  year={2023}
}
```


