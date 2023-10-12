# Analyzing and Mitigating Object Hallucination in Large Vision-Language Models


[Yiyang Zhou*](https://yiyangzhou.github.io/), [Chenhang Cui*](https://gzcch.github.io/), [Jaehong Yoon](https://jaehong31.github.io/), [Linjun Zhang](https://linjunz.github.io/), [Zhun Deng](https://www.zhundeng.org/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/), [Huaxiu Yao](https://www.huaxiuyao.io/)
<div align="center">
*Equal Contribution
</div>
<div align="center">
    <a href="https://arxiv.org/pdf/2310.00754.pdf"><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>

## News
* 🔥 [10.03] Our code and data have been organized and are about to be released!
* [10.03] Our paper is online now: https://arxiv.org/pdf/2310.00754.pdf.

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
 [Downlad](https://huggingface.co/Vision-CAIR/vicuna/tree/main) 

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
 [Downlad](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view) | [Downlad](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view)


Then, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/minigpt4_eval.yaml](eval_configs/minigpt4_eval.yaml#L11) at Line 11. 

**4. How to train your own LURE?**

**(step 1)** Prepare data set

You can modify your data set path [here](minigpt4/configs/datasets/cc_sbu/align.yaml#L5) at Line 5.
The final dataset path would be organized in a single folder, following a structure similar to what's described below:

```
dataset
├── filter_cap.json
└── image
    ├── 2.jpg
    ├── 3.jpg
    ...   
```

The file *'filter_cap.json'* contains our prepared 5000 LURE training data entries. Each sample within includes three fields: *'image_id'* , which represents the name of the image in the training data; *'caption'*, which denotes the detailed description obtained from [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) corresponding to the image; and *'h_caption'*, which signifies the hallucinated description we constructed based on 'caption' (this might include ambiguous objects and contributing objects).


### Model Inference
Prepare model inputs in the format similiar to the following:
```

```
Then, run the following command:
```bash
python output_LURE.py --cfg-path /path/to/config.yaml --gpu-id gpu-id --input_caption /path/to/caption_file  --input_image /path/to/image_file --output_file /path/to/output.jsonl
```
### Model Finetuning
The training samples are stored in xxx.json and orgnized in the following format:
```

```
The dataset would be a single folder in a structure as the following:
```
dataset
├── filter_cap.json
└── image
    ├── 2.jpg
    ├── 3.jpg
    ...   
```

Then, set the path to the dataset in the dataset config file xxx.
To launch the second stage alignment, 
first specify the path to the checkpoint file trained in stage 1 in 
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage2_finetune.yaml).
You can also specify the output path there. 
Then, run the following command. In our experiments, we use 1 A100.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```
## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@misc{zhou2023analyzing,
      title={Analyzing and Mitigating Object Hallucination in Large Vision-Language Models}, 
      author={Yiyang Zhou and Chenhang Cui and Jaehong Yoon and Linjun Zhang and Zhun Deng and Chelsea Finn and Mohit Bansal and Huaxiu Yao},
      year={2023},
      eprint={2310.00754},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


