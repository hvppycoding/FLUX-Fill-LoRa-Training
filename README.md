<!---
Copyright 2022 - The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# FLUX-Fill-LoRa-Training

This repository provides a fork of the [Sebastian-Zok/FLUX-Fill-LoRa-Training](https://github.com/Sebastian-Zok/FLUX-Fill-LoRa-Training), [🤗 Diffusers](https://github.com/huggingface/diffusers) and library with an example script for LoRA training on the new **FLUX.1-Fill** models. The script isn't optimized and was just tested on an NVIDIA A100 GPU. If anyone has a similar script for frameworks like SimpleTuner or SD-scripts, that run on consumer hardware, I would be more than happy to hear!

## Overview

The provided script implements a specific masking strategy, in my case applying a full mask to the image. If your use case requires a different masking approach, you’ll need to adapt the `random_mask` function accordingly.

## Installation

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the `examples/research_projects/dreambooth_inpaint` folder and run
```bash
pip install -r requirements_flux.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell (e.g., a notebook)

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups.
Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.



## Load your Dataset
For my case the dataset consisted of just plain images without image captions. Since I trained the LoRa on a specific task, I used the instance_prompt parameter for all generations. This is much more convinient than the in-context LoRa approach, that I used to learn concepts using the normal FLUX.1-dev model. Also there are no mask images, since it was hard coded for my use case (see random_mask).

## Train

Now, we can launch training using:

```bash
export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux"

accelerate launch train_dreambooth_inpaint_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="A character turnaround 45-degreed to the left" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A character turnaround 45-degreed to the left" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub
```

## Contributions and Feedback
As you might have noticed there is a lot of room for improvement 🙃. Feel free to open issues or submit pull requests to improve this project. If you have insights on adapting this script for other frameworks like SimpleTuner, please share your experiences!
 
