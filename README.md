# Falcon: A Cross-Modal Evaluation Dataset for Comprehensive Safety Perception
## Overview
Falcon is a large-scale, meticulously curated multimodal dataset designed for fine-grained research on safety-related issues in Visual Question Answering (VQA) scenarios. FalconEye is an open-source evaluation model, fine-tuned on Falcon, that specializes in multimodal harm assessment and achieves state-of-the-art performance with efficient inference on a single RTX 4090 GPU.

## Harm Categories
Falcon evaluates VQA pairs across 13 harm categories:

- Illegal Activity
- Hate Speech
- Bias
- Fraud
- Politics
- Privacy Violation
- Unlicensed Advice
- Violence and Physical Harm
- Malware
- Economic Harm
- Abuse
- Unethical Behavior
- Adult Content

## Quick Start
Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
Hardware

· Minimum: 1 × NVIDIA RTX 4090

## Directory Structure


### `eval/`

- **calculate_accuracy.py**  
  Calculates the overall accuracy of the model on harmful content detection tasks. Compares model predictions with ground truth labels to compute accuracy metrics.

- **calculate_harmcategory_accuracy.py**  
  Evaluates the model's classification accuracy for each harm category. Supports fine-grained analysis of model performance across different types of harmful content.

### `finetune/`

- **lora.py**  
  Script for model fine-tuning using the LoRA (Low-Rank Adaptation) method. Enables efficient fine-tuning of multimodal models on the Falcon dataset to improve performance on harmful content detection.

- **Falcon_test_example.csv**   
  An example dataset of the Falcon-test.

## Usage
Download the Falcon dataset and FalconEye model (see Releases or Links).<br>
Run evaluation or inference scripts as described in the documentation.

## License
The Falcon dataset is released under the CC BY 4.0 License.
