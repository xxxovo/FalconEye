# Falcon: A Cross-Modal Evaluation Dataset for Comprehensive Safety Perception
## Overview
Falcon is a large-scale, meticulously curated multimodal dataset designed for fine-grained research on safety-related issues in Visual Question Answering (VQA) scenarios. FalconEye is an open-source evaluation model, fine-tuned on Falcon, that specializes in multimodal harm assessment and achieves state-of-the-art performance with efficient inference on a single RTX 4090 GPU.

## Harm Categories
Falcon evaluates VQA pairs across 13 harm categories:

Illegal Activity
Hate Speech
Bias
Fraud
Politics
Privacy Violation
Unlicensed Advice
Violence and Physical Harm
Malware
Economic Harm
Abuse
Unethical Behavior
Adult Content
## Quick Start
Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
Hardware
· Minimum: 1 × NVIDIA RTX 4090
Usage
Download the Falcon dataset and FalconEye model (see Releases or Links).<br>
Run evaluation or inference scripts as described in the documentation.

## License
The Falcon dataset is released under the CC BY 4.0 License.
