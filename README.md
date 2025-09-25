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
Benchmark Results
Model	Image Acc.	Instruction Acc.	Response Acc.
FalconEye (Ours)	88.56%	91.00%	94.22%
Qwen2.5VL-7B	81.44%	76.17%	80.00%
Beaver-dam	-	-	87.06%
FalconEye also outperforms GPT-4o and other baselines on multiple datasets (see paper for details).

## Quick Start
Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
Hardware
· Minimum: 1 × NVIDIA RTX 4090
Usage
Download the Falcon dataset and FalconEye model (see Releases or Links).
Run evaluation or inference scripts as described in the documentation.

## License
The Falcon dataset is released under the CC BY 4.0 License.
