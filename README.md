# Falcon: A Cross-Modal Evaluation Dataset for Comprehensive Safety Perception
Overview
Falcon is a large-scale, meticulously curated multimodal dataset designed for fine-grained research on safety-related issues in Visual Question Answering (VQA) scenarios. FalconEye is an open-source evaluation model, fine-tuned on Falcon, that specializes in multimodal harm assessment and achieves state-of-the-art performance with efficient inference on a single RTX 4090 GPU.

Features
Falcon Dataset

57,515 VQA samples with rich annotations:
Harmfulness scores for instruction, image, and response
Specific harm categories (13 types)
Explanatory justifications
Supports model training, evaluation, and in-depth analysis of multimodal harm
FalconEye Model

Open-source, instruction-following, robust generalization
Efficient: Runs on a single RTX 4090 GPU
Outperforms closed-source models (e.g., GPT-4o) in harmful content assessment
Falcon-test Dataset

1,800 manually labeled VQA samples
Consistent annotation with three safety labels and harm categories per sample
Used to validate FalconEye’s effectiveness
Harm Categories
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

Quick Start
Requirements
Install dependencies:

Hardware
Minimum: 1 × NVIDIA RTX 4090
Usage
Download the Falcon dataset and FalconEye model (see Releases or Links).
Run evaluation or inference scripts as described in the documentation.
