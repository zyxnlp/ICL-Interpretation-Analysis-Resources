# ICL-Interpretation-Analysis-Resources
Working in progress

## Overview
This repo contains relevant resources from our survey paper [The Mystery of In-Context Learning: A Comprehensive Survey on Interpretation and Analysis](https://arxiv.org/pdf/2311.00237) in EMNLP 2024. In this paper, we present a thorough and organized survey of the research on the interpretation and analysis of ICL. As further research in this area evolves, we will provide timely updates to this survey and this repository.


- [Theoretical Interpretation of ICL](#theoretical-interpretation-of-icl)
    - [Mechanistic Interpretability](#mechanistic-interpretability)
    - [Regression Function Learning](#regression-function-learning)
    - [Gradient Descent and Meta-Optimization](#gradient-descent-and-meta-optimization)
    - [Bayesian Inference](#bayesian-inference)
- [Empirical Analysis of ICL](#empirical-analysis-of-icl)
    - [Pre-training Data](#pre-training-data)
    - [Pre-training Model](#pre-training-model)
    - [Demonstration Order](#demonstration-order)
    - [Input-Label Mapping](#input-label-mapping)

## Theoretical Interpretation of ICL
Researchers in the theoretical category focus on interpreting the fundamental mechanism behind the ICL process through different conceptual lenses.
### Mechanistic Interpretability
* A mathematical framework for transformer circuits (Elhage et al., 2021).
  [[Paper]](https://transformer-circuits.pub/2021/framework/index.html)
  ![](https://img.shields.io/badge/AnthropicBlog%202021-olive)

### Regression Function Learning
* What can transformers learn in-context? a case study of simple function classes (Garg et al., 2022).
  [[Paper]](https://openreview.net/pdf?id=flNZJ2eOet)
  ![](https://img.shields.io/badge/NeurIPS%202022-purple)

### Gradient Descent and Meta-Optimization
* Why Can GPT Learn In-Context?
Language Models Implicitly Perform Gradient Descent as Meta-Optimizers (Dai et al., 2023).
  [[Paper]](https://aclanthology.org/2023.findings-acl.247.pdf)
  ![](https://img.shields.io/badge/ACL%202023-brown)

### Bayesian Inference
* An Explanation of In-context Learning as Implicit Bayesian Inference. (Xie et al., 2022).
  [[Paper]](https://openreview.net/pdf?id=RdJVFCHjUMI)
  ![](https://img.shields.io/badge/ICLR%202022-green)

## Empirical Analysis of ICL
Researchers in the empirical category focus on probing the factors that influence the ICL.
### Pre-training Data
* On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model (Shin et al., 2022).
  [[Paper]](https://aclanthology.org/2022.naacl-main.380.pdf)
  ![](https://img.shields.io/badge/NAACL%202022-brown)
* What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers (B Kim et al., 2021).
  [[Paper]](https://arxiv.org/abs/2109.04650)
  ![](https://img.shields.io/badge/EMNLP%202021-brown)
* Understanding In-Context Learning via Supportive Pretraining Data (Han et al., 2023).
  [[Paper]](https://arxiv.org/abs/2306.15091)
  ![](https://img.shields.io/badge/ACL%202023-brown)
* Mauve: Measuring the gap between neural text and human text using divergence frontiers (Pillutla et al., 2023).
  [[Paper]](https://arxiv.org/abs/2102.01454)
  ![](https://img.shields.io/badge/NeurIPS%202021-brown)
* Impact of Pretraining Term Frequencies on Few-Shot Reasoning (Razeghi et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2202.07206)
  ![](https://img.shields.io/badge/Findings_Of_EMNLP%202022-brown)
* Large Language Models Struggle to Learn Long-Tail Knowledge (Kandpal et al., 2023).
  [[Paper]](https://proceedings.mlr.press/v202/kandpal23a.html)
  ![](https://img.shields.io/badge/PMLR%202023-brown)
* Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression (Raventós et al., 2023).
  [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e10b2c2e1aa4f8083c37dfe269873f8-Abstract-Conference.html)
  ![](https://img.shields.io/badge/NeurIPS%202023-brown)
* Data distributional properties drive emergent in-context learning in transformers. (Chan et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2205.05055)
  ![](https://img.shields.io/badge/NeurIPS%202022-brown)
* THE PRINCIPLE OF LEAST EFFORT (ZIPF et al., 1949).
  [[Paper]](https://wli-zipf.upc.edu/pdf/zipf49-toc.pdf)
  ![](https://img.shields.io/badge/CoNLL%201949-brown)

  
  
  

### Pre-training Model
* Emergent abilities of large language models (Wei et al., 2022).
  [[Paper]](https://openreview.net/pdf?id=yzkSU5zdwD)
  ![](https://img.shields.io/badge/TMLR%202022-navy)

### Demonstration Order
* Fantastically Ordered Prompts and Where to Find Them:
Overcoming Few-Shot Prompt Order Sensitivity (Lu et al., 2022).
  [[Paper]](https://aclanthology.org/2022.acl-long.556.pdf)
  ![](https://img.shields.io/badge/ACL%202022-brown)

### Input-Label Mapping
* Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? (Min et al., 2022).
  [[Paper]](https://arxiv.org/pdf/2202.12837)
  ![](https://img.shields.io/badge/EMNLP%202022-brown)


## Citation

Please consider cite our paper when you find our resources useful!
```
@inproceedings{zhou2023mystery,
  title={The Mystery and Fascination of LLMs: A Comprehensive Survey on the Interpretation and Analysis of Emergent Abilities},
  author={Yuxiang ZHou and Jiazheng Li and Yanzheng Xiang and Hanqi Yan and Lin Gui and Yulan He},
  booktitle={Proc. of EMNLP},
  year={2024},
url={https://arxiv.org/pdf/2311.00237}
}
```
