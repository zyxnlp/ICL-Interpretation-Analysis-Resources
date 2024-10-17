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
* Attention is All you Need (Vaswani et al., 2017).
  [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
  ![](https://img.shields.io/badge/NeurIPS%202017-purple)
* In-context Learning and Induction Heads (Olsson et al., 2022).
  [[Paper]](https://arxiv.org/abs/2209.11895)
* The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains (Edelman et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2402.11004)
* Schema-learning and rebinding as mechanisms of in-context learning and emergence (Swaminathan et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2311.08360)
  ![](https://img.shields.io/badge/NeurIPS%202023-purple)
* Function Vectors in Large Language Models (Todd et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2310.15213)
  ![](https://img.shields.io/badge/ICLR%202024-green)
* Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection (Bai et al., 2023).
  [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf)
  ![](https://img.shields.io/badge/NeurIPS%202023-purple)
### Regression Function Learning
* What can transformers learn in-context? a case study of simple function classes (Garg et al., 2022).
  [[Paper]](https://openreview.net/pdf?id=flNZJ2eOet)
  ![](https://img.shields.io/badge/NeurIPS%202022-purple)
* Transformers as Algorithms: Generalization and Stability in In-context Learning (Li et al., 2023).
  [[Paper]](https://openreview.net/pdf?id=CgB7wCExOF)
  ![](https://img.shields.io/badge/ICML%202023-navy)
* The Closeness of In-Context Learning and Weight Shifting for Softmax Regression (Li et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2304.13276.pdf)
* What learning algorithm is in-context learning? Investigations with linear models (Akyürek et al., 2023).
  [[Paper]](https://openreview.net/pdf?id=0g0X4H8yN4I)
  ![](https://img.shields.io/badge/ICLR%202023-green)
* How Do Transformers Learn In-Context Beyond Simple Functions? A Case Study on Learning with Representations (Guo et al., 2024).
  [[Paper]](https://openreview.net/pdf?id=ikwEDva1JZ)
  ![](https://img.shields.io/badge/ICLR%202024-green)
* In-context Learning and Induction Heads (Olsson et al., 2022).
  [[Paper]](https://arxiv.org/abs/2209.11895)
* Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection (Bai et al., 2023).
  [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/b2e63e36c57e153b9015fece2352a9f9-Paper-Conference.pdf)
  ![](https://img.shields.io/badge/NeurIPS%202023-purple)
### Gradient Descent and Meta-Optimization
* Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers (Dai et al., 2023).
  [[Paper]](https://aclanthology.org/2023.findings-acl.247.pdf)
  ![](https://img.shields.io/badge/ACL%202023-brown)
* The Dual Form of Neural Networks Revisited: Connecting Test Time Predictions to Training Patterns via Spotlights of Attention (Irie et al., 2022).
  [[Paper]](https://proceedings.mlr.press/v162/irie22a/irie22a.pdf)
  ![](https://img.shields.io/badge/NeurIPS%202022-purple)
* Transformers learn in-context by gradient descent (von Oswald et al., 2023).
  [[Paper]](https://proceedings.mlr.press/v202/von-oswald23a/von-oswald23a.pdf)
  ![](https://img.shields.io/badge/ICML%202023-navy)
* Uncovering mesa-optimization algorithms in Transformers (von Oswald et al., 2023).
  [[Paper]](https://arxiv.org/abs/2309.05858)
* In-context Learning and Gradient Descent Revisited (Deutch et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2311.07772v4)
  ![](https://img.shields.io/badge/NAACL%202024-brown)
* Do pretrained Transformers Learn In-Context by Gradient Descent? (Shen et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2310.08540v5)
  ![](https://img.shields.io/badge/ICML%202024-navy)
* Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models (Fu et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2310.17086)
* Numerical analysis (Gautschi., 2011).
  [[Book]](https://books.google.co.uk/books?hl=en&lr=&id=-fgjJF9yAIwC&oi=fnd&pg=PR7&ots=CTbDNPLltY&sig=AkkgsrA_DH502obLg2uXp4W5L6g&redir_esc=y#v=onepage&q&f=false)
### Bayesian Inference
* An Explanation of In-context Learning as Implicit Bayesian Inference. (Xie et al., 2022).
  [[Paper]](https://openreview.net/pdf?id=RdJVFCHjUMI)
  ![](https://img.shields.io/badge/ICLR%202022-green)
* Statistical Inference for Probabilistic Functions of Finite State Markov Chains (Baum and Petrie., 1966).
  [[Paper]](https://api.semanticscholar.org/CorpusID:120208815)
* Large Language Models Are Implicitly Topic Models: Explaining and Finding Good Demonstrations for In-Context Learning (Wang et al., 2023).
  [[Paper]](https://openreview.net/pdf?id=HCkI1b6ksc)
  ![](https://img.shields.io/badge/ICML%202023-navy)
* The Learnability of In-Context Learning (Wies et al., 2023).
  [[Paper]](https://openreview.net/pdf?id=f3JNQd7CHM#:~:text=We%20use%20our%20framework%20in,are%20unchanged%20and%20the%20input)
  ![](https://img.shields.io/badge/NeurIPS%202023-purple)
* A Latent Space Theory for Emergent Abilities in Large Language Models (Jiang., 2023).
  [[Paper]](https://arxiv.org/abs/2304.09960)
* What and How does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization (Zhang
et al., 2023).
  [[Paper]](https://arxiv.org/abs/2305.19420)
* Bayesian Model Selection and Model Averaging (Wasserman., 2000).
  [[Journal]](https://api.semanticscholar.org/CorpusID:11273095)
* In-Context Learning through the Bayesian Prism (Panwar et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2306.04891)
  ![](https://img.shields.io/badge/ICLR%202024-green)  
* What can transformers learn in-context? a case study of simple function classes (Garg et al., 2022).
  [[Paper]](https://arxiv.org/abs/2208.01066)
  ![](https://img.shields.io/badge/NeurIPS%202022-purple)
* What learning algorithm is in-context learning? Investigations with linear models (Akyürek et al., 2023).
  [[Paper]](https://openreview.net/pdf?id=0g0X4H8yN4I)
  ![](https://img.shields.io/badge/ICLR%202023-green)
* An Information-Theoretic Analysis of In-Context Learning (Jeon et al., 2024).
  [[Paper]](https://openreview.net/pdf?id=NQn2tYLv5I)
  ![](https://img.shields.io/badge/ICML%202024-navy)
* In-Context Learning Dynamics with Random Binary Sequences (Bigelow et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2310.17639)
  ![](https://img.shields.io/badge/ICLR%202024-green)
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
  ![](https://img.shields.io/badge/NeurIPS%202021-purple)
* Impact of Pretraining Term Frequencies on Few-Shot Reasoning (Razeghi et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2202.07206)
  ![](https://img.shields.io/badge/Findings_Of_EMNLP%202022-brown)
* Large Language Models Struggle to Learn Long-Tail Knowledge (Kandpal et al., 2023).
  [[Paper]](https://proceedings.mlr.press/v202/kandpal23a.html)
  ![](https://img.shields.io/badge/PMLR%202023-brown)
* Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression (Raventós et al., 2023).
  [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e10b2c2e1aa4f8083c37dfe269873f8-Abstract-Conference.html)
  ![](https://img.shields.io/badge/NeurIPS%202023-purple)
* Data distributional properties drive emergent in-context learning in transformers. (Chan et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2205.05055)
  ![](https://img.shields.io/badge/NeurIPS%202022-purple)
* THE PRINCIPLE OF LEAST EFFORT (ZIPF et al., 1949).
  [[Paper]](https://wli-zipf.upc.edu/pdf/zipf49-toc.pdf)
* Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models (Yadlowsky et al., 2023).
  [[Paper]](https://wli-zipf.upc.edu/pdf/zipf49-toc.pdf)
* What can transformers learn in-context? a case study of simple function classes. (Garg et al., 2022).
  [[Paper]](https://arxiv.org/abs/2208.01066)
  ![](https://img.shields.io/badge/NeurIPS%202022-purple)
* In-Context Learning Creates Task Vectors. ( Hendel et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2310.15916)
  ![](https://img.shields.io/badge/NeurIPS%202022ffff-purple)
  

### Pre-training Model
* Emergent abilities of large language models (Wei et al., 2022).Hoffmann  [[Paper]](https://openreview.net/pdf?id=yzkSU5zdwD)
  ![](https://img.shields.io/badge/TMLR%202022-navy)
* Training Compute-Optimal Large Language Models (Hoffmann et al., 2022).
  [[Paper]](https://arxiv.org/pdf/2203.15556)
  ![](https://img.shields.io/badge/NeurIPS%202022-purple)
* Are Emergent Abilities of Large Language Models a Mirage? (Schaeffer et al., 2022).
  [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf)
  ![](https://img.shields.io/badge/NeurIPS%202023-purple)
* UL2: Unifying Language Learning Paradigms (Tay et al., 2022).
  [[Paper]](https://arxiv.org/pdf/2205.05131)
  ![](https://img.shields.io/badge/ICLR%202023-navy)
* GENERAL-PURPOSE IN-CONTEXT LEARNING BY META-LEARNING TRANSFORMERS (Kirsch et al., 2022).
  [[Paper]](https://arxiv.org/pdf/2212.04458)
  ![](https://img.shields.io/badge/NeurIPS_Workshop%202022-purple)
* In-Context Language Learning: Architectures and Algorithms (Akyürek et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2401.12973)
  ![](https://img.shields.io/badge/ICML%202024-navy)
* In-context Learning and Induction Heads (Olsson et al., 2022).
  [[Paper]](https://arxiv.org/pdf/2209.11895)
  


### Demonstration Order
* Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity (Lu et al., 2022).
  [[Paper]](https://aclanthology.org/2022.acl-long.556.pdf)
  ![](https://img.shields.io/badge/ACL%202022-brown)
* addressing order sensitivity of in-context demonstration examples in causal language models (Xiang et al., 2024).
  [[Paper]](https://aclanthology.org/2024.findings-acl.386/)
  ![](https://img.shields.io/badge/Findings_Of_ACL%202024-brown)
* Calibrate Before Use: Improving Few-shot Performance of Language Models (Zhao et al., 2021).
  [[Paper]](https://proceedings.mlr.press/v139/zhao21c.html)
  ![](https://img.shields.io/badge/PMLR%202021-brown)
* Lost in the Middle: How Language Models Use Long Contexts (Zhao et al., 2024).
  [[Paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630)
  ![](https://img.shields.io/badge/TACL%202024-brown)
* What Makes Good In-Context Examples for GPT-3? (Liu et al., 2022).
  [[Paper]](https://arxiv.org/abs/2101.06804)
  ![](https://img.shields.io/badge/DeeLIO%202022-brown)

  

### Input-Label Mapping
* Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? (Min et al., 2022).
  [[Paper]](https://arxiv.org/pdf/2202.12837)
  ![](https://img.shields.io/badge/EMNLP%202022-brown)
* Ground-Truth Labels Matter: A Deeper Look into Input-Label Demonstrations (Yoo et al., 2022).
  [[Paper]](https://arxiv.org/pdf/2205.12685)
  ![](https://img.shields.io/badge/EMNLP%202022-brown)
* Larger language models do in-context learning differently (Wei et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2303.03846)
* In-Context Learning Learns Label Relationships but Is Not Conventional Learning (Kossen et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2307.12375)
  ![](https://img.shields.io/badge/ICLR%202024-brown)
* What In-Context Learning “Learns” In-Context: Disentangling Task Recognition and Task Learning (Pan et al., 2023).
  [[Paper]](https://www.proquest.com/openview/6417558c5f0a0d6840ee9442822ab099/1?pq-origsite=gscholar&cbl=18750&diss=y)
* Dual Operating Modes of In-Context Learning (Lin et al., 2024).
  [[Paper]](https://arxiv.org/pdf/2402.18819)
  ![](https://img.shields.io/badge/PMLR%202024-brown)
* Large Language Models Can be Lazy Learners: Analyze Shortcuts in In-Context Learning (Tang et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2305.17256)
  ![](https://img.shields.io/badge/Findings_Of_ACL%202023-brown)
* Measuring Inductive Biases of In-Context Learning with Underspecified Demonstrations (Si et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2305.13299)
  ![](https://img.shields.io/badge/ACL%202023-brown)
* Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning (Si et al., 2023).
  [[Paper]](https://arxiv.org/pdf/2305.14160)
  ![](https://img.shields.io/badge/EMNLP%202023-brown)



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
