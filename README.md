# Non-Sequential Graph Script Induction via Multimedia Grounding
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.5-yellow)


**[Non-Sequential Graph Script Induction via Multimedia Grounding](https://aclanthology.org/2023.acl-long.303.pdf)**<br>
Yu Zhou, Sha Li, Manling Li, Xudong Lin, Shih-Fu Chang, Mohit Bansal and Heng Ji <br>
ACL 2023 <br>

## Abstract
Online resources such as WikiHow compile a wide range of scripts for performing everyday tasks, which can assist models in learning to reason about procedures. However, the scripts are always presented in a linear manner, which does not reflect the flexibility displayed by people executing tasks in real life. For example, in the CrossTask Dataset, 64.5% of consecutive step pairs are also observed in the reverse order, suggesting their ordering is not fixed. In addition, each step has an average of 2.56 frequent next steps, demonstrating "branching". In this paper, we propose the new challenging task of non-sequential graph script induction, aiming to capture optional and interchangeable steps in procedural planning. To automate the induction of such graph scripts for given tasks, we propose to take advantage of loosely aligned videos of people performing the tasks. In particular, we design a multimodal framework to ground procedural videos to WikiHow textual steps and thus transform each video into an observed step path on the latent ground truth graph script. This key transformation enables us to train a script knowledge model capable of both generating explicit graph scripts for learnt tasks and predicting future steps given a partial step sequence. Our best model outperforms the strongest pure text/vision baselines by 17.52% absolute gains on F1@3 for next step prediction and 13.8% absolute gains on Acc@1 for partial sequence completion. Human evaluation shows our model outperforming the WikiHow linear baseline by 48.76% absolute gains in capturing sequential and non-sequential step relationships.


![](./media/out.png)


## Reproduce
### Requirements

The models in this paper are runnable on a single Nvidia V-100 GPU and CUDA Version: 12.0. <br><br>
Please see [environment.yml](https://github.com/bryanzhou008/Multimodal-Graph-Script-Learning/blob/main/environment.yml) for specific package requirements.<br><br>

---

### Training

For pre-training our models on HowTo100M, please refer to [pretrain.py](https://github.com/bryanzhou008/Multimodal-Graph-Script-Learning/blob/main/src/pretrain.py).<br><br>
For finetuning models on CrossTask, please refer to  [finetune.py](https://github.com/bryanzhou008/Multimodal-Graph-Script-Learning/blob/main/src/finetune.py).<br><br>
The corresponding data can be downloaded from their respective webpages: [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) and [CrossTask](https://github.com/DmZhukov/CrossTask).<br><br>

---

### Evaluation

To generate probablistic schema graphs as shown in [outputs](https://github.com/bryanzhou008/Multimodal-Graph-Script-Learning/tree/main/sample_output), please refer to [graph.py](https://github.com/bryanzhou008/Multimodal-Graph-Script-Learning/blob/main/src/graph.py).<br><br>

For evaluating trained models on Next Step Prediction and Partial Sequence Completion, please refer to [next_step_prediction.py](https://github.com/bryanzhou008/Multimodal-Graph-Script-Learning/blob/main/src/next_step_prediction.py) and [partial_sequence_completion.py](https://github.com/bryanzhou008/Multimodal-Graph-Script-Learning/blob/main/src/partial_sequence_completion.py).<br><br>


## Citation

If you find the code in this repo useful, please consider citing our paper:

```
@inproceedings{Zhou2023NonSequential,
   title={Non-Sequential Graph Script Induction via Multimedia Grounding},
   author={Zhou, Yu and Li, Sha and Li, Manling and Lin, Xudong and Chang, Shih-Fu and Bansal, Mohit and Ji, Heng},
   booktitle={ACL},
   year={2023},
}
```



## This repo in currently being updated by the authors, please wait for a stable version, thanks!
