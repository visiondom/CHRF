# Where to Focus: Investigating Hierarchical Attention Relationship for Fine-Grained Visual Classification
This repository hosts the source code of our paper: [[ECCV 2022] Where to Focus: Investigating Hierarchical Attention Relationship for Fine-Grained Visual Classification](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840056.pdf). We propose a CrossHierarchical Region Feature (CHRF) learning framework to leverage human attention mechanism for fine-grained visual
classification. A dataset of Attention Reinforced Images on Species TaxonOmy (ARISTO) is collected to record human gaze data for human attention monitoring.

## Collected human gaze

The `.pkl` files in the `dataset` directory save the human gaze point coordinates on images of CUB when performing hierachycal fine-grained classification tasks.

For each image, we present
```
'img_id': the index of the image in CUB, refer to dataset/images.txt
'times': timestamp
'x' and 'y': coordinates of gaze points in a 1280x720 image
'hierarchy': current category hierarchy 
'label': category label in current hierarchy 
``` 
We present a visualization tool `dataset/visual_human_gaze.py` for further reference.

## Training

Pick one configuration file you like in `configs`, and run with it.

```
python tools/train_net.py --config-file configs/${config file} --num-gpus 1
```

**Note**: The overall code framework is borrowed from [detectron2](https://github.com/facebookresearch/detectron2).


## Test

Similar to training, run
```
python tools/train_net.py --config-file configs/${config file} --num-gpus 1 --resume --eval-only
```

## Citation

```
@inproceedings{liu2022focus,
  title={Where to Focus: Investigating Hierarchical Attention Relationship for Fine-Grained Visual Classification},
  author={Liu, Yang and Zhou, Lei and Zhang, Pengcheng and Bai, Xiao and Gu, Lin and Yu, Xiaohan and Zhou, Jun and Hancock, Edwin R},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXIV},
  pages={57--73},
  year={2022},
  organization={Springer}
}
```