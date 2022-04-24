# ASGNN
Codes of paper "ASGNN"

# Datasets
0. Because the data file is too large, we place it on google drive. The address is:https://drive.google.com/drive/folders/1Fs17jeBLEqZ98HFKHXAOGA0qyPadhCBc?usp=sharing
1. directory ```data``` contains three datasets
2. directory ```data_process``` is the data processing
3. references：The original dataset we used is from paper ”Time-aware point-of-interest recommendation“, the download address of this data set is https://www.ntu.edu.sg/home/gaocong/datacode.htm .If you use this dataset, please cite
```
@inproceedings{yuan2013time,
  title={Time-aware point-of-interest recommendation},
  author={Yuan, Quan and Cong, Gao and Ma, Zongyang and Sun, Aixin and Thalmann, Nadia Magnenat-},
  booktitle={Proceedings of the 36th international ACM SIGIR conference on Research and development in information retrieval},
  pages={363--372},
  year={2013}
}
```

# Requirements
- Python 3.6
- networkx 2.5
- tensorflow_gpu 1.15

# Run
```
python main.py
```


# Reference
If you make advantage of the ASGNN model or use the datasets released in our paper, please cite our paper in your manuscript:
```
@article{wang2021attentive,
  title={Attentive sequential model based on graph neural network for next poi recommendation},
  author={Wang, Dongjing and Wang, Xingliang and Xiang, Zhengzhe and Yu, Dongjin and Deng, Shuiguang and Xu, Guandong},
  journal={World Wide Web},
  volume={24},
  number={6},
  pages={2161--2184},
  year={2021},
  publisher={Springer}
}
```
