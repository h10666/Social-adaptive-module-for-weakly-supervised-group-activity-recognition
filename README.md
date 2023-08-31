# Social adaptive module for weakly-supervised group activity recognition.

We give a general **DMS**(**D**ata, **M**odel, **S**olver) code framework for PCTDM, impelemented by Pytorch. You can apply new model or dataset into this framework by modifying the files in `Configs` easily! For further information about me, welcome to my [homepage](https://ruiyan1995.github.io/).


## Requirements
&gt; Ubuntu 16.04  
&gt; pytorch 0.4.1  
= python 2.7  
`pip install dlib`

## The general piplines of GAR
You can run `python GAR.py` to excute all the following steps.
### Step Zero: Preprocessing dataset
- To download [VD](https://github.com/mostafa-saad/deep-activity-rec#dataset) and [BD](https://ruiyan1995.github.io/SAM.html) at './dataset/VD' and './dataset/CAD' folder;
- Add `none.jpg`
- To track the persons and generate the train/test files by using **Processing.py**;

### Step One: Action Level
- run this project via:

&nbsp;&nbsp;`python script_VD.py`;
&nbsp;&nbsp;`python script_BD.py`;



## License and Citation 
Please cite the following paper in your publications if it helps your research.

@inproceedings{yan2018participation,  
&nbsp;&nbsp;&nbsp;&nbsp;title={Social adaptive module for weakly-supervised group activity recognition},  
&nbsp;&nbsp;&nbsp;&nbsp;author={Yan, Rui and Xie, Lingxi and Tang, Jinhui and Shu, Xiangbo and Tian, Qi},  
&nbsp;&nbsp;&nbsp;&nbsp;booktitle={European Conference on Computer Vision},  
&nbsp;&nbsp;&nbsp;&nbsp;pages={208--224},  
&nbsp;&nbsp;&nbsp;&nbsp;year={2020},  
&nbsp;&nbsp;&nbsp;&nbsp;organization={Springer}  
}

## Contact Information
Feel free to create a pull request or contact me by Email = ["ruiyan", at, "njust", dot, "edu", dot, "cn"], if you find any bugs. 
