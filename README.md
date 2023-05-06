## mBEST: minimal Bending Energy Skeleton pixel Traversals

---

*mBEST* is a robust, realtime perception algorithm for deformable linear object (DLO) detection. The algorithm takes as input a binary mask of the DLOs from an image and produces the ordered pixel sequences as well as segmentation masks for each unique DLO.

For the dataset provided in this repo, simple color filtering is used to achieve the binary masks.
For those interested in using *mBEST* for more complex scenes, please use the DCNN model provided by [FASTDLO](https://github.com/lar-unibo/fastdlo) to obtain the initial binary mask.

<p align="center">
<img src="figures/pipeline.png" alt>
<br>
<em> Fig. 1 mBEST Pipeline Overview </em>
</p>

---

### Instructions

All code has been developed and tested on Python 3.6 and Python 3.8. Please install the following dependencies.
```
numpy
numba
opencv-python
matplotlib
scikit-image
scikit-learn
Cython
```

Afterwards, compile functions using the shell script as shown below.
```bash
./install.sh
```

Once all installation steps have been finished, run *mBEST* through the provided python script as shown below.
The simple background with complex DLO configurations dataset used in the manuscript is provided in the `dataset` directory along with ground truth labels in the form of numpy arrays.
```bash
python3 run.py dataset/S3/images/img0.jpg
```

---

Below are some results comparing *mBEST* with *Ariadne+*, *FASTDLO*, and *RT-DLO*.

<p align="center">
<img src="figures/complex_bg_comparison.png" alt>
<img src="figures/simple_bg_comparison.png" alt>
<br>
<em> Fig. 2 mBEST and SOTA comparison </em>
</p>

***

### Citation
If our work has helped your research, please cite the following manuscript.
```
@misc{choi2023mbest,
      title={mBEST: Realtime Deformable Linear Object Detection Through Minimal Bending Energy Skeleton Pixel Traversals}, 
      author={Andrew Choi and Dezhong Tong and Brian Park and Demetri Terzopoulos and Jungseock Joo and Mohammad Khalid Jawed},
      year={2023},
      eprint={2302.09444},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```