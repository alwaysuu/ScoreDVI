# Abstract

Real-world single image denoising is crucial and practical in computer vision. Bayesian inversions combined with score priors now have proven effective for single image denoising but are limited to white Gaussian noise. Moreover, applying existing score-based methods for real-world denoising requires not only the explicit train of score priors on the target domain but also the careful design of sampling procedures for posterior inference, which is complicated and impractical. To address these limitations, we propose a score priors-guided deep variational inference, namely ScoreDVI, for practical real-world denoising. By considering the deep variational image posterior with a Gaussian form, score priors are extracted based on easily accessible minimum MSE Non-$i.i.d$ Gaussian denoisers and variational samples, which in turn facilitate optimizing the variational image posterior. Such a procedure adaptively applies cheap score priors to denoising. Additionally, we exploit a Non-$i.i.d$ Gaussian mixture model and variational noise posterior to model the real-world noise. This scheme also enables the pixel-wise fusion of multiple image priors and variational image posteriors. Besides, we develop a noise-aware prior assignment strategy that dynamically adjusts the weight of image priors in the optimization. Our method outperforms other single image-based real-world denoising methods and achieves comparable performance to dataset-based unsupervised methods.

# Dependency
* Pytorch 3.9
* Scipy
* Skimage

# Usage
Refer to ScoreDVI.py and the denoising function
> python ScoreDVI.py

# Reference

```
@inproceedings{cheng2023score,
  title={Score priors guided deep variational inference for unsupervised real-world single image denoising},
  author={Cheng, Jun and Liu, Tao and Tan, Shan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12937--12948},
  year={2023}
}
```
