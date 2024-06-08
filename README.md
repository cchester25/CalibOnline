# CalibOnline
This method aims to provide an online miscalibration detection and correction scheme for calibrated multi-sensor systems that require long-term operation.

Multisensor fusion is increasingly used in robotic systems due to its improvement in system robustness and accuracy, whitch also poses great challenges for Multisensor calibration. Existing techniques are mainly implemented offline and with the help of targets, which cannot cope with extrinsic perturbation caused by vibrations and deformations while the system is running. In this letter, we present CalibOnline, a novel method for online detecting and correcting extrinsic perturbation. First, this letter introduces a unified data modality for representing LiDAR and cameras, i.e., the depth map, and a robust feature that this modality possesses, i.e., depth discontinuous edges, is explored. Secondly, the effect of extrinsic perturbation on the edge-matching constraints is analyzed and accordingly the miscalibration probabilities are designed to supervise the extrinsic. Finally, the perturbation correction is described as a problem of on-manifolds optimization, which enhances the convergence of the estimated extrinsic. The experimental results in different datasets and scenarios have demonstrated that the proposed method has high robustness and accuracy.

![The pipeline of CalibOnline](https://github.com/cchester25/CalibOnline/blob/main/pipeline.gif)
