## 2021-09-09T06-23-12_custom_vqgan

### Experiment objective:
This experiment was done in order to evaluate the performance of the tamming VQ-VAE on the CelebA-HQ dataset. As the provided results in the paper showed extreme distortion in the images around the nose and eyes (at 1k, 16f configurations), we though about increasing the codebook size to 8k and the resolution to 8f.

### Experiment results:
The reconstruction seems to be improving by a lot, but still the area around the eyes looks strange for some examples, and the exact shape of the mouth, hair, nose .. etc was not properly reconstructed.