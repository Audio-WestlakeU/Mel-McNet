# Mel-McNet
[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2505.19576)

The official repo: **Mel-McNet: A Mel-Scale Framework for Online Multichannel Speech Enhancement** accepted by Interspeech 2025.

Code and the pretrained model will be updated soon.

## Introduction
This work proposes a Mel-scale framework for online multichannel speech enhancement, termed Mel-McNet. It proposes spectral and spatial information with two key components: an effective STFT-to-Mel module compressing multichannel STFT features into Mel-frequency representations, and a modified McNet backbone directly operating in the Mel domain to generate enhanced LogMel spectra. The spectra can be directly fed to vocoders for waveform reconstruction or ASR systems for transcription. Experiments on CHiME-3 show that Mel-McNet can reduce computational complexity by 60% while maintaining comparable enhancement and ASR performance to the original McNet. Mel-McNet also outperforms other SOTA methods, verifying the potentional of Mel-scale speech enhancement.

## Performance

## Citation
If you find our work helpful, please cite
```
@misc{yang2025melmcnetmelscaleframeworkonline,
      title={Mel-McNet: A Mel-Scale Framework for Online Multichannel Speech Enhancement}, 
      author={Yujie Yang and Bing Yang and Xiaofei Li},
      year={2025},
      eprint={2505.19576},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.19576}, 
}
```
