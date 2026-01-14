# Telugu-ASR-Wav2Vec2.0-and-Whisper

## Overview

This project implements an Automatic Speech Recognition (ASR) system for the Telugu language, leveraging two state-of-the-art models: **Wav2Vec2.0** and **Whisper**. The goal is to create a robust ASR system capable of transcribing Telugu speech across various accents, dialects, and noisy environments.

## Key Features

- **Wav2Vec2.0**: A model from Facebook AI that learns representations of speech audio to enable highly accurate transcriptions.
- **Whisper**: A multi-lingual ASR model that can transcribe speech in multiple languages, including Telugu.
- **Custom Configuration**: The model uses optimized configurations that were found during experimentation. You are welcome to experiment with these configurations to improve performance or adjust based on your requirements.

## Requirements

- Python 3.7 or higher
- `transformers` library (for model loading)
- `torch` (PyTorch for deep learning)
- `torchaudio` (for audio processing)
- `librosa` (for feature extraction)
- `numpy` and `pandas` (for data handling)
- `huggingface_hub` (for model downloading from Hugging Face)



## Model Setup

This system uses **Wav2Vec2.0** and **Whisper** for transcribing audio into Telugu text. Both models are pre-trained and fine-tuned on a large dataset of Telugu speech samples.

### Wav2Vec2.0

Wav2Vec2.0 is fine-tuned for speech transcription and has been optimized for better performance on low-resource languages like Telugu. We have utilized the best configuration based on performance tests conducted on the dataset.

### Whisper

Whisper is another model leveraged for cross-lingual ASR. It provides better transcription across different dialects and noise conditions in comparison to traditional models.

## Dataset

The model was trained and evaluated using the following datasets:

- **OpenSLR Telugu Speech Corpus**: This dataset contains diverse speech samples with a wide range of speakers, including regional dialects, ensuring that the model can generalize across various speech conditions.
- **IndicTTS Telugu Dataset**: A high-quality, annotated dataset for training the ASR system with clear and labeled speech recordings in Telugu.

## Configuration

I have optimized the configurations based on my experiments. These configurations include:
- **Sampling rate**: Set to match the characteristics of the dataset.
- **Feature extraction**: Parameters that control the MFCC extraction and normalization.
- **Model hyperparameters**: Fine-tuning the learning rate, batch size, and other model-specific parameters.

Feel free to change these configurations based on your requirements. For example, you may want to try different batch sizes, learning rates, or audio pre-processing methods to further optimize the system for your needs.




## Conclusion

This ASR system for Telugu is powered by **Wav2Vec2.0** and **Whisper**, two of the most advanced ASR models available today. With the optimized configurations in place, the system provides high-quality transcription, making it suitable for real-world applications in speech-to-text technology for the Telugu language.

Feel free to modify and improve the system as needed. Contributions, suggestions, and feedback are always welcome!


