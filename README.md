# JargonTTS
### Demo: In progress 

## Description: 
JargonTTS is an end to end TTS pipeline that can be trained and finetuned to generate speech for jargon in any particular field. The data extraction was done using jargon from popular AI research publications and the speech samples (approximately 2 seconds in average) were generated using SpeechT5. These samples will be given to train a transformer based TTS model. (Taken inspiration from:  https://medium.com/@tttzof351/build-text-to-speech-from-scratch-part-1-ba8b313a504f, the model implementation is a work in progress). 

## Architecture Diagram
Work in progress

## Steps to run the data extraction pipeline
As of 9th Feb 2025, only the data extraction pipeline has been tested end to end. To get it working please follow the following steps: 
1. Import the repository in your local environment (assuming you have an adequately sufficient GPU, if not the same can be done in colab).
2. Setup a virtual env (not needed if you're doing it in colab) and install the necessary dependencies in requirements.txt
3. Run `utils.py`
4. The audio files will be added to the `audio_src` directory.







