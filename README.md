# Course-design-of-signal-processing
Course design of signal processing, by jineng han, 2020-06-10

## Requirement
Required python libraries: Tensorflow with GPU support (>=1.4) + Scipy (>=1.1) + Numpy (>=1.14) + Tqdm (>=4.0.0). To install in your python distribution, run   
`pip install -r requirements.txt`   
Required software (for resampling): [SoX](http://sox.sourceforge.net/)   
To convert `audiofile.wav` to 32-bit floating-point audio at 16kHz sampling rate, run:   
`sox audiofile.wav -r 16000 -b 32 -e float audiofile-float.wav`   

## Quick start(testing)
If you just want to test the model, please download the default validation data by running:   
`./data/download_sedata_onlyval.sh`   
then, run:   
`python ./code/senet_infer.py`   
The denoised files will be stored in the folder _dataset/valset_noisy\_denoised/_, with the same name as the corresponding source files in _dataset/valset_noisy/_.   

## Train denoising network
### Prepare dataset
The dataset can be automatically downloaded and pre-processed (i.e. resampled at 16kHz) by running the script   
`./data/download_sedata.sh`   
To download only the testing data, you can run the reduced script:   
`./data/download_sedata_onlyval.sh`   
### Dataset structure
-dataset
   - _trainset\_noisy/_ (for the noisy speech training files), 
   - _trainset\_clean/_ (for the ground truth clean speech training files), 
   - _valset\_noisy/_ (for the noisy validation files), and 
   - _valset\_clean/_ (for the noisy validation files).
### Training with default parameters
Once you've downloaded in the script download_data.sh, you can directly train a model using the training dataset by running   
`python ./code/senet_train.py`   
The trained model will be stored in the pretrained folder with the names _se\_model.ckpt.*_.

## Train feature loss network
### Prepare dataset
Downloading and pre-processing (i.e., downsampling to 16kHz) the corresponding data can be done by running the script:     `./data/download_lossdata.sh`   
### Train with default parameters
Once the data is downloaded, you can (re-)train a deep feature loss model by running:    
`python ./code/lossnet_train.py`   
The loss model is stored in the pretrained folder by default. A custom output directory for loss model can be specified as:    
`python ./code/lossnet_train.py -o out_folder`   

