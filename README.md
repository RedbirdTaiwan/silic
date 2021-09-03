# SILIC 
## Sound Identification and Labeling Intelligence for Creatures
![SILIC](./model/silic_logo_full.svg)

## Goal
The goal of SILIC is to build an autonomous wildlife sound identification system which can help to monitoring the population status and trends of terrestrial vocal animals in Taiwan by using the data of Passive Acoustic Monitorings (PAMs).

## Objects
 - Object 1: Extract robust species, sound class, time and frequency information from various and complex soundscape recordings.
 - Object 2: Model can be constructed using a dataset as small as possible, and the training audios can be easily and quickly acquired.
 - Object 3: Most species of terrestrial vocal wildlife in Taiwan should be included in model, especially those are hard to be detected with survey methods other than PAM.

## Model
SILIC uses [Python](https://www.python.org/) language and [yolov5 package (Glenn Jocher et al., 2020)](https://github.com/ultralytics/yolov5) to construct a object detection model. Additional [pydub (Robert, 2011)](https://github.com/jiaaro/pydub), [nnAudio (Cheuk et al., 2020)](https://github.com/KinWaiCheuk/nnAudio) and [matplotlib (Hunter, 2007)](https://matplotlib.org/) libraries were imported for audio signal processing and Time–Frequency Representation (TFR).

## Datasets
 - Training and validation: [./dataset/Training_Validation_Dataset.txt](./dataset/Training_Validation_Dataset.txt)
 - Test with evaluation results: [./dataset/evaluation_testset.csv](./dataset/evaluation_testset.csv)

## Tutorials
 - Model Weights:
   -  [./model/exp12](./model/exp12) , including 27 sound classes of 16 species, updated on Apr. 2021
   -  [./model/exp14](./model/exp14) , including 74 sound classes of 52 species, updated on Jul. 2021
 - Scripts of detection: [./silic.ipynb](./silic.ipynb)

## Audio Sources
 - [Macaulay Library](https://www.macaulaylibrary.org/)
 - [xeno-canto](https://www.xeno-canto.org/)
 - [Asian Soundscape](https://soundscape.twgrid.org/)
 - Thinning Forest Monitoring
 - [iNaturalist](https://www.inaturalist.org/)

## Publication

