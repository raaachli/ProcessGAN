# ProcessGAN: Generating Privacy-Preserving Time-Aware Process Data with Conditional GANs
Synthetic process data generator using GAN-based network. Our ProcessGAN model can generate process data from random noise, which can preserve the privacy and augment the datasaet.

## Model Description
- **Model type:** A transformer encoder as generator, and a time-based self-attention model as discriminator.
- **Training data:** Process data (event logs) that contains sequence of activities, and the corresponding timestamps.
- **Evaluation approach:** We trained an off-the-shelf model to score the synthetic sequences. We also used different statistical metrics to evaluate the performance.

## Getting Started

### Requirements
* Python 3.10
* Pytorch 1.13.1
* Matplotlib 

### Usage
#### Training
To train our model and test the performance:
```bash
python run.py --mode [MODE] --model [MODEL] --data [DATASET]
```
_**--mode [MODE]**_: Replace [MODE] with an integer from 1, 2, 3, or 4. 

1: Vanilla ProcessGAN without auxillary losses.

2: ProcessGAN with activity-based loss.

3: ProcessGAN with time-based loss.

4: ProcessGAN with both auxillary losses.

_**--model [MODEL]**_: Replace [MODEL] between "trans_attn" and "trans".

"trans_attn": Our ProcessGAN model with transformer-based generator and time-based discriminator.

"trans": Baseline GAN model using transformer-encoder as both generator and discriminator.

_**--data [DATASET]**_: Replace [DATASET] with "SEP", where we used the [public sepsis process dataset](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639).

#### Evaluation
The result will be generated at every 100 epoch, including the activity sequence and the timestamp differential sequence.

The statistical evaluation will also be automatically generated at every 100 epoch.

To train the off-the-shelf classifier, run _off_the_shelf_classifier_time.py_ under 'eval/' folder.

## Contact
If you have any questions, feel free to contact the author at kl734@scarletmail.rutgers.edu.

## Acknowledgments
The implementation of the discriminator is modified from [SASRec](https://github.com/JiachengLi1995/TiSASRec). Thanks the authors for the great work.