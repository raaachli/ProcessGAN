# ProcessGAN: Generating Privacy-Preserving Time-Aware Process Data with Conditional GANs 
This repository is the implementation of the ProcessGAN model, a synthetic process data generator using GAN-based network. 

Process data constructed from event logs provides valuable insights into how processes unfold over time. 
Because process data with confidential information cannot be shared, 
research is limited using process data and analytics in the process mining domain. 
We introduced a synthetic process data generation task to address the limitation of sharable process data. 
We introduced a generative adversarial network, called ProcessGAN, 
to generate process data with activity sequences and corresponding timestamps. 
Our ProcessGAN model can generate process data from random noise, which can preserve the privacy and augment the datasaet.

## 🗒 Model Description
- **Model type:** A transformer encoder as generator, and a time-based self-attention model as discriminator.
- **Training data:** Process data (event logs) that contains sequence of activities, and the corresponding timestamps.
- **Evaluation approach:** We trained an off-the-shelf model to score the synthetic sequences. We also used different statistical metrics to evaluate the performance.

## ✊ Getting Started

### 🖋 Requirements
* Python 3.10
* Pytorch 1.13.1
* Matplotlib 

### ⭐ Usage
#### 🏃‍♀️Training 
To train our model and test the performance:
```bash
python run.py --mode [MODE] --model [MODEL] --data [DATASET]
```
_**--mode [MODE]**_: Choose a training mode based on different loss values to add.

Replace [MODE] with a training mode. 

'vanilla': Vanilla ProcessGAN without auxillary losses.

'act_loss': ProcessGAN with activity-based loss.

'time_loss': ProcessGAN with time-based loss.

'act_time_loss': ProcessGAN with both auxillary losses.

_**--model [MODEL]**_: Choose from different discriminator model.

Replace [MODEL] between "trans_attn" and "trans".

"trans_attn": Our ProcessGAN model with transformer-based generator and time-based discriminator.

"trans": Baseline GAN model using transformer-encoder as both generator and discriminator.

_**--data [DATASET]**_: Select a dataset to train.

Replace [DATASET] with "SEP" or "BPI", which is the [public sepsis process dataset](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639), or [public BPI dataset](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204).

#### 🔍 Evaluation

The evaluation solutions are implemented in 'eval/'. We used four statistical metrics to evaluate the synthetic process:
- Sequence length
- Sequence variance 
- Activity type error
- Activity timestamp error

The statistical evaluation will be automatically generated at every 100 epoch.

To train the off-the-shelf classifier, run _off_the_shelf_classifier_time.py_ under 'eval/' folder.

## 📩 Contact
If you have any questions, feel free to contact the author at kl734@scarletmail.rutgers.edu.

## 📝 Acknowledgments
The implementation of the discriminator is modified from [SASRec](https://github.com/JiachengLi1995/TiSASRec). Thanks the authors for the great work.