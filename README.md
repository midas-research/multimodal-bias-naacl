# An Empirical Investigation of Bias in the Multimodal Analysis of Financial Earnings Calls

This codebase contains the python scripts for An Empirical Investigation of Bias in the Multimodal Analysis of Financial Earnings Calls.

NAACL '21 paper [coming soon](#)

Dataset and Models used for Bias Analysis adapted from MDRM(What You Say and How You Say It Matters:
Predicting Financial Risk Using Verbal and Vocal Cues) [dataset](https://github.com/GeminiLn/EarningsCall_Dataset) | [paper](https://www.aclweb.org/anthology/P19-1038.pdf)

## Environment & Installation Steps

Python 3.6
Keras 2.4.3
Librosa 0.8.0
Matplotlib 3.3.2
Numpy 1.19.2
Pandas 1.1.3
Parselmouth 1.1.1
Scikit Learn 0.23.2
Scipy 1.5.2
Seaborn 0.11.0
Tensorflow 2.4.1
Tqdm 4.50.2

```bash
pip install -r requirements.txt
```
## Dataset Details

#### Audio feature extraction

  The Praat[5] based audio features can be extracted using the following command. This script requires the original audio recordings of the MDRM [dataset](https://github.com/GeminiLn/EarningsCall_Dataset) the sample for which can be seen at ./OGdataset

  ```bash
  python audio_feat_extraction_praat.py
  ```
  This will store 18 of the audio features at the path ./data/audio_featDict.pkl and the remaining at ./data/audio_featDictMark2.pkl
  The samples for these files have been added. Each .pkl file contains a dictionary of dictionaries of for all the segmented audio recordings.

#### Textual feature extraction

  Text features have been extracted using FinBERT[code](https://github.com/ProsusAI/finBERT)[6]. Each feature vector is 786 dimensional whose sample can be seen at ./data/finbert_earnings.pkl

#### Gender data

  The speakers are mapped from all the earnings calls to their self declared genders. The sample can be seen  at ./data/genders.pkl. For this we perform web scrapping from [Reuters](https://www.thomsonreuters.com/en/profiles.html) (pronouns), [Crunchbase](https://www.crunchbase.com/discover/people) where the genders are self-declared and the available genders from the Wikidata API.

#### Dataset splits

  The train, test and validation splits are performed temporally to avoid predicting on past data. The splits are performed according to [MDRM](https://www.aclweb.org/anthology/P19-1038.pdf). The sample data can be seen at ./data/*.csv
  Each csv contains the following columns : ticker, name, year, month, day, future_3, future_7, future_15, future_30, text_file_name, past_3, past_7, past_15, past_30

## Run

Execute the following steps in the same environment

For audio feature statistical analysis:

```bash
python audio_feat_stat_analysis.py
```
For analysis of the error disparity in performance:

```bash
python bias_in_ec.py
```
For analysis of the error disparity by varying the male:female ratio in the training set:

```bash
python train_dist_curves.py
```

## Cite

If our work was helpful in your research, please kindly cite this work:

```
@inproceedings{sawhney2021multimodalbias,
  title={An Empirical Investigation of Bias in the Multimodal Analysis of Financial Earnings Calls},
  author={
    Sawhney, Ramit and
    Aggarwal, Arshiya and
    Shah, Rajiv Ratn
  },
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics - Human Language Technologies},
  year={2021}
}
```

## Ethical Considerations

Degradation in the performance of speech models could be due to discernible noise and indiscernible sources like  demographic bias:  age,  gender,  dialect, culture, etc ([1], [2], [3]).  Studies also show that AI can deploy biases against black people in criminal sentencing ([4], [3]). Although we only account for the gender bias in our study, we acknowledge that there could exist other kinds of bias due to age, accent, culture, ethnic and regional disparities in audio cues, as the publicly available earnings calls majorly have companies belonging to the US.Moreover, only publicly available earnings calls have been used limiting the scope of the data. This also limits the availability of genders in the data to only male and female.  In the future, we hope to increase the amount of data to expand our study to more categories and types of sensitive attributes.


### References

[1] Josh  Meyer,   Lindy  Rauchenstein,   Joshua  D  Eisenberg,  and  Nicholas  Howell.  2020.   Artie  bias  corpus:   An  open  dataset  for  detecting  demographic bias in speech applications.  In Proceedings of The 12th  Language  Resources  and  Evaluation  Conference, pages 6462–6468.

[2] Tatsunori B Hashimoto, Megha Srivastava, Hongseok Namkoong, and Percy Liang. 2018.   Fairness without  demographics  in  repeated  loss  minimization. arXiv preprint arXiv:1806.08010.

[3] Rachael Tatman and Conner Kasten. 2017. Effects of talker dialect, gender & race on accuracy of bing speech and youtube automatic captions. In INTERSPEECH, pages 934–938.

[4] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. 2016. Machine Bias: There’s software used across the country to predict future criminals. And it’s biased against blacks. [Online; accessed 23-May-2016].

[5] Paul Boersma and Vincent Van Heuven. 2001. Speak and unspeak with praat. Glot Int, 5:341–347.

[6] Dogu Araci. 2019. Finbert: Financial sentiment analysis with pre-trained language models. arXiv preprint arXiv:1908.10063.
