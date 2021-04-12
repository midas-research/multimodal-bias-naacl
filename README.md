# An Empirical Investigation of Bias in the Multimodal Analysis of Financial Earnings Calls

This codebase contains the python scripts for An Empirical Investigation of Bias in the Multimodal Analysis of Financial Earnings Calls.

NAACL '21 paper [coming soon](#)

Dataset and Models used for Bias Analysis adapted from MDRM(What You Say and How You Say It Matters:
Predicting Financial Risk Using Verbal and Vocal Cues) [dataset](https://github.com/GeminiLn/EarningsCall_Dataset) | [paper](https://www.aclweb.org/anthology/P19-1038.pdf)

## Environment & Installation Steps

Python 3.6
Keras
Matplotlib
Numpy
Scipy
Seaborn
Pandas
Scikit Learn
Tensorflow

```bash
pip install -r requirements.txt
```

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

Degradation in the performance of speech models could be due to discernible noise and indiscernible sources like  demographic bias:  age,  gender,  dialect, culture, etc ([1], [2], [3]).  Studies also show that AI can deploy biases against black people in criminal sentencing ([5], [6]). Although we only account for the gender bias in our study, we acknowledge that there could exist other kinds of bias due to age, accent, culture, ethnic and regional disparities in audio cues, as the publicly available earnings calls majorly have companies belonging to the US.Moreover, only publicly available earnings calls have been used limiting the scope of the data. This also limits the availability of genders in the data to only male and female.  In the future, we hope to increase the amount of data to expand our study to more categories and types of sensitive attributes.


### References

[1] Josh  Meyer,   Lindy  Rauchenstein,   Joshua  D  Eisenberg,  and  Nicholas  Howell.  2020.   Artie  bias  corpus:   An  open  dataset  for  detecting  demographic bias in speech applications.  In Proceedings of The 12th  Language  Resources  and  Evaluation  Conference, pages 6462–6468.

[2] Tatsunori B Hashimoto, Megha Srivastava, Hongseok Namkoong, and Percy Liang. 2018.   Fairness without  demographics  in  repeated  loss  minimization. arXiv preprint arXiv:1806.08010.

[3]

[4]

[5]

[6]
