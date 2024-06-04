# Using the National Trauma Data Bank (NTDB) and machine learning to predict trauma patient mortality at admission

## Notes

1. NTDB has aproximately 968,665 as of 2016
2. Both categorical and numerical features
3. Incident files are in .csv format that can be used by pandas to construct matrices
4. 351,253 (2016) of the patients contained missing data (unclear how much was missing i.e. was a single feature missing vs. many)
5. Of note, patients with missing data (again unclear how much) had 1.4 times higher death rates than patients without missing data (possible case to look at these patients separately and as part of the greater dataset)
6. This paper imputed missing features if there were 2 or fewer missing (accounted for 181,821 additional patients originally filtered out. Interesting choice to perform this however the death rate of this cohort more closely resembled that of the population of data with complete information. This is not suprising as that would have been the dataset used in training phase but runs great risk of bias.... curious to know how they internally validated these imputation)
7. This dataset is heavily skewed toward patients who survived (Survived = 778,803; Deceased = 20,930 [2016 data])
8. In this paper they used random 85% of deceased patients for training, and took the N of this group of random survived patients to form training split (not sure if I like this as it is not reflective of larger population, curious to know if results didn't work well with survivorship bias. Also throws out a lot of training data (only 35,580 in training set) and uses huge test set (potentially good if sufficient hyperparameter search completed).
9. "A limitation of this model is that it was trained based on the vital signs of patients upon
admission. While the model was accurate, predicting the time-series evolution of the patient
will requires dynamical training data. A natural extension would be to train a model that can
predict patient-risk in real-time if time-series trauma patient data is available.
On the machine learning side, one limitation of our approach was the random undersampling procedure for balancing the number of survived patients with the number of deceased patients in the training set."
10. Overall cool paper, could explore future directions for research if an interesting idea can be thought of for retrospective study with machine learning. Still unsure of what the practical application of this model is unless can be shown to be substantially different than physician ability.