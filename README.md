# NER_Adaptive_Resampling

## Resampling Functions and Their Implementation

For resampling functions mentioned in our paper, please see NER_Adaptive_Resampling.py.

  For the purpose of using BUS(Balanced UnderSampling) method, using the following code:

  	NER_Adapative_Resampling(inputpath, outputpath)
  	NER_Adapative_Resampling.BUS()

  For using our methods(sc, sCR, sCRD, nsCRD), please copy the following codes:

  	NER_Adapative_Resampling(inputpath, outputpath)
  	NER_Adapative_Resampling.resamp(one_of_this_method)

## Loss Functions for Shallow Model

  For this part, see shallow_loss_functions.py for details.

  As the LogisticRegression function in sklearn is well-packaged, we make modification on its original 
  loss function part instead of altering the organization of code dramatically.
  Please follow comments in the .py file to switch Focal Loss/Dice Loss.

 
## Loss Functions for Deep Model

  For this part, see deep_loss_functions.py for details.

  These functions are based on Tensorflow 1.12.0.

## Main Bodies of Models

  Links concerning models we used in experiments:

  Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
	
  Conditional Random Field: https://sklearn-crfsuite.readthedocs.io/en/latest/
	
  Bi-LSTM: https://github.com/guillaumegenthial/sequence_tagging
	
  BERT: https://github.com/kyzhouhzau/BERT-NER

## Datasets

  Links concerning dataset we used:
	
  CoNLL 2003: https://www.kaggle.com/alaakhaled/conll003-englishversion
	
  GMB Subset: https://www.kaggle.com/shoumikgoswami/annotated-gmb-corpus
	
  AnEM: https://github.com/juand-r/entity-recognition-datasets/tree/master/data/AnEM
	
  WNUT2017: https://github.com/leondz/emerging_entities_17
	
  OntoNotes 5.0: https://catalog.ldc.upenn.edu/LDC2013T19
