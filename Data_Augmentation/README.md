# Data Augmentation for Named Entity Recoginition

This repo provides a convenient script to generate augmented dataset according to methods mentioned in 

> Xiang Dai and Heike Adel. 2020. An Analysis of Simple Data Augmentation for Named Entity Recognition. In COLING, Online.

Please see https://github.com/boschresearch/data-augmentation-coling2020 for more details.

## Usage

For using our script, please see the script called 'augment_script'

Run the following code:

> Augment_data(inputpath,outputpath, aug_method)

Set input and output paths as you need. Augmentation method must be selected between ['SR', 'LwTR', 'MR', 'SiS'], which corresponds to [synonym_replacement, replace_token, replace_mention, shuffle_within_segments]. See the paper or the 'augment.py' for more details.

For the selection of augmentation method, we chose the one with best performance in their paper, and applied it in our experiments.

