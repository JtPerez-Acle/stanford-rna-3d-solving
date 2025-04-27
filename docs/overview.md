[https://www.kaggle.com/competitions/stanford-rna-3d-folding]

RNA is vital to life’s most essential processes, but despite its significance, predicting its 3D structure is still difficult. Deep learning breakthroughs like AlphaFold have transformed protein structure prediction, but progress with RNA has been much slower due to limited data and evaluation methods.

This competition builds on recent advances, like the deep learning foundation model RibonanzaNet, which emerged from a prior Kaggle competition. Now, you’ll take on the next challenge—predicting RNA’s full 3D structure.

Your work could push RNA-based medicine forward, making treatments like cancer immunotherapies and CRISPR gene editing more accessible and effective. More fundamentally, your work may be the key step in illuminating the folds and functions of natural RNA molecules, which have been called the 'dark matter of biology'.

This competition is made possible through a worldwide collaborative effort including the organizers, experimental RNA structural biologists, and predictors of the CASP16 and RNA-Puzzles competitions; Howard Hughes Medical Institute; the Institute of Protein Design; and Stanford University School of Medicine.

Submissions are scored using TM-score ("template modeling" score), which goes from 0.0 to 1.0 (higher is better):

TM-score=max⎛⎝⎜⎜1Lref∑i=1Lalign11+(did0)2⎞⎠⎟⎟

where:

    Lref is the number of residues solved in the experimental reference structure ("ground truth").

    Lalign is the number of aligned residues.

    di is the distance between the ith pair of aligned residues, in Angstroms.

    d0 is a distance scaling factor in Angstroms, defined as:

    d0=0.6(Lref−0.5)1/2−2.5

for Lref ≥ 30; and d0 = 0.3, 0.4, 0.5, 0.6, or 0.7 for Lref <12, 12-15, 16-19, 20-23, or 24-29, respectively.

The rotation and translation of predicted structures to align with experimental reference structures are carried out by US-align. To match default settings, as used in the CASP competitions, the alignment will be sequence-independent.

For each target RNA sequence, you will submit 5 predictions and your final score will be the average of best-of-5 TM-scores of all targets. For a few targets, multiple slightly different structures have been captured experimentally; your predictions' scores will be based on the best TM-score compared to each of these reference structures.
Submission File

For each sequence in the test set, you can predict five structures. Your notebook should look for a file test_sequences.csv and output submission.csv. This file should contain x, y, z coordinates of the C1' atom in each residue across your predicted structures 1 to 5:

ID,resname,resid,x_1,y_1,z_1,... x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,... -7.301,9.023,8.932
R1107_2,G,1,-8.02,11.014,14.606,... -7.953,10.02,12.127
etc.

You must submit five sets of coordinates.

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

    CPU Notebook <= 8 hours run-time
    GPU Notebook <= 8 hours run-time
    Internet access disabled
    Freely & publicly available external data is allowed, including pre-trained models
    Submission file must be named submission.csv
    Submission runtimes have been slightly obfuscated. If you repeat the exact same submission you will see up to 5 minutes of variance in the time before you receive your score.
