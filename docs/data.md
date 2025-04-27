[https://www.kaggle.com/competitions/stanford-rna-3d-folding/data]
Dataset Description

In this competition you will predict five 3D structures for each RNA sequence.
Competition Phases and Updates

This is a code competition that will proceed in three phases.

    Initial model training phase. At launch, expect approximately 25 sequences in the hidden test set. Some of those sequences are used for a private leaderboard to allow the host to track progress on wholly unseen data. During this phase the public test set sequences includes–but is not limited to–targets from the 2024 CASP16 competition whose structures have not yet been publicly released in the PDB database.
    Model training phase 2. On April 23rd we will update the hidden test set and reset the leaderboard. Sequences in the current public test set will be added to the train data [still TBD], all sequences currently in the private set will be rolled into the new public set, and new sequences will be added to the public test set.
    Future data phase. Your selected submissions will be run against a completely new private test set generated after the end of the model training phases. There will be up to 40 sequences in the test set, all of them used for the private leaderboard.

Files

    [train/validation/test]_sequences.csv - the target sequences of the RNA molecules.
        target_id - (string) An arbitrary identifier. In train_sequences.csv, this is formatted as pdb_id_chain_id, where pdb_id is the id of the entry in the Protein Data Bank and chain_id is the chain id of the monomer in the pdb file.
        sequence - (string) The RNA sequence. For test_sequences.csv, this is guaranteed to be a string of A, C, G, and U. For some train_sequences.csv, other characters may appear.
        temporal_cutoff - (string) The date in yyyy-mm-dd format that the sequence was published. See Additional Notes.
        description - (string) Details of the origins of the sequence. For a few targets, additional information on small molecule ligands bound to the RNA is included. You don't need to make predictions for these ligand coordinates.
        all_sequences - (string) FASTA-formatted sequences of all molecular chains present in the experimentally solved structure. In a few cases this may include multiple copies of the target RNA (look for the word "Chains" in the header) and/or partners like other RNAs or proteins or DNA. You don't need to make predictions for all these molecules; if you do, just submit predictions for sequence. Some entries are blank.

    [train/validation]_labels.csv - experimental structures.
        ID - (string) that identifies the target_id and residue number, separated by _. Note: residue numbers use one-based indexing.
        resname - (character) The RNA nucleotide ( A, C, G, or U) for the residue.
        resid - (integer) residue number.
        x_1,y_1,z_1,x_2,y_2,z_2,… - (float) Coordinates (in Angstroms) of the C1' atom for each experimental RNA structure. There is typically one structure for the RNA sequence, and train_labels.csv curates one structure for each training sequence. However, in some targets the experimental method has captured more than one conformation, and each will be used as a potential reference for scoring your predictions. validation_labels.csv has examples of targets with multiple reference structures (x_2,y_2,z_2, etc.). 

    train_[sequences/labels].v2.csv - extracted from the protein data bank with full text search for keyword RNA relaxed filter for unstructured RNAs based on pairwise C1' distances, where 20% of residues have to be close to some other residue that is over 4 bases apart.

    sample_submission.csv
        Same format as train_labels.csv but with five sets of coordinates for each of your five predicted structures (x_1,y_1,z_1,x_2,y_2,z_2,…x_5,y_5,z_5).
        You must submit five sets of coordinates.

    MSA/ contains multiple sequence alignments in FASTA format for each target, named {target_id}.MSA.fasta. During evaluation with hidden test sequences, your notebook will have access to these MSA files for the test sequences.

Additional notes

    The validation_sequences.csv and test_sequences.csv publicly provided here comprise 12 targets from the 2022 CASP15 competition which have been a widely used test set in the RNA modeling field.
    If you choose to use the provided 12 CASP15 targets in validation_sequences.csv for validation, make sure that you train only on train_sequences.csv that have temporal_cutoff before the test_sequences (2022-05-27 is a safe date). If you wish, you can use train_sequences.csv with temporal_cutoff after this date as an additional validation set.
    Once you begin hill climbing on the competition's actual Public Leaderboard, you can use all the train_sequences.csv and indeed all 3D structural information that you can find in the PDB database, since the competition's actual leaderboard targets are not released in the PDB database. However, note that the 12 CASP15 targets provided here in validation_sequences.csv will be 'burned' since they will be in your training set.
    RNA chains from the same or different PDB entries that share sequence are given as different entries in train_sequences.csv. You may consider deduplicating these entries and merging the various available structures into additional x_2,y_2,z_2, etc. labels, as has been done with validation_sequences.csv
    If you use RibonanzaNet (as in the competition starting notebook) it does not use information from the PDB before CASP15 and so is expected to be valid for use for all test sets. If you are using other neural networks, make sure to check their temporal cutoffs for training data.
    If you are prompting a large language model you should request information that is available before the temporal_cutoff for each target. Otherwise, information from preprints or blog posts on CASP16 targets that were released after CASP16 competition end (2024-09-18) may leak into your submissions, and you will get a Public Leaderboard score that may be deceptively inflated compared to the CASP16 expert baseline or your eventual Private Leaderboard score. Only notebooks that beat the CASP16 expert baseline while also paying close attention to temporal_cutoff will be eligible for the Early Sharing prizes!

Additional files

    The developers of RFdiffusion have made available a synthetic data set of over 400,000 RNA structures here.

Download data: https://www.kaggle.com/datasets/andrewfavor/uw-synthetic-rna-structures
