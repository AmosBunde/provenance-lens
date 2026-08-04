# Provenance Lens: evaluation report

Frozen test split; every interval is a 1000-resample bootstrap at 95%.

## Results

| track | accuracy | macro F1 | AUROC | parse failures |
|---|---|---|---|---|
| baseline | 0.9547 [0.9517, 0.9579] | 0.9547 [0.9517, 0.9579] | 0.9854 [0.9837, 0.9872] | 0.0000 |
| reasoner | absent | absent | absent | absent |


## Calibration

Temperature 2.904 fitted on validation only; baseline test ECE 0.0344 before scaling, 0.0024 after. Reliability diagrams alongside.

## Findings

1. The tuned baseline stands at macro F1 0.9547 on the frozen test split with AUROC 0.9854; the validation-to-test gap of +0.0003 recorded at freeze time indicates no overfitting in selection.
2. Negative result, stated plainly: the baseline fails on the copy_move stratum with within-type accuracy 0.00 (equivalently, copy-move recall); it labels essentially every copy-moved image authentic. The strong aggregate is a cifake separability number, not evidence of manipulation detection in general, and copy-move localization is precisely where grounded regional evidence gives the reasoner a real opportunity.
3. The reasoner comparison is pending an external input: batch inference needs API credentials and a spend decision, so the headline question (does grounded reasoning beat the classifier) remains open and unclaimed. No number here should be read as an answer to it.
4. Per-type numbers inherit the corpus confound (types nested in sources) and the copy_move stratum holds only 109 test assets, so its intervals are wide by construction; claims on that slice stay proportionate to those intervals. See breakdown artifacts.
5. Temperature scaling improves baseline calibration (ECE 0.0344 to 0.0024); the temperature was fitted on validation only, enforced in code.
