## Running the Code

All execution scripts are available in the `scripts/` directory and can be used to reproduce the experimental results reported in the ICASSP’26 paper.

Before running, make sure to update the `DATA` path and execute all commands from the root directory `BiMoRS/`.

### Base-to-New Class Generalization

These experiments correspond to Section 4.1 (Table 1).

You will need the scripts `scripts/base2new_train.sh` and `scripts/base2new_test.sh`. The training script learns the model using base classes, while the testing script evaluates the trained model on novel classes. Both scripts take two arguments: `DATASET` and `SEED`.

`DATASET` should be one of the dataset configuration names found in `BiMoRS/configs/datasets/` (e.g., `patternnet`, `rsicd`).

Example commands for running experiments (three random seeds shown below):

```bash
# seed=1
bash scripts/cocoop/base2new_train.sh patternnet 1
bash scripts/cocoop/base2new_test.sh patternnet 1

# seed=2
bash scripts/cocoop/base2new_train.sh patternnet 2
bash scripts/cocoop/base2new_test.sh patternnet 2

# seed=3
bash scripts/cocoop/base2new_train.sh patternnet 3
bash scripts/cocoop/base2new_test.sh patternnet 3
```

After completing all runs, the directory structure will look like:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

To compute the average accuracy on the base classes, run:

```bash
python parse_test_res.py output/base2new/train_base/imagenet/shots_16/BiMoRS/vit_b16_c4_ep10_batch1_ctxv1
```

To compute the average accuracy on the novel classes, run:

```bash
python parse_test_res.py output/base2new/test_new/imagenet/shots_16/BiMoRS/vit_b16_c4_ep10_batch1_ctxv1 --test-log
```

### Cross-Dataset Evaluation

These results correspond to Section 4.2 (Table 2).

Training is performed using `scripts/crossdata_train.sh`, where the source dataset is fixed to `patternnet`. Run training with different random seeds:

```bash
# seed=1
bash scripts/crossdata_train.sh 1

# seed=2
bash scripts/crossdata_train.sh 2

# seed=3
bash scripts/crossdata_train.sh 3
```

After training, evaluate the learned model on other datasets using:

```bash
for SEED in 1 2 3
do
    bash scripts/crossdata_test.sh rsicd ${SEED}
    bash scripts/crossdata_test.sh resisc45 ${SEED}
    bash scripts/crossdata_test.sh mlrsnet ${SEED}
done
```

### Domain Generalization

These experiments correspond to Section III (Table III).

The procedure is identical to the cross-dataset setting above, except that evaluation is carried out on the domain-shifted version of PatternNet, namely `patternnetv2`.
