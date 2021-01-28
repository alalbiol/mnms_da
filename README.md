# Domain Adaptation for M&Ms Challenge

Dataset specifications.

|            | Siemens | Philips | GE | Canon | Total |
|:----------:|:-------:|:-------:|:--:|:-----:|:-----:|
|    Label   |    A    |    B    |  C |   D   |       |
|   Centres  |    1    |  2 & 3  |  4 |   5   |       |
|  Training  |    75   | 50 & 25 | 25 |   0   |  175  |
| Validation |    4    |  5 & 5  | 10 |   10  |   34  |
|   Testing  |    16   | 19 & 21 | 40 |   40  |  136  |
|   Overall  |    95   | 74 & 51 | 75 |   50  |  345  |

Training data for GE Vendor are unlabeled.

### Requirements

!conda install -c conda-forge ffmpeg -y
!conda install tsnecuda -c cannylab -y
Pytoch >= 1.6 

```shell script
export SLACK_TOKEN='you_slack_token'
```

#### Data Preparation
```shell
./scripts/mms2d.sh only_data
./scripts/mms2d.sh only_meta
python3 tools/nifti2slices.py --data_path data/MMs
python3 tools/testgt2phases.py
```

### Guidelines

*MMs dataset naming*:
  - `_full` Get all volumes (not only segmented 'ED' and 'ES' phases volumes).
  - `_unlabeled` Get only unlabeled volumes (for 'ED' and 'ES' phases)
  - `_centre*xyz*` Get volumes (for 'ED' and 'ES' phases) for selected centres. Example `_centre1`, `_centre13`. Last one picks centres 1 and 3. Available Centres from 1 to 5.
  - `_vendor*jkl*` Get volumes (for 'ED' and 'ES' phases) for selected vendors. Example `_centreC`, `_vendorAB`. Last one picks vendors A and B. Available Vendors 'A', 'B', 'C', 'D'.
  - `_all` Get all segmented (which is ED and ES phases) slices (Training Labeled + Training Unlabeled + Validation + Test)

Normalization:
  - none
  - reescale
  - reescale_phase
  - reescale_full_vol
  - standardize
  - standardize_phase
  - standardize_full_vol

**Coral Loss**: The weight of the CORAL loss (λ) is set in such way that at the end of training the
classification loss and CORAL loss are roughly the same. It seems be a reasonable
choice as we want to have a feature representation that is both discriminative
and also minimizes the distance between the source and target domains  

### ToDo

