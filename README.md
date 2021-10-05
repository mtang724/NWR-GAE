# NWR-GAE

## Required Packages

Before to execute NWR-GAE, it is necessary to install the following packages using requirments.txt:

`pip install -r requirements.txt`

## Overall Structure

The repository is organised as follows:
- `data/` contains the necessary python files for generating synthetic data;
- `datasets/` contains the necessary dataset files for real-world datasets;
- `edgelists/` contains the necessary dataset files for real-world datasets in edgelist format;
- `src/` contains the implementation of the NW-GAE pipeline (`model.py`) and our training file (`train.py`);
- `src/utils` contains the necessary processing subroutines (`utils.py`).

## Basic Usage
### Support Datasets
**Proximity:** cora, citeseer, pubmed

**Structure:** cornell, texas, wisconsin

**Mixed:** chameleon, squirrel, film (actor)

### Support Parameters
**--dataset**, supported datasets above, default: chameleon

**--lr**, learning rate for neighborhood reconstructor, default: 5e-7

**--epoch_num**, training epoch size, default: 4

**--lambda_loss**, balance weights for degree and neighborhood information decoder, default: 0.1

**--sample_size**, size of neighborhood down sampling, default: 5

**--dimension**, dimension of final output embeddings, default: input feature dimension

### Example

```bash
cd src
python train.py --dataset cornell # real-world datasets
python train.py --dataset_type synthetic # Synthetic datasets
```

, the default setting can run most of the state-of-art results. 