# GDPR120Q

This respository contains the code for the paper *GDPR120Q: An Annotated Q&A Corpus for Privacy Compliance* (ref. LNAI). 

The dataset is available on the website of the Smart Law Hub: http://gdpr120.smartlawhub.eu/. 

Download the dataset and place it in the `paper_data` folder. The main models checkpoints are available on Zenodo: https://zenodo.org/doi/10.5281/zenodo.10440240

The paper uses the `preprocess.py` script to preprocesses the dataset for the `transformer` library (https://github.com/huggingface/transformers). Models have been trained using the scripts in the `scripts` folder. For example:

```
/bin/bash ./scripts/BERT.sh
```

The `evaluation_cuad.py` file is based on https://github.com/TheAtticusProject/cuad/blob/main/evaluate.py 

Please cite using the following reference:

```
(ref. LNAI).
```

 
