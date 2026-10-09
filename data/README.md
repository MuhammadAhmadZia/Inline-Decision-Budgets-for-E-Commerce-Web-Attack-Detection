# Dataset

This repository does not redistribute CSIC 2010. The notebook downloads it at run time.

## Source

HTTP DATASET CSIC 2010, created by Carmen Torrano Gimenez, Alejandro Perez Villegas and
Gonzalo Alvarez Maranon at the Information Security Institute of the Spanish National
Research Council (CSIC). The dataset is HTTP traffic generated against an e-commerce web
application with a shopping cart, user registration and payment pages.

Original page: https://www.isi.csic.es/dataset/

## What the notebook downloads

The notebook fetches the three original plain-text files from a public GitHub mirror, so
no account or credentials are needed:

https://raw.githubusercontent.com/msudol/Web-Application-Attack-Datasets/master/OriginalDataSets/csic_2010/

| File | Expected size (bytes) |
| --- | --- |
| normalTrafficTraining.txt | 20,148,988 |
| normalTrafficTest.txt | 20,151,204 |
| anomalousTrafficTest.txt | 15,734,523 |

The notebook checks each size after download and warns if one does not match.

## What the study uses

Following common practice with this dataset, the study combines the 36,000 normal
training requests with the 25,065 anomalous requests, giving 61,065 requests. The
separate file of 36,000 normal test requests is downloaded but not used, and is
available for future threshold validation.

## Duplicates

A request counts as a duplicate when its method, full URL and body are byte-identical to
another request after parsing. By that rule 35,457 rows (58.1%) are duplicates and 25,608
unique requests remain. Removing them raises the attack share from 41.0% to 62.3%,
because normal requests repeat far more often than attacks do. The deduplicated set is
the primary setting in the paper.
