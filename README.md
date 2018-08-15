# Benchmarking Probabilistic Programming Languages

Code used in the scope of a Master thesis submitted to

Technische Universität Berlin

School IV - Electrical Engineering and Computer Science

Department of Database Systems and Information Management

###### Supervised by:
Prof. Dr. Volker Markl

Alexander Renz-Wieland


## Original environment

* PyStan version 2.17.1.0
* PyMC3 version 3.4.1 
* Edward version 1.3.5

* TensorFlow version 1.6
* Theano version 1.0.2
* scikit-learn version 0.19.2

All experiments are conducted on Google Cloud Datalab version 1.2.20180713, on a virtual machine instance with 16 vCPUs (1vCPU = single hypercore of Intel(R) Xeon(R) CPU @ 2.30GHz) and 72 GB of RAM. 

## Algorithms 

* NUTS
* ADVI
* Gibbs sampler
* Metropolis-Hastings
* Hamiltonian/Hybrid Monte Carlo

## Experiments
* One Way Normal [1]
* D-Dimensional Hypershpere (Donut)
* Gaussian Mixture Model
* Logistic Regression for Credit Card Fraud Detection [2]


## References

[1] Betancourt, Michael J. and Girolami, Mark. Hamiltonian Monte Carlo for Hierarchical Models. 2013.

[2] Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
