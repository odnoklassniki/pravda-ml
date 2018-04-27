# ok-ml-pipelines
This project is used to define machine learning pipelines on top of Spark. 
This an *extension*, not a replacement, of the Spark ML package with a focus on structural aspects
of distributed machine learning deployments. Core features added by the project are:

* Ability to add "transparent" technical stages to ML pipeline (eg. caching, sampling, repartitioning, etc.) -
these stages are included into learning pipeline, but then automatically excluded from the resulting model 
not to influence inference performance.
* Ability to execute certain pipeline stages in parallel to achieve better cluster utilization - provides 
an order of magnitude improvement for cross-validation, model segmentation, grid search and other ML stages
with external parallelism.
* Ability to collect extra information about the model (learning curve history, weights statistics and etc.)
in a form of DataFrame greatly simplifies analysis of the learning process and helps to identify potential
improvements.
* Improved model evaluation capabilities allowing for extra metrics, including non-scalar (eg. full ROC-curve),
and statistical analysis of the metrics.

In addition to structural improvements there are few ML algorithms incorporated:
* Language detection and preprocessing with a focus on ex-USSR languages.
* LSH-based deduplication for texts.
* Improved distributed implementation of variance reduced SGD.
* Multi-label version of LBFGS with a matrix gradient.
* Feature selection based on the stability of features importance in cross-validation.

Slides available from JBreak 2018 demo: https://cloud.mail.ru/public/77xY/GKAfB3mjn

Set of usage examples available on Zepl:
* Basic stat and ML features: https://www.zepl.com/viewer/notebooks/bm90ZTovL2RtaXRyeWJ1Z2F5Y2hlbmtvL2I4YWUyN2Y4YWMwMzRlNzZiMmQyOWFkZGRhOGIwMWQ1L25vdGUuanNvbg