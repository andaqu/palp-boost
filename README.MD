# PALP-BOOST

Personality-Aware Link Predictor Boosting, a system to be infused with other topological link predictors to optimise their performance; taking advantage of (computationally recognised) personalities.

## Folder directory structure

```
palp
├── data
│   ├── network (*)
│   ├── normalised_twitter_personality.csv
│   ├── raw_twitter_personality.csv
|   └── stats.csv (*)
├── gae
├── predictors
├── saved
├── analyser.py
├── helper.py
├── manager.py
└── palpboost.py
```

> `(*)` not included in repository.

* `data`:
  * `network`: Dataset containing Twitter network, obtained from `[1]`. It should contain a `raw` and a `filtered` folder, the former is the raw ego-network edgelists obtained from the aformentioned source; while the latter is a filtered version, where a node is kept if it's found to have a recognised personality.
  * `normalised_twitter_personality.csv`: Personality file for the Twitter network, normalised between [0-1] for each trait.
  * `raw_twitter_personality.csv`: Personality file for the Twitter network, as obtained from the trained models.
  * `stats.csv`: Enumerated following edges, enlisted by personality - used for SPSS analysis.
* `gae`: TensorFlow implementation of the (Variational) Graph Auto-Encoder model, used for pre-processing and splitting. Please refer to `[2]`.
* `predictors`: As of 09/05, 5 Link Predictors are incoporated (AdamicAcar, JaccardCoefficient, PreferentialAttachment, SpectralClustering, Node2Vec). Please refer to `[3]`.
* `saved`: Saved graph objects, train-val-test splits, and calculated centroids containing each personality tendency given the train split for some graph. Filenames are split as follows `twitter-{N}-{connected_components}-{p}` where:
  * `N` = it is a subset containing `1/N` of the entire graph.
  * `connected_components` = which kind of connected components are considered (this can be 'strong', 'weak' or nothing at all, to which that would be indicate that it does not filter by components).
  * `p` = it is a split, where (p*100)% is the percentage of the test data.
* `analyser.py`: Analyses personality dimensions, followee tendencies, degree count and PALPBoost score effectiveness.
* `helper.py`: Helper functions.
* `manager.py`: To execute code from.
* `palpboost.py`: PALPBoost, containing calls to `helper.py` to construct the needed centroids, reflecting personality tendencies.

`[1]:` https://snap.stanford.edu/data/ego-Twitter.html \
`[2]:` https://github.com/tkipf/gae \
`[3]:` https://github.com/lucashu1/link-prediction