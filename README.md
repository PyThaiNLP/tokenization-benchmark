# Word Tokenisation Benchmark for Thai

<div align="center">
    <img src="https://i.imgur.com/2IuLbyR.png"/>
</div>

## Objective
This repository is a framework for benchmarking tokenisation algorithms for Thai. It has a command-line interface that allows users to conviniently execute the benchmarks as well as a module interface for later use in their development pipelines.

## Metrics
### Character-Level
### Word-Level

## Installation (TBD)
```
pip ...
```

## Usages (to be updated)
1. Command-line Interface 
    ```
    > python ./scripts/tokenisation-benchmark.py \
        --input ./data/best-2010-deepcut.txt \
        --dataset best-2010
    # Sample output
    Benchmarking ./data/best-2010-deepcut.txt against ./data/best-2010/TEST_100K_ANS.txt with 2252 samples in total
    ============== Benchmark Result ==============
                     metric       mean±std       min    max
              char_level:tp    47.82±47.22  1.000000  354.0
              char_level:tn  144.19±145.97  1.000000  887.0
              char_level:fp      1.34±2.02  0.000000   23.0
              char_level:fn      0.70±1.19  0.000000   14.0
       char_level:precision      0.96±0.08  0.250000    1.0
          char_level:recall      0.98±0.04  0.500000    1.0
              char_level:f1      0.97±0.06  0.333333    1.0
        word_level:accuracy      0.95±0.11  0.000000    1.0
    ```
2. Module Interface
    ```
    from pythainlp.benchmarks import word_tokenisation as bwt
    # ref_samples = array of reference tokenised samples
    # tokenised_samples = array of tokenised samples, aka. from your algorithm

    # dataframe contains metrics for each sample
    df = bwt.benchmark(ref_samples, tokenised_samples)
    ```

## Acknowledgements
TBD.