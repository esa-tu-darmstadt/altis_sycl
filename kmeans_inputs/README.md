# Inputs for kmeans

Compared to all other Altis benchmarks, **kmeans** does _not_ offer standardized inputs. This means that the `--size 1/2/3` arguments do not work here. Instead, we use the files in this directory as inputs for our measurements:

- Size 1: `2048_16.txt`
- Size 2: `10000_20.txt`
- Size 3: `65536_32.txt`

These files are obtained from the original (non-SYCL) Altis repository (https://github.com/utcs-scea/altis, directory `data/kmeans`).
