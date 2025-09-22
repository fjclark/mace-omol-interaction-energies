To run:

Install the environment with:

```
mamba env create -f environment.yml
```

Then, install my bugfixed version of GGNImplicitSolvent with 


```
pip install git+https://github.com/fjclark/GNNImplicitSolvent.git@bugfix-pass-partial-charges
```

and run
```
python dy_hyd_streamlined
```

