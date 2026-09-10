# Hello brain

Run `flybrain run examples/hello_brain.py` from the checkout root. It uses the
synthetic 10-neuron fixture and requires no network or data download.

For the actual public connectome, first `flybrain pull malecns`, then replace
`FlyBrain("synthetic")` with `FlyBrain("malecns")` in the script.
