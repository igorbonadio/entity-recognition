# Entity Recognition

TODO

## Requirements

* Python 3
* `sh setup-env.sh`

## Training

To train, just run:

```bash
python train.py
```

## API

To build the API, run:

```bash
docker build -t entity-recognition .
```

To start serving:


```bash
docker run --rm -p 8080:8080 entity-recognition
```

It should start the API on port 8080. To test it, follow the link below:

[http://localhost:8080/predict?text=O%20Brasil%20%C3%A9%20o%20maior%20pa%C3%ADs%20da%20America%20Latina](http://localhost:8080/predict?text=O%20Brasil%20%C3%A9%20o%20maior%20pa%C3%ADs%20da%20America%20Latina)


## References

* [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF by Ma et Hovy](https://arxiv.org/abs/1603.01354)
* [Guillaume Genthial's implementation](https://github.com/guillaumegenthial/tf_ner)