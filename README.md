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

## References

* [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF by Ma et Hovy](https://arxiv.org/abs/1603.01354)
* [Guillaume Genthial's implementation](https://github.com/guillaumegenthial/tf_ner)