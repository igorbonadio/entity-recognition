# Entity Recognition

TODO

## Requirements

* Python 3
* `sh setup-env.sh` will download and instal the required python libs

## Training

```bash
python train.py
```

## API

To start the API, run:

```
python api.py
```

It will start the API on port 8080. To test it, follow the link below:

[http://localhost:8080/predict?text=teste](http://localhost:8080/predict?text=teste)


## References

* [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF by Ma et Hovy](https://arxiv.org/abs/1603.01354)
* [Guillaume Genthial's implementation](https://github.com/guillaumegenthial/tf_ner)