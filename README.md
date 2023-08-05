# glove-840B-leveldb

This repository contains the code to serve the GloVe word embeddings (https://nlp.stanford.edu/projects/glove/) from a leveldb database for use with FaVe.

## Usage

### Docker
```
docker build -t onepeerlabs/glove-840b-leveldb .
```

```
docker run -d \
    -p 9876:9876 \
    onepeerlabs/glove-840b-leveldb
```

## API Specification

### TODO

## Thanks

Thanks to the authors of GloVe, Jeffrey Pennington, Richard Socher, and Christopher D. Manning, for making their embeddings available.