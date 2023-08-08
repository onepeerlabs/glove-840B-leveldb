FROM golang:1.20-alpine AS build_base
RUN apk add bash ca-certificates git gcc g++ libc-dev
WORKDIR /go/src/github.com/onepeerlabs/glove-840B-leveldb
ENV GO111MODULE=on
COPY go.mod .
COPY go.sum .
RUN go mod download

FROM build_base AS server_builder
ARG TARGETARCH
ARG EXTRA_BUILD_ARGS=""
COPY . .
RUN GOOS=linux GOARCH=$TARGETARCH go build $EXTRA_BUILD_ARGS \
      -ldflags '-w -extldflags "-static" ' \
      -o /vectorizer ./cmd/server

FROM alpine AS vectorizer
COPY --from=server_builder /vectorizer /bin/vectorizer
COPY --from=server_builder /go/src/github.com/onepeerlabs/glove-840B-leveldb/embeddings /embeddings
RUN apk add --no-cache --upgrade ca-certificates openssl
ENTRYPOINT ["/bin/vectorizer"]
