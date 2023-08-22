package main

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"unicode"

	"github.com/onepeerlabs/glove-840B-leveldb/pkg"
	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/opt"
)

// Vectorizer returns vectorized text
type Vectorizer struct {
	db        *leveldb.DB
	stopWords map[string]int
}

var (
	// basic English stopwords
	stopWords = []string{
		"the",
		"an",
		"of",
		"in",
		"and",
		"to",
		"was",
		"is",
		"for",
		"on",
		"as",
		"a",
		"b",
		"c",
		"d",
		"e",
		"f",
		"g",
		"h",
		"i",
		"j",
		"k",
		"l",
		"m",
		"n",
		"o",
		"p",
		"q",
		"r",
		"s",
		"t",
		"u",
		"v",
		"w",
		"x",
		"y",
		"z",
	}
	v *Vectorizer
)

// initDB initializes the LevelDB database in read-only mode
func initDB(dbPath string) (*leveldb.DB, error) {
	opts := &opt.Options{
		ReadOnly: true,
	}
	db, err := leveldb.OpenFile(dbPath, opts)
	if err != nil {
		return nil, err
	}
	return db, nil
}

func main() {
	dbPath := os.Getenv("LEVELDB_PATH")
	if dbPath == "" {
		dbPath = "./embeddings"
	}

	portStr := os.Getenv("VECTORIZER_PORT")
	port, err := strconv.Atoi(portStr)
	if err != nil || port <= 0 {
		port = 9876 // Default port if not provided or invalid
	}

	if dbPath == "" {
		log.Fatal("LevelDB path is required. Use -dbpath flag to provide it.")
	}

	db, err := initDB(dbPath)
	if err != nil {
		log.Fatal(err)
	}

	defer func() {
		db.Close()
	}()
	stopWordsMap := map[string]int{}
	for _, word := range stopWords {
		stopWordsMap[word] = 1
	}

	v = &Vectorizer{db: db, stopWords: stopWordsMap}

	http.HandleFunc("/health", v.healthHandler)
	http.HandleFunc("/vectorize", v.vectorizeHandler)

	fmt.Printf("Server listening on port %d...\n", port)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", port), nil))
}

func (*Vectorizer) healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func (vtcrzr *Vectorizer) vectorizeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var requestBody map[string][]string

	err := json.NewDecoder(r.Body).Decode(&requestBody)
	if err != nil {
		http.Error(w, "Failed to decode request body "+err.Error(), http.StatusBadRequest)
		return
	}

	queryStrings, ok := requestBody["query"]
	if !ok {
		http.Error(w, "Missing 'query' field in request body", http.StatusBadRequest)
		return
	}

	vectorized, err := vtcrzr.Corpi(queryStrings)
	if err != nil {
		http.Error(w, "Failed to vectorize "+err.Error(), http.StatusBadRequest)
		return
	}

	responseBody := make(map[string][]float32)
	responseBody["vector"] = vectorized.ToArray()
	response, err := json.Marshal(responseBody)
	if err != nil {
		http.Error(w, "Failed to send response "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(response)
}

func split(corpus string) []string {
	return strings.FieldsFunc(corpus, func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	})
}

func (vtcrzr *Vectorizer) Corpi(corpi []string) (*pkg.Vector, error) {
	var (
		corpusVectors []pkg.Vector
		err           error
	)
	for i, corpus := range corpi {
		parts := split(corpus)
		if len(parts) == 0 {
			continue
		}

		corpusVectors, err = vtcrzr.vectors(parts)
		if err != nil {
			return nil, fmt.Errorf("at corpus %d: %v", i, err)
		}
	}
	if len(corpusVectors) == 0 {
		return nil, fmt.Errorf("no vectors found for corpus")
	}

	vector, err := computeCentroid(corpusVectors)
	if err != nil {
		return nil, err
	}

	return vector, nil
}

func computeCentroid(vectors []pkg.Vector) (*pkg.Vector, error) {
	var occr = make([]uint64, len(vectors))

	for i := 0; i < len(vectors); i++ {
		occr[i] = uint64(102)
	}
	weights, err := occurrencesToWeight(occr)
	if err != nil {
		return nil, err
	}

	return ComputeWeightedCentroid(vectors, weights)
}

func occurrencesToWeight(occs []uint64) ([]float32, error) {
	max, min := maxMin(occs)

	weigher := makeLogWeigher(min, max)
	weights := make([]float32, len(occs))
	for i, occ := range occs {
		res := weigher(occ)
		weights[i] = res
	}

	return weights, nil
}

func maxMin(input []uint64) (max uint64, min uint64) {
	if len(input) >= 1 {
		min = input[0]
	}

	for _, curr := range input {
		if curr < min {
			min = curr
		} else if curr > max {
			max = curr
		}
	}

	return
}

func makeLogWeigher(min, max uint64) func(uint64) float32 {
	return func(occ uint64) float32 {
		// Note the 1.05 that's 1 + minimal weight of 0.05. This way, the most common
		// word is not removed entirely, but still weighted somewhat
		return float32(2 * (1.05 - (math.Log(float64(occ)) / math.Log(float64(max)))))
	}
}

func ComputeWeightedCentroid(vectors []pkg.Vector, weights []float32) (*pkg.Vector, error) {

	if len(vectors) == 0 {
		return nil, fmt.Errorf("can not compute centroid of empty slice")
	} else if len(vectors) != len(weights) {
		return nil, fmt.Errorf("can not compute weighted centroid if len(vectors) != len(weights)")
	} else if len(vectors) == 1 {
		return &vectors[0], nil
	} else {
		vectorLen := vectors[0].Len()

		var newVector = make([]float32, vectorLen)
		var weightSum float32 = 0.0

		for vectorI, v := range vectors {
			if v.Len() != vectorLen {
				return nil, fmt.Errorf("vectors have different lengths", v.Len(), vectorLen)
			}

			weightSum += weights[vectorI]
			vector := v.ToArray()
			for i := 0; i < vectorLen; i++ {
				newVector[i] += vector[i] * weights[vectorI]
			}
		}

		for i := 0; i < vectorLen; i++ {
			newVector[i] /= weightSum
		}

		result := pkg.NewVector(newVector)
		return &result, nil
	}
}

func (vtcrzr *Vectorizer) getVectorForWord(word string) (*pkg.Vector, error) {
	if _, ok := vtcrzr.stopWords[strings.ToLower(word)]; ok {
		return nil, nil
	}
	var value []byte
	value, err := vtcrzr.db.Get([]byte(word), nil)
	if errors.Is(err, leveldb.ErrNotFound) {
		value, err = vtcrzr.db.Get([]byte(strings.ToLower(word)), nil)
		if err != nil {
			return nil, nil
		}
	}

	vector := make([]float32, 300)
	err = gob.NewDecoder(bytes.NewBuffer(value)).Decode(&vector)
	if err != nil {
		return nil, err
	}
	v := pkg.NewVector(vector)

	return &v, nil
}

func (vtcrzr *Vectorizer) vectors(words []string) ([]pkg.Vector, error) {
	vectors := make([]pkg.Vector, len(words))
	for wordPos := 0; wordPos < len(words); wordPos++ {
		vector, err := vtcrzr.getVectorForWord(words[wordPos])
		if err != nil {
			return nil, err
		}
		if vector != nil {
			// this compound word exists, use its vector and occurrence
			vectors[wordPos] = *vector
		}
	}

	finalVectors := []pkg.Vector{}
	for _, v := range vectors {
		if v.Len() > 0 {
			finalVectors = append(finalVectors, v)
		}
	}
	return finalVectors, nil
}
