package fasttext

// #cgo LDFLAGS: -L${SRCDIR}/fastText/lib -lfasttext-wrapper -lstdc++ -lm -pthread

import "C"

func (idx *faissIndex) Search(x []float32, k int64) (distances []float32, labels []int64, err error) {
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	if c := C.faiss_Index_search(idx.idx, C.idx_t(n), (*C.float)(&x[0]), C.idx_t(k), (*C.float)(&distances[0]), (*C.idx_t)(&labels[0])); c != 0 {
		err = getLastError()
	}
	return
}
