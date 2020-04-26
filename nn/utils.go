package nn

import (
	"math"
	"math/rand"
)

func copyArray(dst, src []float64) []float64 {
	dst = dst[:0]
	for _, v := range src {
		dst = append(dst, v)
	}
	return dst
}

func mse(actual, target []float64) float64 {
	var sum float64
	for i, n := range actual {
		sum += math.Pow(n-target[i], 2)
	}
	return sum / float64(len(actual))
}

// RandomSimple .
func RandomSimple() float64 {
	return -0.5 + rand.Float64()
}

// RandomNormal .
func RandomNormal() float64 {
	return rand.NormFloat64() * 0.1
}
