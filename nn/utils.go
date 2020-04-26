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

func fdx(x float64) float64 {
	if x < 0 {
		return 0.01
	}
	return 1
}

func fx(x float64) float64 {
	if x < 0 {
		return 0.01 * x
	}
	return x
}

func mse(actual, target []float64) float64 {
	var sum float64
	for i, n := range actual {
		sum += math.Pow(n-target[i], 2)
	}
	return sum / float64(len(actual))
}

func randomWeight(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
