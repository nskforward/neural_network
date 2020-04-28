package nn

import "math"

// Sigmoid .
type Sigmoid struct{}

// FuncSigmoid .
func FuncSigmoid() Sigmoid {
	return Sigmoid{}
}

func (f Sigmoid) fx(x float64) float64 {
	return 1. / (1. + math.Exp(-x))
}

func (f Sigmoid) fdx(x float64) float64 {
	return f.fx(x) * (1. - f.fx(x))
}
