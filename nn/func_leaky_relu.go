package nn

// LeakyReLu .
type LeakyReLu struct {
	koeff float64
}

// FuncLeakyReLu .
func FuncLeakyReLu() LeakyReLu {
	return LeakyReLu{
		koeff: 0.01,
	}
}

func (f LeakyReLu) fdx(x float64) float64 {
	if x < 0 {
		return f.koeff
	}
	return 1
}

func (f LeakyReLu) fx(x float64) float64 {
	if x < 0 {
		return f.koeff * x
	}
	return x
}
