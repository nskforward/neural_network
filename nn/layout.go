package nn

import "fmt"

// Layout .
type Layout struct {
	index   int
	neurons []*Neuron
	prev    *Layout
	next    *Layout
}

// NewLayout .
func NewLayout(index, neurons, links int, prev *Layout) *Layout {
	lay := &Layout{
		neurons: make([]*Neuron, neurons),
		index:   index,
	}
	for i := range lay.neurons {
		lay.neurons[i] = NewNeuron(i, index, links, lay)
	}
	if prev != nil && prev.next == nil {
		prev.next = lay
	}
	return lay
}

// Print .
func (lay *Layout) Print() {
	for _, n := range lay.neurons {
		fmt.Println(n)
	}
}

// Calculate .
func (lay *Layout) Calculate(input []float64, output []float64) []float64 {
	output = output[:0]
	for _, n := range lay.neurons {
		n.Activate(input)
		output = append(output, n.out)
	}
	return output
}
