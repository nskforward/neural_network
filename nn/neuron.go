package nn

import (
	"fmt"
	"log"
)

// Neuron .
type Neuron struct {
	neuronID   int
	layoutID   int
	links      int
	sum        float64
	out        float64
	weight     []float64
	input      []float64
	layout     *Layout
	activation ActivationFunc
	wrange     []float64
}

// ActivationFunc .
type ActivationFunc interface {
	fx(x float64) float64
	fdx(x float64) float64
}

// NewNeuron .
func NewNeuron(neuronID, layoutID, links int, lay *Layout, f ActivationFunc) *Neuron {
	n := &Neuron{
		links:      links + 1,
		weight:     make([]float64, links+1),
		input:      make([]float64, links+1),
		neuronID:   neuronID,
		layoutID:   layoutID,
		layout:     lay,
		activation: f,
	}
	n.initWeights()
	return n
}

// InitWeights .
func (n *Neuron) initWeights() {
	for i := range n.weight {
		n.weight[i] = RandomSimple()
		//n.weight[i] = RandomNormal()
	}
}

// Activate .
func (n *Neuron) Activate(input []float64) {
	if len(input) != len(n.weight)-1 {
		fmt.Println("input:", input)
		fmt.Println("weights:", n.weight)
		log.Fatalf("[%d:%d] mismatch input length (got %d, expected %d)", n.layoutID, n.neuronID, len(input), len(n.weight)-1)
	}
	copy(n.input, input)
	n.input[len(n.input)-1] = 1
	n.sum = 0
	for i, w := range n.weight {
		n.sum += n.input[i] * w
	}
	n.out = n.activation.fx(n.sum)
}

func (n *Neuron) String() string {
	return fmt.Sprintf("%d:%d %v", n.layoutID, n.neuronID, n.weight)
}

// Fdx .
func (n *Neuron) Fdx() float64 {
	return n.activation.fdx(n.out)
}
