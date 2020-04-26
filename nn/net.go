package nn

import (
	"fmt"
	"log"
)

// Net .
type Net struct {
	layouts []*Layout
	inputs  int
}

// NewNet .
func NewNet(inputs int) *Net {
	return &Net{
		inputs:  inputs,
		layouts: make([]*Layout, 0, 2),
	}
}

// AddLayout .
func (net *Net) AddLayout(neurons int) {
	if len(net.layouts) == 0 {
		net.layouts = append(net.layouts, NewLayout(len(net.layouts), neurons, net.inputs, nil))
		return
	}
	net.layouts = append(net.layouts, NewLayout(len(net.layouts), neurons, len(net.GetLastLayout().neurons), net.GetLastLayout()))
}

// Calculate .
func (net *Net) Calculate(input, out []float64) []float64 {
	if net.inputs != len(input) {
		log.Fatalln("input data length must be same as net input length")
	}
	in := make([]float64, 0, 64)
	in = copyArray(in, input)
	//fmt.Println("net input:", in)
	for _, lay := range net.layouts {
		out = lay.Calculate(in, out)
		in = copyArray(in, out)
	}
	return out
}

// Print .
func (net *Net) Print() {
	for _, lay := range net.layouts {
		lay.Print()
	}
}

// GetLastLayout .
func (net *Net) GetLastLayout() *Layout {
	return net.layouts[len(net.layouts)-1]
}

// Test .
func (net *Net) Test(inputSet, expectSet [][]float64) {
	fmt.Println("testing started...")
	output := make([]float64, 8)
	for i, input := range inputSet {
		output = net.Calculate(input, output)
		expect := expectSet[i]
		fmt.Printf("input: %v, output: %v, expect: %v\n", input, output, expect)
	}
	fmt.Println("testing finished")
}

// ResetWeights .
func (net *Net) ResetWeights() {
	for _, lay := range net.layouts {
		for _, n := range lay.neurons {
			n.initWeights()
		}
	}
}
