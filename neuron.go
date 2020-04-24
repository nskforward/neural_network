package main

import (
	"fmt"
	"math/rand"
	"time"
)

// Neuron .
type Neuron struct {
	weight []float32
}

// Layer .
type Layer struct {
	neuron []*Neuron
}

// Net .
type Net struct {
	layer []*Layer
}

// NewNet .
func NewNet() *Net {
	rand.Seed(time.Now().UnixNano())
	return &Net{
		layer: make([]*Layer, 0),
	}
}

// AddLayer .
func (net *Net) AddLayer(neurons int) {
	links := 1
	if len(net.layer) > 0 {
		links = len(net.layer[len(net.layer)-1].neuron)
	}
	layer := &Layer{
		neuron: make([]*Neuron, neurons),
	}
	for i := 0; i < neurons; i++ {
		layer.neuron[i] = &Neuron{
			weight: make([]float32, links),
		}
	}
	for _, n := range layer.neuron {
		for i := range n.weight {
			n.weight[i] = rand.Float32()
		}
	}
	net.layer = append(net.layer, layer)
}

// Print .
func (net *Net) Print() {
	for i, layer := range net.layer {
		fmt.Println()
		switch i {
		case 0:
			fmt.Printf("input:\t")
		case len(net.layer) - 1:
			fmt.Printf("output:\t")
		default:
			fmt.Printf("hidden:\t")
		}
		fmt.Printf("(%d)", len(layer.neuron))
		for _, n := range layer.neuron {
			fmt.Print(" [")
			for ii, w := range n.weight {
				if ii == 0 {
					fmt.Print(w)
				} else {
					fmt.Printf(" %f", w)
				}
			}
			fmt.Print("]")
		}
	}
	fmt.Printf("\n\n")
}
