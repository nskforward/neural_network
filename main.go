package main

import (
	"github.com/nskforward/neural_network/nn"
)

func main() {
	net := nn.NewNet(2)
	net.AddLayout(2, nn.FuncLeakyReLu())
	net.AddLayout(1, nn.FuncLeakyReLu())

	trainingSet := [][]float64{
		[]float64{0, 0},
		[]float64{1, 0},
		[]float64{0, 1},
		[]float64{1, 1},
	}

	expectedSet := [][]float64{
		[]float64{0},
		[]float64{1},
		[]float64{1},
		[]float64{0},
	}
	nn.TrainNetBP(net, 0.1, 0.001, trainingSet, expectedSet)
	net.Test(trainingSet, expectedSet)
	net.Print()
}
