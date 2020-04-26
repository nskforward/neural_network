package main

import (
	"math/rand"
	"time"

	"github.com/nskforward/neural_network/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	net := nn.NewNet(2)
	net.AddLayout(4)
	net.AddLayout(1)

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
