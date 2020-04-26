package nn

import (
	"fmt"
	"log"
)

// TrainNetBP .
func TrainNetBP(net *Net, speed float64, precision float64, inputSet, expectSet [][]float64) {
	if len(inputSet) != len(expectSet) {
		log.Fatalln("input set len must be equal expect len")
	}
	var mse float64 = 0
	var i int64
	attempts := 0
	for {
		i++
		if attempts > 20 {
			break
		}
		mse1 := trainEpochBP(net, speed, inputSet, expectSet)
		fmt.Printf("#%d: %.12f\n", i, mse)
		if mse1 == 0 {
			fmt.Printf("###### FAIL ######\nweight size too high\n")
			net.ResetWeights()
			i = 0
			attempts++
			continue
		}
		if mse1 < precision {
			fmt.Printf("###### SUCCESS ######\n")
			break
		}
		if mse1 > 0.2 && i > 300000 {
			fmt.Printf("###### FAIL ######\ntrainig too slow\n")
			net.ResetWeights()
			i = 0
			attempts++
			continue
		}
		if mse1 == mse {
			fmt.Printf("###### FAIL ######\ntraining has no progress\n")
			net.ResetWeights()
			i = 0
			attempts++
			continue
		}
		mse = mse1
	}
	fmt.Println("attempts:", attempts)
}

func trainEpochBP(net *Net, speed float64, inputSet, expectSet [][]float64) float64 {
	var max float64 = 0
	output := make([]float64, 8)
	layer := net.GetLastLayout()
	for i, input := range inputSet {
		output = net.Calculate(input, output)
		mse := mse(output, expectSet[i])
		if mse > max {
			max = mse
		}
		for j, o := range output {
			edev := o - expectSet[i][j]
			if !correctWeightsBP(layer.neurons[j], edev, speed) {
				return 0
			}
		}
	}
	return max
}

func correctWeightsBP(n *Neuron, edev float64, speed float64) bool {
	dw := edev * n.Fdx()
	for i, w := range n.weight {
		n.weight[i] = w - n.input[i]*dw*speed
		if n.weight[i] > 30 || n.weight[i] < -30 {
			return false
		}
		if n.layout.prev != nil {
			edev1 := n.weight[i] * dw
			correctWeightsBP(n.layout.prev.neurons[i], edev1, speed)
		}
	}
	return true
}
