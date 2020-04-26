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
		mse1 := trainEpochBP(net, speed, inputSet, expectSet)
		if mse1 == 0 {
			if attempts == 10 {
				fmt.Println("fail, learning speed too hight")
				break
			}
			net.ResetWeights()
			attempts++
			continue
		}
		fmt.Printf("#%d: %f\n", i, mse)
		if mse1 < precision {
			fmt.Printf("training successfully finished with %d weight reset\n", attempts)
			break
		}
		if mse1 > 0.2 && i > 100000 {
			fmt.Println("fail, please correct speed or add more neurons")
			break
		}
		if mse1 == mse {
			fmt.Println("fail, please correct speed or add more neurons")
			break
		}
		mse = mse1
	}
}

func trainEpochBP(net *Net, speed float64, inputSet, expectSet [][]float64) float64 {
	var sum float64
	output := make([]float64, 8)
	layer := net.GetLastLayout()
	for i, input := range inputSet {
		output = net.Calculate(input, output)
		sum += mse(output, expectSet[i])
		for j, o := range output {
			edev := o - expectSet[i][j]
			if !correctWeightsBP(layer.neurons[j], edev, speed) {
				return 0
			}
		}
	}
	return sum / float64(len(inputSet))
}

func correctWeightsBP(n *Neuron, edev float64, speed float64) bool {
	dw := edev * fdx(n.out)
	for i, w := range n.weight {
		n.weight[i] = w - n.input[i]*dw*speed
		if n.weight[i] > 3 || n.weight[i] < -3 {
			return false
		}
		if n.layout.prev != nil {
			edev1 := n.weight[i] * dw
			correctWeightsBP(n.layout.prev.neurons[i], edev1, speed)
		}
	}
	return true
}
