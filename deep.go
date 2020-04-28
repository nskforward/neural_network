package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

func goDeep() {
	var data = training.Examples{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}
	n := deep.NewNeural(&deep.Config{
		Inputs: 2,
		Layout: []int{3, 1},
		/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
		Activation: deep.ActivationReLU,
		/* Determines output layer activation & loss function:
		ModeRegression: linear outputs with MSE loss
		ModeMultiClass: softmax output with Cross Entropy loss
		ModeMultiLabel: sigmoid output with Cross Entropy loss
		ModeBinary: sigmoid output with binary CE loss */
		Mode: deep.ModeBinary,
		/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
		Weight: deep.NewNormal(1.0, 0.0),
		/* Apply bias */
		Bias: true,
	})

	// params: learning rate, momentum, alpha decay, nesterov
	optimizer := training.NewAdam(0.001, 0.9, 0.999, 1e-8)
	// params: optimizer, verbosity (print stats at every 50th iteration)
	trainer := training.NewTrainer(optimizer, 100)

	//training, heldout := data.Split(0.5)
	trainer.Train(n, data, data, 10000) // training, validation, iterations

	fmt.Println(data[0].Input, "=>", n.Predict(data[0].Input))
	fmt.Println(data[1].Input, "=>", n.Predict(data[1].Input))
	fmt.Println(data[2].Input, "=>", n.Predict(data[2].Input))
	fmt.Println(data[3].Input, "=>", n.Predict(data[3].Input))

	dump, _ := n.Marshal()
	ioutil.WriteFile(filepath.Join(getRootDir(), "setttings.dat"), dump, 0644)

	fmt.Println("finished")
}

func getRootDir() string {
	dir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}
	return dir
}
