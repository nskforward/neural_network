package main

import "log"

func main() {
	net := NewNet()
	net.AddLayer(4)
	net.AddLayer(3)
	net.AddLayer(2)
	net.AddLayer(1)
	net.Print()

	log.Println("finished")
}
