import data_loader as loader
import network

train, valid, test = loader.load_data_wrapper()
net = network.Network([784, 100, 10])
net.SGD(train, 50, 100, 0.1, test)

