import data_loader as loader
import network

train, valid, test = loader.load_data_wrapper()
net = network.Network([784, 30, 50, 10])
net.SGD(train, 50, 30, 0.5, test)

