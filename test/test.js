"use strict";

var FeedforwardNeuralNetwork = require("..");

describe('Feedforward Neural Networks', function () {

    it('Training the neural network with XOR operator', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[0], [1], [1], [0]];

        var xorNN = new FeedforwardNeuralNetwork([2, 4, 1]);
        xorNN.train(trainingSet, predictions, 500, 0.3, 0.3);
        var results = xorNN.predict(trainingSet);

        (results[0]).should.be.approximately(predictions[0], 3e-1);
        (results[1]).should.be.approximately(predictions[1], 3e-1);
        (results[2]).should.be.approximately(predictions[2], 3e-1);
        (results[3]).should.be.approximately(predictions[3], 3e-1);
    });

    it('Training the neural network with AND operator', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[1, 0], [1, 0], [1, 0], [0, 1]];

        var andNN = new FeedforwardNeuralNetwork([2, 3, 2]);
        andNN.train(trainingSet, predictions, 500, 0.3, 0.3);

        var results = andNN.predict(trainingSet);

        (results[0][0]).should.be.greaterThan(results[0][1]);
        (results[1][0]).should.be.greaterThan(results[1][1]);
        (results[2][0]).should.be.greaterThan(results[2][1]);
        (results[3][0]).should.be.lessThan(results[3][1]);
    });

    it('Export and import', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[0], [1], [1], [1]];

        var orNN = new FeedforwardNeuralNetwork([2, 4, 1]);
        orNN.train(trainingSet, predictions, 500, 0.3, 0.3);

        var model = orNN.export();
        var neworNN = FeedforwardNeuralNetwork.load(model);

        var results = neworNN.predict(trainingSet);

        (results[0]).should.be.approximately(predictions[0], 3e-1);
        (results[1]).should.be.approximately(predictions[1], 3e-1);
        (results[2]).should.be.approximately(predictions[2], 3e-1);
        (results[3]).should.be.approximately(predictions[3], 3e-1);
    });

    it('multiclass clasification', function () {
        var trainingSet = [[0, 0], [0, 1], [1, 0], [1, 1]];
        var predictions = [[2], [0], [1], [0]];

        var nn = new FeedforwardNeuralNetwork([2, 4, 1]);
        nn.train(trainingSet, predictions, 300, 0.5, 0.1);

        var result = nn.predict(trainingSet);

        result[0].should.be.approximately(2, 1e-1);
        result[1].should.be.approximately(0, 1e-1);
        result[2].should.be.approximately(1, 1e-1);
        result[3].should.be.approximately(0, 1e-1);
    });

    it('big case', function () {
        var trainingSet = [[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [1, 3], [1, 4], [4, 1],
                            [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [5, 5], [4, 5], [3, 5]];
        var predictions = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
                            [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]];

        var nn = new FeedforwardNeuralNetwork([2, 10, 2]);
        nn.train(trainingSet, predictions, 200, 0.1, 0.1);

        var result = nn.predict([[5, 4]]);

        result[0][0].should.be.lessThan(result[0][1]);
    });
});