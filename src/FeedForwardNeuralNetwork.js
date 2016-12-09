'use strict';

var Matrix = require('ml-matrix');

var Layer = require('./Layer');
var Utils = require('./Utils');
const ACTIVATION_FUNCTIONS = require('./ActivationFunctions');

class FeedForwardNeuralNetworks {

    /**
     * Create a new Feedforword neural network model.
     * @param {object} options
     * @param {Array} [options.hiddenLayers] - Array that contains the sizes of the hidden layers.
     * @oaram {number} [options.iterations] - Number of iterations at the training step.
     * @param {number} [options.learningRate] - Learning rate of the neural net (also known as epsilon).
     * @poram {number} [options.regularization] - Regularization parameter af the neural net.
     * @poram {string} [options.activation] - activation function to be used. (options: 'tanh'(default),
     * 'identity', 'logistic', 'arctan', 'softsign', 'relu', 'softplus', 'bent', 'sinusoid', 'sinc', 'gaussian').
     */
    constructor(options) {
        if (options === undefined) options = {};
        if (options.model) {
            // load network
            this.hiddenLayers = options.hiddenLayers;
            this.iterations = options.iterations;
            this.learningRate = options.learningRate;
            this.regularization = options.regularization;
            this.dicts = options.dicts;
            this.activation = options.activation;
            this.model = new Array(options.layers.length);

            for (var i = 0; i < this.model.length; ++i) {
                this.model[i] = Layer.load(options.layers[i]);
            }
        } else {
            // default constructor
            this.hiddenLayers = options.hiddenLayers === undefined ? [10] : options.hiddenLayers;
            this.iterations = options.iterations === undefined ? 50 : options.iterations;

            this.learningRate = options.learningRate === undefined ? 0.01 : options.learningRate;
            //this.momentum = options.momentum === undefined ? 0.1 : options.momentum;
            this.regularization = options.regularization === undefined ? 0.01 : options.regularization;

            this.activation = options.activation === undefined ? 'tanh' : options.activation;
            if (!(this.activation in Object.keys(ACTIVATION_FUNCTIONS))) {
                //console.warn("Setting default activation function: 'tanh'");
                this.activation = 'tanh';
            }
        }
    }

    /**
     * Function that build and initialize the neural net.
     * @param {number} inputSize - total of features to fit.
     * @param {number} outputSize - total of labels of the prediction set.
     */
    buildNetwork(inputSize, outputSize) {
        var size = 2 + (this.hiddenLayers.length - 1);
        this.model = new Array(size);

        // input layer
        this.model[0] = new Layer({
            inputSize: inputSize,
            outputSize: this.hiddenLayers[0],
            activation: this.activation,
            regularization: this.regularization,
            epsilon: this.learningRate
        });

        // hidden layers
        for (var i = 1; i < this.hiddenLayers.length; ++i) {
            this.model[i] = new Layer({
                inputSize: this.hiddenLayers[i - 1],
                outputSize: this.hiddenLayers[i],
                activation: this.activation,
                regularization: this.regularization,
                epsilon: this.learningRate
            });
        }

        // output layer
        this.model[size - 1] = new Layer({
            inputSize: this.hiddenLayers[this.hiddenLayers.length - 1],
            outputSize: outputSize,
            activation: 'exp',
            regularization: this.regularization,
            epsilon: this.learningRate
        });
    }

    /**
     * Train the neural net with the given features and labels.
     * @param {Matrix|Array} features
     * @param {Matrix|Array} labels
     */
    train(features, labels) {
        features = Matrix.checkMatrix(features);
        this.dicts = Utils.dictOutputs(labels);

        var inputSize = features.columns;
        var outputSize = Object.keys(this.dicts.inputs).length;

        this.buildNetwork(inputSize, outputSize);

        for (var i = 0; i < this.iterations; ++i) {
            var probabilities = this.propagate(features);
            this.backpropagation(features, labels, probabilities);
        }
    }

    /**
     * Propagate the input(training set) and retrives the probabilities of each class.
     * @param {Matrix} X
     * @return {Matrix} probabilities of each class.
     */
    propagate(X) {
        var input = X;
        for (var i = 0; i < this.model.length; ++i) {
            input = this.model[i].forward(input);
        }

        // get probabilities
        return input.divColumnVector(Utils.sumRow(input));
    }

    /**
     * Function that applies the backpropagation algorithm on each layer of the network
     * in order to fit the features and labels.
     * @param {Matrix} features
     * @param {Array} labels
     * @param {Matrix} probabilities - probabilities of each class of the feature set.
     */
    backpropagation(features, labels, probabilities) {
        for (var i = 0; i < probabilities.length; ++i) {
            probabilities[i][this.dicts.inputs[labels[i]]] -= 1;
        }

        // remember, the last delta doesn't matter
        var delta = probabilities;
        for (var i = this.model.length - 1; i >= 0; --i) {
            var a = i > 0 ? this.model[i - 1].a : features;
            delta = this.model[i].backpropagation(delta, a);
        }

        for (var i = 0; i < this.model.length; ++i) {
            this.model[i].update();
        }
    }

    /**
     * Predict the output given the feature set.
     * @param {Array|Matrix} features
     * @return {Array}
     */
    predict(features) {
        features = Matrix.checkMatrix(features);
        var outputs = new Array(features.rows);
        var probabilities = this.propagate(features);
        for (var i = 0; i < features.rows; ++i) {
            outputs[i] = this.dicts.outputs[probabilities.maxRowIndex(i)[1]];
        }

        return outputs;
    }

    /**
     * Export the current model to JSOM.
     * @return {object} model
     */
    toJSON() {
        var model = {
            model: 'FNN',
            hiddenLayers: this.hiddenLayers,
            iterations: this.iterations,
            learningRate: this.learningRate,
            regularization: this.regularization,
            activation: this.activation,
            dicts: this.dicts,
            layers: new Array(this.model.length)
        };

        for (var i = 0; i < this.model.length; ++i) {
            model.layers[i] = this.model[i].toJSON();
        }

        return model;
    }

    /**
     * Load a Feedforward Neural Network with the current model.
     * @param {object} model
     * @return {FeedForwardNeuralNetworks}
     */
    static load(model) {
        if (model.model !== 'FNN') {
            throw new RangeError('the current model is not a feed forward network');
        }

        return new FeedForwardNeuralNetworks(model);
    }
}

module.exports = FeedForwardNeuralNetworks;
