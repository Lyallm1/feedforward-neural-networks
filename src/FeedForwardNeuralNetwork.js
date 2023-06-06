import ACTIVATION_FUNCTIONS from './activationFunctions';
import { Layer } from './Layer';
import { Matrix } from 'ml-matrix';
import { OutputLayer } from './OutputLayer';

export default class FeedForwardNeuralNetworks {
  constructor(options) {
    options = options || {};
    if (options.model) {
      this.hiddenLayers = options.hiddenLayers;
      this.iterations = options.iterations;
      this.learningRate = options.learningRate;
      this.regularization = options.regularization;
      this.dicts = options.dicts;
      this.activation = options.activation;
      this.activationParam = options.activationParam;
      this.model = new Array(options.layers.length);
      for (let i = 0; i < this.model.length - 1; ++i) this.model[i] = Layer.load(options.layers[i]);
      this.model[this.model.length - 1] = OutputLayer.load(options.layers[this.model.length - 1]);
    } else {
      this.hiddenLayers = options.hiddenLayers || [10];
      this.iterations = options.iterations || 50;
      this.learningRate = options.learningRate || 0.01;
      this.regularization = options.regularization || 0.01;
      this.activation = options.activation || 'tanh';
      this.activationParam = options.activationParam || 1;
      if (!(this.activation in Object.keys(ACTIVATION_FUNCTIONS))) this.activation = 'tanh';
    }
  }

  buildNetwork(inputSize, outputSize) {
    this.model = new Array(this.hiddenLayers.length + 1);
    this.model[0] = new Layer({ inputSize: inputSize, outputSize: this.hiddenLayers[0], activation: this.activation, activationParam: this.activationParam, regularization: this.regularization, epsilon: this.learningRate });
    for (let i = 1; i < this.hiddenLayers.length; ++i) this.model[i] = new Layer({ inputSize: this.hiddenLayers[i - 1], outputSize: this.hiddenLayers[i], activation: this.activation, activationParam: this.activationParam, regularization: this.regularization, epsilon: this.learningRate });
    this.model[this.hiddenLayers.length] = new OutputLayer({ inputSize: this.hiddenLayers[this.hiddenLayers.length - 1], outputSize: outputSize, activation: this.activation, activationParam: this.activationParam, regularization: this.regularization, epsilon: this.learningRate });
  }

  train(features, labels) {
    features = Matrix.checkMatrix(features);
    this.dicts = dictOutputs(labels);
    if (!this.model) this.buildNetwork(features.columns, Object.keys(this.dicts.inputs).length);
    for (let i = 0; i < this.iterations; ++i) this.backpropagation(features, labels, this.propagate(features));
  }

  propagate(X) {
    for (const layer of this.model) X = layer.forward(X);
    return X.divColumnVector(X.sum('row'));
  }

  backpropagation(features, labels, probabilities) {
    for (let i = 0; i < probabilities.rows; ++i) probabilities.set(i, this.dicts.inputs[labels[i]], probabilities.get(i, this.dicts.inputs[labels[i]]) - 1);
    this.model.reverse().forEach((layer, i) => probabilities = layer.backpropagation(probabilities, i > 0 ? this.model[i - 1].a : features))
    for (const layer of this.model) layer.update();
  }

  predict(features) {
    features = Matrix.checkMatrix(features);
    return Array.from({ length: features.rows }, i => this.dicts.outputs[this.propagate(features).maxRowIndex(i)[1]]);
  }

  toJSON() {
    const model = { model: 'FNN', hiddenLayers: this.hiddenLayers, iterations: this.iterations, learningRate: this.learningRate, regularization: this.regularization, activation: this.activation, activationParam: this.activationParam, dicts: this.dicts, layers: Array.from(this.model) };
    this.model.forEach((layer, i) => model.layers[i] = layer.toJSON());
    return model;
  }

  static load(model) {
    if (model.model !== 'FNN') throw new RangeError('the current model is not a feed forward network');
    return new FeedForwardNeuralNetworks(model);
  }
}

function dictOutputs(array) {
  const inputs = {}, outputs = {};
  let index = 0;
  for (const val of array) if (!inputs[val]) {
    [inputs[val], outputs[index]] = [index, val];
    index++;
  }
  return { inputs, outputs };
}
