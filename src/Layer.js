import ACTIVATION_FUNCTIONS from './activationFunctions';
import { Matrix } from 'ml-matrix';

export class Layer {
  constructor(options) {
    this.inputSize = options.inputSize;
    this.outputSize = options.outputSize;
    this.regularization = options.regularization;
    this.epsilon = options.epsilon;
    this.activation = options.activation;
    this.activationParam = options.activationParam;
    const selectedFunction = ACTIVATION_FUNCTIONS[options.activation], params = selectedFunction.activation.length;
    this.activationFunction = (i, j) => this.set(i, j, (params > 1 ? val => selectedFunction.activation(val, options.activationParam) : selectedFunction.activation)(this.get(i, j)));
    this.derivate = (i, j) => this.set(i, j, (params > 1 ? val => selectedFunction.derivate(val, options.activationParam) : selectedFunction.derivate)(this.get(i, j)));
    if (options.model) {
      this.W = Matrix.checkMatrix(options.W);
      this.b = Matrix.checkMatrix(options.b);
    } else {
      this.W = Matrix.rand(this.inputSize, this.outputSize);
      this.b = Matrix.zeros(1, this.outputSize);
      this.W.apply((i, j) => this.set(i, j, this.get(i, j) / Math.sqrt(options.inputSize)));
    }
  }
  
  forward(X) {
    const z = X.mmul(this.W).addRowVector(this.b);
    z.apply(this.activationFunction);
    this.a = z.clone();
    return z;
  }

  backpropagation(delta, a) {
    this.dW = a.transpose().mmul(delta);
    this.db = Matrix.rowVector(delta.sum('column'));
    return delta.mmul(this.W.transpose()).mul(a.clone().apply(this.derivate));
  }

  update() {
    this.dW.add(this.W.clone().mul(this.regularization));
    this.W.add(this.dW.mul(-this.epsilon));
    this.b.add(this.db.mul(-this.epsilon));
  }

  toJSON() {
    return { model: 'Layer', inputSize: this.inputSize, outputSize: this.outputSize, regularization: this.regularization, epsilon: this.epsilon, activation: this.activation, W: this.W, b: this.b };
  }

  static load(model) {
    if (model.model !== 'Layer') throw new RangeError('the current model is not a Layer model');
    return new Layer(model);
  }
}
