const logistic = val => 1 / (1 + Math.exp(-val)), expELU = (val, param) => val < 0 ? param * (Math.exp(val) - 1) : val;

export default {
  tanh: { activation: Math.tanh, derivate: val => 1 - val**2 }, identity: { activation: val => val, derivate: () => 1 },
  logistic: { activation: logistic, derivate: val => logistic(val) * (1 - logistic(val)) }, arctan: { activation: Math.atan, derivate: val => 1 / (val**2 + 1) },
  softsign: { activation: val => val / (1 + Math.abs(val)), derivate: val => 1 / ((1 + Math.abs(val)) * (1 + Math.abs(val))) }, relu: { activation: val => val < 0 ? 0 : val, derivate: val => val < 0 ? 0 : 1 },
  softplus: { activation: val => Math.log(1 + Math.exp(val)), derivate: val => 1 / (1 + Math.exp(-val)) }, bent: { activation: val => (Math.sqrt(val**2 + 1) - 1) / 2 + val, derivate: val => val / (2 * Math.sqrt(val**2 + 1)) + 1 },
  sinusoid: { activation: Math.sin, derivate: Math.cos }, sinc: { activation: val => val === 0 ? 1 : Math.sin(val) / val, derivate: val => val === 0 ? 0 : (Math.cos(val) - Math.sin(val) / val) / val },
  gaussian: { activation: val => Math.exp(-(val**2)), derivate: val => -2 * val * Math.exp(-(val**2)) }, 'parametric-relu': { activation: (val, param) => val < 0 ? param * val : val, derivate: (val, param) => val < 0 ? param : 1 },
  'exponential-elu': { activation: expELU, derivate: (val, param) => val < 0 ? expELU(val, param) + param : 1 }, 'soft-exponential': { activation: (val, param) => {
    if (param < 0) return -Math.log(1 - param * (val + param)) / param;
    if (param > 0) return (Math.exp(param * val) - 1) / param + param;
    return val;
  }, derivate: (val, param) => param < 0 ? 1 / (1 - param * (param + val)) : Math.exp(param * val) }
};
