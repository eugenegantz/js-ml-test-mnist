'use strict';


function sigmoid(x) {
	return 1 / (1 + Math.pow(Math.E, -x));
}

sigmoid.derivative = function(x) {
	return sigmoid(x) * (1 - sigmoid(x));
};

function quadraticAverage(t, y) {
	return 0.5 * ((t - y) ** 2);
}

quadraticAverage.derivative = function(t, y) {
	return y - t;
};


function _avgQuad(output, target) {
	let sum = 0;

	for (let i = 0; i < output.length; i++)
		sum += Math.pow(target[i] - output[i], 2);

	return sum / output.length;
}


// ------------


function _maxComparison(output, target) {
	let max = { i: 0, v: -Infinity };

	for (let i = 0; i < output.length; i++) {
		if (output[i] > max.v) {
			max.v = output[i];
			max.i = i;
			max.t = target[i];
		}
	}

	return max;
}


// ------------


function arrayShuffle(arr) {
	return arr.sort(() => Math.round(Math.random()) ? -1 : 1);
}


// ------------


function multiplyMatrixVector(matrix, vector) {
	let res = [];

	for (let i = 0, len1 = matrix.length; i < len1; i++) {
		res[i] = 0;

		for (let j = 0, len2 = matrix[i].length; j < len2; j++)
			res[i] += matrix[i][j] * vector[j];
	}

	return res;
}


function sumVectorVector(vec1, vec2) {
	let vec = [];

	for (let i = 0, len = vec1.length; i < len; i++)
		vec[i] = vec1[i] + vec2[i];

	return vec;
}


function randomRange(from, to) {
	return from + Math.round(Math.random() * (to - from));
}


const
	ML = {

		activationFunctions: {
			sigmoid,
		},


		errorFunctions: {
			quadraticAverage,
		},


		_createWeights(layers) {
			let weights = [];

			// Важно. Иначе сеть не обучается
			let rand = () => Math.random() * 2 - 1;

			for (let i = 0; i < layers.length - 1; i++) {
				let layer0 = layers[i];
				let layer1 = layers[i + 1];

				weights[i] = (new Array(layer1.length)).fill(1);
				weights[i] = weights[i].map(() => (new Array(layer0.length)).fill(rand(i)));
			}

			return weights;
		},


		_createBiases(layers) {
			let biases = [];
			let rand = () => Math.random() * 2 - 1;

			for (let i = 0; i < layers.length - 1; i++)
				biases[i] = (new Array(layers[i + 1].length)).fill(1).map(() => rand());

			return biases;
		},


		createPredictor(arg) {
			let {
				weights,
				biases,
				layers,
				activationFunction,
				errorFunction,
			} = arg;

			if (!weights && layers)
				weights = this._createWeights(layers);

			if (!biases && layers)
				biases = this._createBiases(layers);

			let netInputs = [];

			return {
				netInputs,

				layers,

				weights,

				biases,

				activationFunction,

				errorFunction,

				predict(vec) {
					let { activationFunction } = this;

					for (let i = 0; i < this.weights.length; i++) {
						let net;
						let matrix = this.weights[i];
						let biases = this.biases[i];

						this.layers[i] = vec;

						net = multiplyMatrixVector(matrix, vec);
						net = sumVectorVector(net, biases);
						vec = net.map(activationFunction);

						this.netInputs[i + 1] = net;
						this.layers[i + 1] = vec;
					}

					return vec;
				},

				train(arg) {
					let {
						learningRate = 1,
						data,
						epochs = 1,
					} = arg;

					let d0 = new Date();

					let predictor   = this;
					let actFunc     = this.activationFunction;
					let errFunc     = this.errorFunction;

					let getLearningRate = (epoch) => {
						return learningRate;

						/*
						let r = 10;
						let d = 0.5;

						return learningRate * Math.pow(d, Math.floor((1 + epoch) / r));
						*/
					};

					for (let epoch = 0; epoch < epochs; epoch++) {
						data = arrayShuffle(data);

						let _log = {
							totalIterations: 0,
							successIterations: 0,
						};

						for (let d = 0; d < data.length; d++) {
							let _data = data[d];

							predictor.predict(_data.input);

							predictor.delta = [];

							let target = _data.output;
							let { layers, weights, netInputs, biases } = predictor;
							let output = layers[layers.length - 1];

							let precision = Math.abs(1 - _avgQuad(output, target));
							let _maxComp = _maxComparison(output, target);

							_log.totalIterations++;

							if (_maxComp.t)
								_log.successIterations++;

							// if (precision >= 0.95 && step)
							// break;

							// Обход каждого слоя в направлении обратном импульсу
							for (let v = layers.length - 1; v > 0; v--) {
								let netInput            = netInputs[v];
								let input               = layers[v - 1];
								let output              = layers[v];
								let weightsMatrix       = weights[v - 1];
								let biasesMatrix        = biases[v - 1];
								let isInnerLayer        = v !== layers.length - 1;

								predictor.delta[v]      = [];

								// Рассчет малых дельт (дельты выхода)
								for (let j = 0; j < output.length; j++) {
									// Для скрытых слоев
									if (isInnerLayer) {
										let _sum = 0;

										for (let k = 0; k < weights[v].length; k++)
											_sum += weights[v][k][j] * predictor.delta[v + 1][k];

										predictor.delta[v][j] = _sum * actFunc.derivative(netInput[j]);
									}

									// Для последнего слоя
									else {
										predictor.delta[v][j] = errFunc.derivative(target[j], output[j]) * actFunc.derivative(netInput[j]);
									}
								}

								let _learningRate = getLearningRate(epoch);

								// Пересчет весов
								for (let i = 0; i < weightsMatrix.length; i++) {
									for (let j = 0; j < weightsMatrix[i].length; j++) {
										let dw = -_learningRate * input[j] * predictor.delta[v][i];

										weightsMatrix[i][j] = weightsMatrix[i][j] + dw;
									}

									biasesMatrix[i] += -_learningRate * predictor.delta[v][i];
								}
							}

							console.log(''
								+ `data: ${d + 1} of ${data.length},`
								+ `\tepochs: ${epoch + 1} of ${epochs},`
								+ `\tq_avg: ${precision},`
								+ `\ts:${_log.successIterations / _log.totalIterations}`
								+ `\tm: [${_maxComp.i}, ${_maxComp.v}, ${_maxComp.t}]`
							)
						}
					}

					let d1 = new Date();

					console.log(`time: ${d1 - d0}`);

					return predictor;
				},
			};
		},

	};

module.exports = { ML };