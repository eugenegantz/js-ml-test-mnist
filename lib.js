'use strict';

const
	ML = {

		matrix: {
			multiplyMatrixVector(matrix, vector) {
				let res = [];

				for (let i = 0; i < matrix.length; i++) {
					res[i] = 0;

					for (let j = 0; j < matrix[i].length; j++)
						res[i] += matrix[i][j] * vector[j];
				}

				return res;
			},
		},


		/**
		 * Логистическая функция. Сигмоида
		 *
		 * @param {Number} x
		 *
		 * @return {Number} -> от 0 до 1
		 * */
		sigmoid(x) {
			return 1 / (1 + Math.pow(Math.E, -x));
		},


		sigmoidVector(arr) {
			let res = [];

			for (let i = 0; i < arr.length; i++)
				res[i] = ML.sigmoid(arr[i]);

			return res;
		},


		_createWeights(layers) {
			let weights = [];

			// Важно. Иначе сеть не обучается
			let randWeight = (layer) => {
				return this.sigmoid(6 - Math.random() * 12) * (layer + 1);
			};

			for (let i = 0; i < layers.length - 1; i++) {
				let layer0 = layers[i];
				let layer1 = layers[i + 1];

				weights[i] = (new Array(layer1.length)).fill(1);
				weights[i] = weights[i].map(() => (new Array(layer0.length)).fill(randWeight(i)));
			}

			return weights;
		},


		/**
		 * Создать сеть
		 *
		 * @param {Object} args
		 * @param {Array} args.layers
		 * @param {Array} args.weights
		 *
		 * @return {Object}
		 * */
		createPredictor(args) {
			let { weights, layers } = args;

			if (!weights && layers)
				weights = this._createWeights(layers);

			return {
				layers,
				weights,
				predict(vec) {
					for (let i = 0; i < this.weights.length; i++) {
						let matrix = this.weights[i];

						this.layers[i] = vec;

						vec = ML.sigmoidVector(ML.matrix.multiplyMatrixVector(matrix, vec));

						this.layers[i + 1] = vec;
					}

					return vec;
				},
			};
		},


		/**
		 * Обучить сеть
		 *
		 * @param {Object} arg
		 * @param {Array} arg.data - данные для обучения
		 * @param {Array} arg.predictor - сеть
		 * @param {Array} arg.target - целевые узлы
		 * @param {Number=1} arg.learningRate - коэф. скорости движения градиента
		 * @param {Number=1} arg.steps - количество повторов на каждый образец
		 *
		 * @return {Array}
		 * */
		train(arg) {
			let {
				learningRate = 1,
				predictor,
				data,
				steps = 1,
			} = arg;

			for (let _data of data) {
				for (let step = 0; step < steps; step++) {
					predictor.predict(_data.input);

					predictor.delta = [];

					let target = _data.output;
					let { layers, weights } = predictor;

					// Обход каждого слоя в направлении обратном импульсу
					for (let v = layers.length - 1; v > 0; v--) {
						let input               = layers[v - 1];
						let output              = layers[v];
						let weightsMatrix       = weights[v - 1];
						let isInnerLayer        = v !== layers.length - 1;

						predictor.delta[v]      = [];

						// Рассчет малых дельт (дельты выхода)
						for (let j = 0; j < output.length; j++) {
							// Для скрытых слоев
							if (isInnerLayer) {
								let _sum = 0;

								for (let k = 0; k < weights[v].length; k++)
									_sum += weights[v][k][j] * predictor.delta[v + 1][k];

								predictor.delta[v][j] = _sum * (output[j] * (1 - output[j]));
							}

							// Для последнего слоя
							else {
								predictor.delta[v][j] = (output[j] - target[j]) * (output[j] * (1 - output[j]));
							}
						}

						// Пересчет весов
						for (let i = 0; i < weightsMatrix.length; i++) {
							for (let j = 0; j < weightsMatrix[i].length; j++) {
								let dw = -learningRate * input[j] * predictor.delta[v][i];

								weightsMatrix[i][j] = weightsMatrix[i][j] + dw;
							}
						}
					}
				}
			}

			return predictor;
		},

	};

module.exports = { ML };