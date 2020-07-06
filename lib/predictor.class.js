'use strict';

import {
	avgQuad,
	arrayShuffle,
	maxComparison,
	sumVectorVector,
	multiplyMatrixVector,
} from './utils.js';

const
	voidFunction = () => {};


export default class Predictor {

	constructor(opts = {}) {
		let {
			layers,
			errorFunction,
			activationFunction,
		} = opts;

		if (!layers)
			throw new Error('opts.layers is not defined');

		this._layers                = this._initLayers(layers);

		this.activationFunction     = activationFunction;
		this.errorFunction          = errorFunction;
	}


	_initLayers(layers) {
		let _layers = [];

		for (let i = 0; i < layers.length; i++) {
			if (layers[i + 1])
				_layers[i + 1] = this._initLayer(layers[i + 1], null);

			_layers[i] = this._initLayer(_layers[i] || layers[i], _layers[i + 1]);
		}

		return _layers;
	}


	_initLayer(currLayer, nextLayer) {
		let _currLayer = null;

		if (typeof currLayer === 'object') {
			_currLayer = currLayer;
		}

		else if (typeof currLayer === 'number') {
			_currLayer = this._createLayer(currLayer);
		}

		if (!_currLayer)
			return _currLayer;

		if (!_currLayer.nodes && _currLayer.weights)
			_currLayer.nodes = new Array(_currLayer.weights[0].length).fill(0);

		if (typeof _currLayer.nodes === 'number')
			_currLayer.nodes = new Array(_currLayer.nodes).fill(0);

		if (!_currLayer.biases)
			_currLayer.biases = this._createBiases(_currLayer.nodes.length);

		if (!_currLayer.weights && nextLayer)
			_currLayer.weights = this._createWeights(_currLayer.nodes.length, nextLayer.nodes.length);

		return _currLayer;
	}


	_getLayerLength(layer) {
		if (typeof layer === 'object' && layer.nodes)
			return layer.nodes.length;

		else if (typeof layer === 'number')
			return layer;
	}


	_createLayer(layer) {
		let _layer = {
			net     : null,
			nodes   : null,
			biases  : null,
			weights : null,
		};

		_layer.nodes = new Array(layer).fill(0);

		return _layer;
	}


	_createWeights(layer0L, layer1L) {
		let rand = () => Math.random() * 2 - 1;

		return new Array(layer1L).fill(1).map(() => {
			return new Array(layer0L).fill(1).map(() => {
				return rand();
			});
		});
	}


	_createBiases(layer1L) {
		let rand = () => Math.random() * 2 - 1;

		return new Array(layer1L).fill(1).map(() => rand());
	}


	predict(vec) {
		let { activationFunction } = this;

		for (let i = 0; i < this._layers.length; i++) {
			let net;
			let layerCurr       = this._layers[i];
			let layerNext       = this._layers[i + 1];
			let weightMatrix    = layerCurr.weights;
			let biasesVector    = layerCurr.biases;

			layerCurr.nodes = vec;

			if (layerNext) {
				net = multiplyMatrixVector(weightMatrix, vec);
				net = sumVectorVector(net, biasesVector);
				vec = net.map(activationFunction);

				layerNext.net   = net;
				layerNext.nodes = vec;
			}
		}

		return vec;
	}


	train(options) {
		let {
			data,
			onLog = voidFunction,
			epochs = 1,
			learningRate = 1,
		} = options;

		let predictor   = this;
		let actFunc     = this.activationFunction;
		let errFunc     = this.errorFunction;
		let delta       = [];

		let _log = {};

		for (let epoch = 0; epoch < epochs; epoch++) {
			data = arrayShuffle(data);

			_log.epoch              = epoch;
			_log.totalIterations    = 0;
			_log.successIterations  = 0;

			for (let d = 0; d < data.length; d++) {
				let _data = data[d];

				_log.dataIndex = d;

				predictor.predict(_data.input);

				let target      = _data.output;
				let layers      = predictor._layers;
				let output      = layers[layers.length - 1].nodes;
				let _maxComp    = maxComparison(output, target);

				_log.comp = _maxComp;
				_log.outputErrorAverage = Math.abs(1 - avgQuad(output, target));

				_log.totalIterations++;

				if (_maxComp.t)
					_log.successIterations++;

				// Обход каждого слоя в направлении обратном импульсу
				for (let v = layers.length - 1; v > 0; v--) {
					let layerCurr           = layers[v];
					let layerPrev           = layers[v - 1];

					let net                 = layerCurr.net;
					let input               = layerPrev.nodes;
					let output              = layerCurr.nodes;

					let isInnerLayer        = v !== layers.length - 1;

					delta[v]                = delta[v] || [];

					// Рассчет малых дельт (дельты выхода)
					for (let j = 0; j < output.length; j++) {
						// Для скрытых слоев
						if (isInnerLayer) {
							let _sum = 0;

							for (let k = 0; k < layerCurr.weights.length; k++)
								_sum += layerCurr.weights[k][j] * delta[v + 1][k];

							delta[v][j] = _sum * actFunc.derivative(net[j]);
						}

						// Для последнего слоя
						else {
							delta[v][j] = errFunc.derivative(target[j], output[j]) * actFunc.derivative(net[j]);
						}
					}

					// Пересчет весов
					for (let i = 0; i < layerPrev.weights.length; i++) {
						for (let j = 0; j < layerPrev.weights[i].length; j++)
							layerPrev.weights[i][j] = layerPrev.weights[i][j] + (-learningRate * input[j] * delta[v][i]);

						layerPrev.biases[i] += -learningRate * delta[v][i];
					}
				}

				onLog(_log);
			}
		}
	}


	getState() {
		return this._layers.map(layer => {
			return {
				weights: layer.weights,
				biases: layer.biases,
				nodes: layer.nodes.length,
			};
		});
	}

}