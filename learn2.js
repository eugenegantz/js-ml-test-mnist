'use strict';

const fs = require('fs');
const { ML } = require('./lib.js');
const mnist = require('mnist');

(async function main() {
	let set = mnist.set(8000, 10);

	let trainingSet = set.training;
	let testSet = set.test;

	function arrayShuffle(arr) {
		return arr.sort(() => Math.round(Math.random()) ? -1 : 1);
	}

	// ->   x    x    x   ->
	// 784, 512, 128, 32, 10

	let layers = [
		new Array(28 * 28),
		new Array(512),
		new Array(128),
		[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	];

	console.log(trainingSet[0].input.length);

	let predictor = ML.createPredictor({
		layers                  : layers,
		activationFunction      : ML.activationFunctions.sigmoid,
		errorFunction           : ML.errorFunctions.quadraticAverage,
	});

	debugger;

	predictor.train({
		data                    : trainingSet,
		epochs                  : 1,
		learningRate            : 0.001,
	});

	let weightsData = 'module.exports = ' + JSON.stringify({
		layers: predictor.layers,
		weights: predictor.weights,
		biases: predictor.biases,
	}, null, '\t');

	testSet.forEach(row => {
		let { input, output } = row;

		console.log(output, predictor.predict(input));
	});

	fs.writeFile('./weights.js', weightsData, () => {
		console.log(`DONE`);
	});
})();