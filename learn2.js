'use strict';

const fs = require('fs');
const { ML } = require('./lib.js');
const mnist = require('mnist');

(async function main() {
	let set = mnist.set(8000, 10);

	let trainingSet = set.training;
	let testSet = set.test;

	let layers = [
		new Array(28 ** 2),
		new Array(256),
		new Array(256),
		[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	];

	console.log(trainingSet[0].input.length);

	let predictor = ML.createPredictor({ layers });

	debugger;

	predictor = ML.train({
		data: trainingSet,
		predictor,
		steps: 1,
	});

	let weightsData = 'module.exports = ' + JSON.stringify({
		layers: predictor.layers,
		weights: predictor.weights,
	}, null, '\t');

	testSet.forEach(row => {
		let { input, output } = row;

		console.log(output, predictor.predict(input));
	});

	fs.writeFile('./weights.js', weightsData, () => {
		console.log(`DONE`);
	});
})();