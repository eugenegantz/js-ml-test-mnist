'use strict';

const fs = require('fs');
const { ML } = require('./lib.js');

(async function main() {
	function createSetArray1({ length }) {
		let _set = [];
		let p = 0;

		for (let i = 0; i < length; i++) {
			let row = new Array(5).fill(0);

			if (!(p in row))
				p = 0;

			row[p++] = 1;

			_set.push({ input: row, output: row });
		}

		return _set;
	}

	function createSetArray2({ length }) {
		let _set = [];

		for (let i = 0; i < length; i++) {
			let row = new Array(10).fill(1).map(() => Math.round(Math.random()));

			_set.push({ input: row, output: row });
		}

		return _set;
	}

	function arrayShuffle(arr) {
		return arr.sort(() => Math.round(Math.random()) ? -1 : 1);
	}

	let trainingSet = arrayShuffle(createSetArray2({ length: 60000 }));
	let testSet = createSetArray2({ length: 5 });

	let layers = [
		new Array(5).fill(0),
		new Array(5).fill(0),
		// new Array(2).fill(0),
		new Array(5).fill(0),
		// new Array(5).fill(0),
		// new Array(5).fill(0),
	];

	let predictor = ML.createPredictor({
		layers                  : layers,
		activationFunction      : ML.activationFunctions.sigmoid,
		errorFunction           : ML.errorFunctions.quadraticAverage,
	});

	debugger;

	predictor.train({
		data: trainingSet,
		predictor,
		epochs: 100,
		learningRate: 0.0001,
	});

	let weightsData = 'module.exports = ' + JSON.stringify({
		layers: predictor.layers,
		weights: predictor.weights,
	}, null, '\t');

	testSet.forEach(row => {
		let { input, output } = row;

		console.log(output, predictor.predict(input));
	});

	fs.writeFile('./data/weights.js', weightsData, () => {
		console.log(`DONE`);
	});
})();