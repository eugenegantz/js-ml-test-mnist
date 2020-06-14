'use strict';

const
	{ ML }      = require('./lib.js'),
	fs          = require('fs'),
	idxData     = require('idx-data');

const
	IMAGE_SIZE = 28,
	IMAGE_LENGTH = IMAGE_SIZE ** 2;

function createData(images, labels) {
	let res = [];

	labels.data.forEach((label, index) => {
		let from = IMAGE_LENGTH * index;
		let to = IMAGE_LENGTH * index + IMAGE_LENGTH;

		let output = new Array(10).fill(0);
		let input = [];

		output[label] = 1;
		images.data.slice(from, to).forEach((v, i) => input[i] = v / 255);

		res.push({ input, output });
	});

	return res;
}

(async () => {
	const [
		trainingImages,
		trainingLabels,
		testImages,
		testLabels,
	] = await Promise.all([
		idxData.loadBits('./data/train-images.idx3-ubyte'),
		idxData.loadBits('./data/train-labels.idx1-ubyte'),
		idxData.loadBits('./data/t10k-images.idx3-ubyte'),
		idxData.loadBits('./data/t10k-labels.idx1-ubyte'),
	]);

	let trainData = createData(trainingImages, trainingLabels);
	let testData = createData(testImages, testLabels).slice(0, 10);

	// ->   x    x    x   ->
	// 784, 512, 128, 32, 10

	let layers = [
		new Array(28 * 28),
		new Array(512),
		new Array(128),
		new Array(32),
		[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	];

	let predictor = ML.createPredictor({
		layers                  : layers,
		activationFunction      : ML.activationFunctions.sigmoid,
		errorFunction           : ML.errorFunctions.quadraticAverage,
	});

	debugger;

	predictor.train({
		data                    : trainData,
		epochs                  : 100,
		learningRate            : 0.001,
	});

	let weightsData = 'module.exports = ' + JSON.stringify({
		layers: predictor.layers,
		weights: predictor.weights,
		biases: predictor.biases,
	}, null, '\t');

	testData.forEach(row => {
		let { input, output } = row;

		console.log(output, predictor.predict(input));
	});

	fs.writeFile('./data/weights.js', weightsData, () => {
		console.log(`DONE`);
	});
})();