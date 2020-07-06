'use strict';

import ML from './lib/ml.js';
import fs from 'fs';
import idxData from 'idx-data';


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

async function getLayers() {
	let filename = 'predictor-state.js';

	let e = fs.existsSync('./data/' + filename);

	if (e)
		return (await import('./data/' + filename)).default;

	return [28 * 28, 512, 128, 32, 10];
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


	// -----------------------

	let layers = await getLayers();

	console.log(layers);

	let predictor = new ML.Predictor({
		layers: layers,
		activationFunction: ML.activationFunctions.sigmoid,
		errorFunction: ML.errorFunctions.quadraticAverage,
	});


	// -----------------------


	function onLog(_log) {
		console.log(''
			+ `data: ${_log.dataIndex + 1} of ${trainData.length},`
			+ `\tepochs: ${_log.epoch + 1} of ${trainOpts.epochs},`
			+ `\tq_avg: ${_log.outputErrorAverage},`
			+ `\ts:${_log.successIterations / _log.totalIterations}`
			+ `\tm: [${_log.comp.i}, ${_log.comp.v}, ${_log.comp.t}]`
		);
	}

	let trainOpts = {
		onLog                   : onLog,
		data                    : trainData,
		epochs                  : 1,
		learningRate            : 0.001,
	};

	predictor.train(trainOpts);


	// -----------------------


	testData.forEach(row => {
		let { input, output } = row;

		console.log(output, predictor.predict(input));
	});


	// -----------------------


	function writeWeights() {
		let state = predictor.getState();
		let json = 'export default ' + JSON.stringify(state);

		fs.writeFileSync('./data/predictor-state.js', json);

		console.log(`weights saved on disk: done`);
	}

	await writeWeights();
})();