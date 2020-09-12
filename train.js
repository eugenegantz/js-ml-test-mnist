'use strict';

import ML from './lib/ml.js';
import * as utils from './lib/utils.js';
import modFS from 'fs';
import modPath from 'path';
import modURL from 'url';
import idxData from 'idx-data';

const
	IMAGE_SIZE = 28,
	IMAGE_LENGTH = IMAGE_SIZE ** 2;

const
	__dirname = modPath.dirname(modURL.fileURLToPath(import.meta.url));

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

function createLinerOperator(height, width) {
	function _getPixelIndex(x, y) {
		return y * height + x;
	}

	function _imageArrayGet(vec, x, y) {
		let i = _getPixelIndex(x, y);

		return vec[i];
	}

	function _imageArraySet(vec, x, y, value) {
		if (
			   x < 0
			|| y < 0
			|| x > (width - 1)
			|| y > (height - 1)
		) {
			return;
		}

		let i = _getPixelIndex(x, y);

		return vec[i] = value;
	}

	return {
		linerTransform(vec, matrix) {
			let vec0 = vec.slice(0, Infinity).fill(0);

			for (let y = 0; y < height; y++) {
				for (let x = 0; x < width; x++) {
					let [tx, ty] = utils.multiplyMatrixVector(matrix, [x, y]);

					tx = Math.round(tx);
					ty = Math.round(ty);

					let value = _imageArrayGet(vec, x, y);

					_imageArraySet(vec0, tx, ty, value);
				}
			}

			return vec0;
		},
		offsetTransform(vec, dx, dy) {
			let vec0 = vec.slice(0, Infinity).fill(0);

			for (let y = 0; y < height; y++) {
				for (let x = 0; x < width; x++) {
					let tx = Math.round(x + dx);
					let ty = Math.round(y + dy);

					let value = _imageArrayGet(vec, x, y);

					_imageArraySet(vec0, tx, ty, value);
				}
			}

			return vec0;
		},
	}
}


const {
	linerTransform,
	offsetTransform,
} = createLinerOperator(28, 28);

const
	LINER_OP_SCALE = function(x) {
		return [
			[1 * x,    0    ],
			[0,        1 * x],
		];
	},

	LINER_OP_ROTATE = function(angle) {
		return [
			[Math.cos(angle), -Math.sin(angle)],
			[Math.sin(angle), Math.cos(angle)],
		];
	};


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
	// для повышения разнообразия выборки применить линейные операторы: маштаб, поворот, смещение

	let _transformedTrainData = [];

	let op1 = LINER_OP_ROTATE(-Math.PI / 6);
	let op2 = LINER_OP_ROTATE(Math.PI / 6);
	let op3 = LINER_OP_SCALE(1.1);
	let op4 = LINER_OP_SCALE(0.9);

	for (let data of trainData) {
		_transformedTrainData.push({ input: linerTransform(data.input, op1), output: data.output });
		_transformedTrainData.push({ input: linerTransform(data.input, op2), output: data.output });
		_transformedTrainData.push({ input: linerTransform(data.input, op3), output: data.output });
		_transformedTrainData.push({ input: linerTransform(data.input, op4), output: data.output });

		_transformedTrainData.push({ input: offsetTransform(data.input, -4, 0), output: data.output });
		_transformedTrainData.push({ input: offsetTransform(data.input, 4, 0), output: data.output });

		_transformedTrainData.push({ input: offsetTransform(data.input, 0, -4), output: data.output });
		_transformedTrainData.push({ input: offsetTransform(data.input, 0, 4), output: data.output });

		_transformedTrainData.push({ input: offsetTransform(data.input, 4, 4), output: data.output });
		_transformedTrainData.push({ input: offsetTransform(data.input, -4, -4), output: data.output });
	}

	trainData = trainData.concat(_transformedTrainData);


	async function getLayers() {
		console.log(pathModel);

		let e = modFS.existsSync(pathModel);

		if (e)
			return (await import(modURL.pathToFileURL(pathModel))).default;

		return [28 * 28, 512, 128, 32, 10];
	}


	// -----------------------
	// вернуть параметры процесса

	let args = process.argv.slice(0, 2).reduce((obj, str) => {
		let [key, value] = str.split('=');

		obj[key] = value;

		return obj;
	}, {});


	let pathModel = args.model || './models/predictor-state.js';

	pathModel = modPath.isAbsolute(pathModel)
		? pathModel
		: modPath.join(__dirname, pathModel);


	// -----------------------
	// инициализация

	let layers = await getLayers();

	console.log(layers);

	let predictor = new ML.Predictor({
		layers: layers,
		activationFunction: ML.activationFunctions.sigmoid,
		errorFunction: ML.errorFunctions.quadraticAverage,
	});


	// -----------------------
	// обучить модель

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
		epochs                  : 5,
		learningRate            : 0.001,
	};

	predictor.train(trainOpts);


	// -----------------------
	// тестирование модели

	testData.forEach(row => {
		let { input, output } = row;

		console.log(output, predictor.predict(input));
	});


	// -----------------------
	// зафиксировать состояние модели

	function writeWeights() {
		let state = predictor.getState();
		let json = 'export default ' + JSON.stringify(state);

		modFS.writeFileSync(pathModel, json);

		console.log(`done: predictor state saved on disk`);
	}

	writeWeights();
})();