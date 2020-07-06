'use strict';

import ML from './lib/ml.js'
import layers from './data/predictor-state.js';

document.addEventListener('DOMContentLoaded', () => {
	globalThis.ML = ML;

	let predictor;
	let elemCanvas  = document.querySelector('canvas');
	let ctx         = elemCanvas.getContext('2d');
	let drawing     = 0;

	elemCanvas.addEventListener('mouseup', () => drawing = 0);
	elemCanvas.addEventListener('mousedown', () => drawing = 1);

	elemCanvas.addEventListener('mousemove', (e) => {
		if (!drawing)
			return;

		ctx.rect(e.offsetX / 10, e.offsetY / 10, 2, 2);
		ctx.fill();

		let imageData = ctx.getImageData(0, 0, 28, 28);
		let vec = [];
		let row = [];

		for (let v = 0, i = 0; i < imageData.data.length; i++) {
			row.push(imageData.data[i] / 255);

			if (row.length === 4) {
				vec.push(row[3]);
				row.splice(0, 99);
			}
		}

		let res = predictor.predict(vec);
		let max = Math.max(...res);

		console.clear();
		console.log(max, res.indexOf(max));
	});
	
	document.querySelector('#clear-canvas').addEventListener('click', () => {
		ctx.clearRect(0, 0, elemCanvas.width, elemCanvas.height);
	}, false);

	globalThis.predictor = predictor = new ML.Predictor({
		activationFunction      : ML.activationFunctions.sigmoid,
		errorFunction           : ML.errorFunctions.quadraticAverage,
		layers                  : layers,
	});
});