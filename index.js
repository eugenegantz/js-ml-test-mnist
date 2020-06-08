'use strict';

export { ML } from './lib.js';

document.addEventListener('DOMContentLoaded', () => {
	globalThis.ML = ML;

	// Слой 1. 10 узлов ввода
	// Слой 2. 5 узлов скрытых
	// Слой 3. 8 узлов вывода
	/*
	let network = [
		// Веса для первого слоя
		[
			[0.5, 2, 0, 2, 0, 0.2, 0, 0, 0, 0],
			[0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		],

		// Веса для второго слоя
		[
			[0.5, 0, 0, 0, 0,],
			[0, 0, 0.3, 0, 0,],
			[0, 0, 0, 0, 0,],
			[0, 0, 0, 0, 0,],
			[0, 0, 0, 0, 0,],
			[0, 0, 0, 0, 0,],
			[0, 0, 0, 0, 0,],
			[0, 0, 0, 0, 0,],
		],
	];
	*/

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

		// console.log(e);
	});

	globalThis.predictor = ML.createPredictor({
		layers: [
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0,],
			[0, 0,],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		],
	});
});