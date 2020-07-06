'use strict';

export function sigmoid(x) {
	return 1 / (1 + Math.pow(Math.E, -x));
}

sigmoid.derivative = x => sigmoid(x) * (1 - sigmoid(x));


export function quadraticAverage(t, y) {
	return 0.5 * ((t - y) ** 2);
}

quadraticAverage.derivative = (t, y) => y - t;


export function avgQuad(output, target) {
	let sum = 0;

	for (let i = 0; i < output.length; i++)
		sum += Math.pow(target[i] - output[i], 2);

	return sum / output.length;
}


export function maxComparison(output, target) {
	let max = { i: 0, v: -Infinity };

	for (let i = 0; i < output.length; i++) {
		if (output[i] > max.v) {
			max.v = output[i];
			max.i = i;
			max.t = target[i];
		}
	}

	return max;
}


export function arrayShuffle(arr) {
	return arr.sort(() => Math.round(Math.random()) ? -1 : 1);
}


export function multiplyMatrixVector(matrix, vector) {
	let res = [];

	for (let i = 0, len1 = matrix.length; i < len1; i++) {
		res[i] = 0;

		for (let j = 0, len2 = matrix[i].length; j < len2; j++)
			res[i] += matrix[i][j] * vector[j];
	}

	return res;
}


export function sumVectorVector(vec1, vec2) {
	let vec = [];

	for (let i = 0, len = vec1.length; i < len; i++)
		vec[i] = vec1[i] + vec2[i];

	return vec;
}