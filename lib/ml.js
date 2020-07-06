'use strict';

import * as utils from './utils.js';
import Predictor from './predictor.class.js';


export default {

	activationFunctions: {
		sigmoid: utils.sigmoid,
	},


	errorFunctions: {
		quadraticAverage: utils.quadraticAverage,
	},


	Predictor,

}