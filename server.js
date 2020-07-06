'use strict';

import _fs from 'fs';
import _http from 'http';
import _path from 'path';
import _url from 'url';

const
	MODULES = {
		fs      : _fs,
		http    : _http,
		path    : _path,
		url     : _url,
	};

const __dirname = MODULES.path.dirname(MODULES.url.fileURLToPath(import.meta.url));

console.log(`__dirname = ` + __dirname);

MODULES.http.createServer((req, res) => {
	let path = req.url.split('?')[0];

	path = MODULES.path.join(__dirname, path);

	console.log(path, __dirname, path);

	let ext = MODULES.path.extname(path).toLowerCase();

	MODULES.fs.readFile(path, (err,data) => {
		if (err) {
			res.writeHead(404);
			res.end(JSON.stringify(err));

			return;
		}

		if (ext === '.css')
			res.setHeader('Content-type','text/css');

		else if (ext === '.html')
			res.setHeader('Content-type','text/html');

		else if (ext === '.txt')
			res.setHeader('Content-type','text/plain');

		else if (ext === '.js')
			res.setHeader('Content-type','application/javascript');

		else if (ext === '.json')
			res.setHeader('Content-type','application/json');

		res.writeHead(200);
		res.end(data);
	});
}).listen(8080);