'use strict';

const
	MODULES = {
		fs      : require('fs'),
		http    : require('http'),
		path    : require('path'),
	};

MODULES.http.createServer((req, res) => {
	let path = req.url.split('?')[0];

	path = MODULES.path.join(__dirname, path);

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