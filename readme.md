##### Движок и модель классификатора для обучения библиотеки mnist на javascript

Инициализация. Установка зависимостей
```
# выполнить
npm install
```

Копировать в директорию ./data/ файлы библиотеки mnist  
```
./data/train-labels.idx1-ubyte
./data/train-images.idx3-ubyte
./data/t10k-labels.idx1-ubyte
./data/t10k-images.idx3-ubyte
```

Обучить модель
```
# выполнить
node ./train.js
```

Тестирование в интерфейсе. Запуск локального веб-сервера
```
# выполнить
node ./server.js
```

Локальный веб-сервер
```
http://localhost:8080/index.html
```