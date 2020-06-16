# MelGAN
Неофициальная реализация [MelGAN](https://arxiv.org/abs/1910.06711)
![Иллюстрация к проекту](https://github.com/vlomme/MelGan-WavGan/blob/master/scheme.png)

## Пример использования на [Сolab](https://colab.research.google.com/github/vlomme/MelGan-WavGan/blob/master/MELGAN.ipynb)

## Использование
### Настройки
Отредактируйте hparams.py

Поместите аудиофайлы для тренировки в data_dir

Поместите аудиофайлы для тестирования в test_dir

Поместите аудиофайлы для использования в generate_dir

### Предобработка
Запустите `python melgan.py -r p -f mel`

### Обучение
Запустите `python melgan.py -r t -f mel`

### Использование
Запустите `python melgan.py -r g -f wav` для файлов в wav

Или запустите `python melgan.py -r g -f mel` для файлов в mel, в нужном формате

# WavGAN
Моя сеть. На вход принимает сгенерированный Гриффин лим звук и пытается его подправить. Архитектура генератора похожа на U-net, а дискриминатор взят из MelGAN

## Пример использования на [Сolab](https://colab.research.google.com/github/vlomme/MelGan-WavGan/blob/master/MELGAN.ipynb)

## Использование
### Настройки
Отредактируйте hparams.py

Поместите аудиофайлы для тренировки в data_dir

Поместите аудиофайлы для тестирования в test_dir

Поместите аудиофайлы для использования в generate_dir

### Предобработка
запустите `python wavgan.py -r p -f mel`, или используйте мелспектрограммы из melgan.

Или запустите `python wavgan.py -r p -f wav`, если хотите сразу сгенерировать сигнал Гриффин Лима для ускорения обучения

### Обучение
Запустите `python wavgan.py -r t -f mel` для обучения на mel. Звук будет долго синтезироваться Гриффин Лимом на лету

Или запустите `python wavgan.py -r t -f wav` для обучения на заранее сгенерированных wav.

### Использование
Запустите `python wavgan.py -r g -f wav` для файлов в wav

Или запустите `python melgan.py -r g -f mel` для файлов в mel, в нужном формате. Звук будет сначала синтезироваться Гриффин Лимом
