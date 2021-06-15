# AGAIN-VC + MelGAN + WavGAN
Неофициальная реализация клонирования голоса [AGAIN-VC](https://arxiv.org/pdf/2011.00316.pdf)([код](https://github.com/KimythAnly/AGAIN-VC)), вокодера [MelGAN](https://arxiv.org/abs/1910.06711) и мой вокодер WavGAN

## Пример использования 
### [MelGAN+WavGAN на Сolab](https://colab.research.google.com/github/vlomme/AGAIN-MelGan-Voice-Cloning/blob/master/MELGAN.ipynb). 
### [AGAIN_VC на Сolab](https://colab.research.google.com/github/vlomme/AGAIN-MelGan-Voice-Cloning/blob/master/AGAIN_VC.ipynb). 
### [Предобученные веса](https://drive.google.com/uc?id=10tLduS5fGNWby7IKvfltuIfUWUeAp9SM) многоголосой моделей

# AGAIN_VC
![AGAIN_VC](https://github.com/vlomme/MelGan-WavGan/blob/master/again_model.png)
### Настройки
Отредактируйте hparams.py

Поместите аудиофайлы для тренировки в data_dir

### Предобработка
Запустите `python again.py -r p`

### Обучение
Запустите `python again.py -r t`

### Использование
Запустите `python again.py  -r g -s path_source_file  -t path_target_file`

# MelGAN
![MelGAN](https://github.com/vlomme/MelGan-WavGan/blob/master/melgan_model.png)
### Настройки
Отредактируйте hparams.py

Поместите аудиофайлы для тренировки в data_dir

Поместите аудиофайлы для использования в generate_dir

### Предобработка
Запустите `python melgan.py -r p -f mel`

### Обучение
Запустите `python melgan.py -r t -f mel`

### Использование
Запустите `python melgan.py -r g -f wav` для файлов в wav из которого будет считаться мелспектр

Или запустите `python melgan.py -r g -f mel` для файлов в mel, в нужном формате

# WavGAN
Моя сеть. На вход принимает сгенерированный Гриффин лим звук и пытается его подправить. Архитектура генератора похожа на U-net, а дискриминатор взят из MelGAN

### Настройки
Отредактируйте hparams.py

Поместите аудиофайлы для тренировки в data_dir

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
