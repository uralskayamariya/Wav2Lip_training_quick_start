# **Wav2Lip**: Обучение модели генерации движения губ

В репозитории собран материал про быстрому запуску обучения модели Wav2Lip.

Модель позволяет на основе входного изображения или видео человека, а также аудиофайла речи того же или любого другого человека получить сгенерированное гибридное видео. Таким образом, можно сделать так, что любой человек (из входного изображения или видео) будет говорить любую речь (из входного аудиофайла), и при этом движения его губ будут соответствовать тому, что он говорит.

Выполнено на основе репозитория:
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip.git)

# Требования к компьютеру
На компьютере необходимо наличие видеокарты.
Репозиторий протестирован на:
- Windows 10, Google Colab
- Двух компьютерах с видеокартами: Nvidia RTX 2080 Super и Cuda Toolkit 11.4, Nvidia RTX 3060 и Cuda Toolkit 11.3.

# Установка
## Установка в Windows
1. Скачайте библиотеку ffmpeg (используется для подготовки датасета): 
  https://github.com/BtbN/FFmpeg-Builds/releases
  <br>Выберите файл: ffmpeg-n5.0-latest-win64-gpl-5.0.zip
  <br>Распакуйте данный архив и добавьте путь к папке ffmpeg-n5.0-latest-win64-gpl-5.0/bin в системный и в пользовательский Path переменных среды Windows.
2. Откройте командную строку и скачайте репозиторий: 
<br>git clone https://github.com/uralskayamariya/Wav2Lip_training_quick_start.git
3. Файлы объемом более 50 Мб не загружаются на github, поэтому:
  <br>[скачайте](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) файл модели для детектирования лиц,
  <br>переименуйте скачанный файл в s3fd.pth,
  <br>положите переименованный файл по адресу face_detection/detection/sfd.
4. Скачайте также файлы предобученных моделей и положите их в папку models:
    - [Wav2Lip](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)
    - [Wav2Lip + GAN](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW)
    - [Expert Discriminator](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP)
    - [Visual Quality Discriminator](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo)
### Установка через загрузку готовой виртуальной среды Anaconda
1. Отредактируйте последнюю строку в файле готовой среды Anaconda Wav2Lip.yml, который лежит в корне скачаного репозитория:
  <br>![image](https://user-images.githubusercontent.com/86780783/180801441-460162d3-8aae-4dac-9fed-1db042315a54.png)
  <br>Здесь нужно указать путь, по которому Anaconda создает виртуальные среды на Вашем компьютере.
2. Откройте командную строку Anaconda.
3. Установите виртуальную среду Anaconda с помощью команды: conda env create --force -f Wav2Lip.yml

Если виртуальная среда Anaconda по каким-то причинам не установилась воспользуйтесь следующей инструкцией:
### Самостоятельная установка виртуальной среды Anaconda
1. Откройте командную строку Anaconda
2. Создайте виртуальную среду Anaconda:
  <br>conda create -n Wav2Lip python=3.8
3. Активируйте созданную среду:
  <br>conda activate Wav2Lip
4. В командной строке Anaconda перейдите в скачанный репозиторий:
  <br>cd Wav2Lip_training_quick_start
6. Установите необходимые библиотеки:
  <br>pip install -r requirements_win.txt
7. Установите PyTorch:
  <br>pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  
### Установка в Google Colab
1. В управлении средами подключите аппаратный ускоритель GPU.
2. Подключите свой Google Drive:
    <br>from google.colab import drive
    <br>drive.mount('/content/drive')
3. Перейдите в корневую папку своего диска:
    <br>%cd /content/drive/My Drive
4. Скопируйте данный репозиторий:
    <br>!git clone https://github.com/uralskayamariya/Wav2Lip_training_quick_start.git
5. Удалите tensorflow:
    <br>!pip uninstall tensorflow tensorflow-gpu
7. Перейдите в скачанный репозиторий и установите дополнительные пакеты:
    <br>!cd Wav2Lip_training_quick_start && pip install -r requirements.txt
8. Скачайте модель для детектирования лиц:
    <br>!wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "Wav2Lip_training_quick_start/face_detection/detection/sfd/s3fd.pth"
9. Скачайте предобученные модели для обучения и инференса:
    - !wget "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW" -O "Wav2Lip_training_quick_start/models/wav2lip.pth"
    - !wget "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW" -O "Wav2Lip_training_quick_start/models/wav2lip_gan.pth"
    - !wget "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP" -O "Wav2Lip_training_quick_start/models/lipsync_expert.pth"
    - !wget "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo" -O "Wav2Lip_training_quick_start/models/visual_quality_disc.pth"

# Подготовка данных
Для обучения моделей потребуются видеофайлы говорящих людей.
Из видеофайлов формируются изображения лиц говорящего человека для каждого кадра, а также аудиофайлы для каждого видео. Дополнительно необходимо сформировать списки путей к обучающим, валидационным и тестовым данным.
## Подготовка изображений и звуковых файлов
В качестве входных данных для обучения модели требуются:
1. Изображения лиц людей, вырезанные последовательно из видео говорящего человека;
2. Звуковой файл говорящего человека, вырезанный из того же видео.
Звуковая дорожка должна соответствовать движению губ на кадрах. Источником звука может быть любой файл, поддерживаемый ffmpeg, например: *.wav, *.mp3 или даже видеофайл, из которого код автоматически извлекает аудио.

Входные данные можно подготовить из видео двумя разными способами.
1. Воспользоваться скриптом preprocess.py.
    - Для этого разложите видео по папкам как показано в каталоге data_root.
  <br>Каждое видео должно лежать в папке с пятизначным номером, начиная с 00000, 00001 и т.д.
    - Откройте командную строку Анаконды в созданной виртуальной среде.
    - Перейдите в каталог данного репозитория: cd Wav2Lip_training_quick_start.
    - запустите скрипт: python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/ --batch_size 8
  <br>Результат обработки в виде изображений лиц и звуковых файлов сохранится в каталог lrs2_preprocessed с такой же структурой папок, как были разложены видео. Образец присутствует в репозитории. 

2. Воспользоваться любыми библиотеками для вырезания аудио-дорожек из видео-файлов (например, ffmpeg), а также библиотеками для детектирования лиц на изображениях (например, dlib). При этом эпизоды, на которых лицо в кадре отсутствует или не может быть детектировано, должны быть вырезаны как из изображений, так и из звуковых файлов.

## Подготовка списков обучающих, валидационных и тестовых данных
В папке со скачанным репозиторием Wav2Lip_training_quick_start перейдите в папку filelists. В ней должны находится 3 файла: train.txt, val.txt, test.txt. В каждом из них прописаны пути к папкам с изображениями и аудио-файлами в следующем виде:
<br> data_root/00000
<br> data_root/00001
<br> data_root/00002
<br>...
<br>Для обучающих, валидационных и тестовых данных должны быть прописаны соответствующие номера папок в соотвествующих файлах. Все видео должны быть разными. Для демонстрации структуры файлов я взяла одно и то же видео и для трейна, и для валидации, и для теста.
<br>Для обучения моделей требуется прописать более одной папки в файле train.txt, иначе обучение не запустится.

# Обучение моделей (одинаково для Windows и Colab)
Обучение генеративных сетей основано на обучении дискриминатора и генератора.
В процессе установки были скачаны предобученные модели дискриминатора и генератора. Они подходят для инференса.
Для получения оптимального результата для ваших данных необходимо дообучить как дискриминтор, так и генератор.

## Весь процесс обучения
1. Откройте командную строку Анаконды в созданной виртуальной среде.
2. Перейдите в скачанный репозиторий: cd Wav2Lip_training_quick_start.
3. Запустите обучение дискриминатора: python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir checkpoints --checkpoint_path models/lipsync_expert.pth
<br>Для получения хорошего качества loss дискриминатора должна снизиться до ~ 0,25.
4. После того, как дискриминатор обучен запустите обучение генератора. Для обучения генератора в репозитории присутствует две модели: с GAN и без GAN.
    - Запуск обучения модели с GAN:
<br>python hq_wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir checkpoints --checkpoint_path models/wav2lip_gan.pth --disc_checkpoint_path models/visual_quality_disc.pth --syncnet_checkpoint_path models/lipsync_expert.pth
    - Запуск обучения модели без GAN:
<br>python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir checkpoints --syncnet_checkpoint_path models/lipsync_expert.pth --checkpoint_path models/wav2lip.pth
<br>Вместо lipsync_expert.pth указывается путь к файлу модели дискриминатора, обученного на предыдущем шаге.
<br>Для получения хорошего качества loss генератора на валидации должна снизиться до ~ 0,2.

<br> В случае обучения модели с GAN можно добиться лучшего качества результата.


## Настройка гиперпараметров
Настроить гиперпараметры обучения можно в файле hparams.py. Например, можно настроить количество эпох обучения - nepochs, интервал сохранения модели - checkpoint_interval, размер батча - batch_size.
<br>Рекомендуется экспериментировать с настройкой параметров после того, как запуск процесса обучения начал работать с базовыми параметрами файла hparams.py.
<br>Дополнительные рекомендации по настройке процесса обучения присутствуют [здесь](https://github.com/Rudrabha/Wav2Lip.git).


# Инференс
Откройте командную строку Anaconda и или ноутбук и воспользуйтесь следующими командами:
1. Перейдите в каталог данного репозитория:
<br>cd Wav2Lip_training_quick_start
2. Запустите инференс:
<br>python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "путь к видеофайлу или изображению" --audio "путь к аудиофайлу"
<br>Например: python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "sample_data/foto.png" --audio "sample_data/audio.wav"

<br>На выходе получим видео говорящего человека (из входного видео или изображения) со звуком из входного аудиофайла. При этом движения губ говорящего человека будут соответствовать речи из аудиофайла.
Результат сохраняется по умолчанию в results/result_voice.mp4. Этот путь можно изменить.