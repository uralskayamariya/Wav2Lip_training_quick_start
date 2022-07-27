# **Wav2Lip**: Обучение модели генерации движения губ

В репозитории собран материал про быстрому запуску обучения модели Wav2Lip.

Модель позволяет на основе входного изображения или видео человека, а также аудиофайла речи того же или любого другого человека получить сгенерированное гибридное видео.

Таким образом, можно сделать так, что любой человек будет говорить любую речь, содержащуюся во входном аудиофайле, и при этом движения его губ будут соответствовать тому, что он говорит.

Выполнено на основе репозитория:
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip.git)

# Обучение
## Требования к компьютеру
Репозиторий протестирован на:
- Windows 10
- Видеокартах Nvidia RTX 2080 Super, Nvidia RTX 3060.
- Cuda Toolkit 11.x

## Установка
### В Windows
1. Скачайте репозиторий: git clone https://github.com/uralskayamariya/Wav2Lip_training_quick_start.git
3. Отредактируйте последнюю строку в файле готовой среды Anaconda Wav2Lip.yml, который лежит в корне скачаного репозитория:
  <br>![image](https://user-images.githubusercontent.com/86780783/180801441-460162d3-8aae-4dac-9fed-1db042315a54.png)
  <br>Здесь нужно указать путь, по которому Anaconda создает виртуальные среды на Вашем компьютере.
3. Откройте командную строку Anaconda
4. Установите виртуальную среду Anaconda с помощью команды: conda env create --force -f Wav2Lip.yml
5. Скачайте библиотеку ffmpeg: 
  https://github.com/BtbN/FFmpeg-Builds/releases
  <br>Выберите файл: ffmpeg-n5.0-latest-win64-gpl-5.0.zip
  <br>Распакуйте данный архив и добавьте путь к папке ffmpeg-n5.0-latest-win64-gpl-5.0/bin в системный и в пользовательский Path переменных среды Windows.
6. Файлы объемом более 50 Мб не загружаются на github, поэтому [Скачайте](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) файл модели для детектирования лиц,
  <br>переименуйте скачанный файл в s3fd.pth,
  <br>положите переименованный файл по адресу face_detection/detection/sfd.
7. Для fine-tunung скачайте также файлы предобученных моделей и положите их в папку models:
    - [Wav2Lip](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)
    - [Wav2Lip + GAN](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW)
    - [Expert Discriminator](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP)
    - [Visual Quality Discriminator](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo)

Для установки пакетов без Anaconda воспользуйтесь следующей инструкцией:


### В Google Colab
Все действия по установке выполняются аналогично Windows.

Для установки пакетов без Anaconda воспользуйтесь следующей инструкцией:
1. pip uninstall tensorflow tensorflow-gpu
2. cd Wav2Lip && pip install -r requirements.txt
3. 


## Подготовка данных
В качестве входных данных для обучения модели требуются:
1. изображения лиц людей, вырезанные последовательно из видео говорящего человека;
2. звуковой файл говорящего человека, вырезанный из того же видео.
Звуковая дорожка должна соответствовать движению губ на кадрах.

Входные данные можно подготовить из видео двумя разными способами.
1. Воспользоваться скриптом preprocess.py.
    - Для этого разложите видео по папкам как показано в каталоге data_root.
  <br>Каждое видео должно лежать в папке с пятизначным номером, начиная с 00000, 00001 и т.д.
    - Откройте командную строку Анаконды в созданной виртуальной среде.
    - Перейдите в каталог данного репозитория: cd Wav2Lip_training_quick_start.
    - запустите скрипт: python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/ --batch_size 8
  <br>Результат обработки в виде изображений лиц и звуковых файлов сохранится в каталог lrs2_preprocessed с такой же структурой папок, как были разложены видео. Образец присутствует в репозитории. 

2. Воспользоваться библиотеками для вырезания аудио-дорожек из видео-файлов, а также библотеками для детектирования лиц на изображениях. При этом эпизоды, на которых лицо в кадре отсутствует или не может быть детектировано, должны быть вырезаны как из изображений, так и из звуковых файлов.


## Обучение моделей
1. В папке со скачанным репозиторием Wav2Lip_training_quick_start перейдите в папку filelists. В ней должны находится 3 файла: train.txt, val.txt, test.txt. В каждом из них прописаны пути к папкам с изображениями и аудио-файлами в следующем виде:
<br> data_root/00000
<br> data_root/00001
<br> data_root/00002
<br>...
<br>Для обучающих, валидационных и тестовых данных должны быть прописаны соответствующие номера папок в соотвествующих файлах. Все видео должны быть разными. Для демонстрации структуры файлов я взяла одно и то же видео и для трейна, и для валидации, и для теста.
2. Настройте параметры обучения в файле hparams.py. В частности, для проверки процесса обучения необходимо настроить количество эпох обучения nepochs, интервал сохранения модели checkpoint_interval, размер батча batch_size.
3. Откройте командную строку Анаконды в созданной виртуальной среде.
4. Перейдите в скачанный репозиторий: cd Wav2Lip_training_quick_start.
5. Запустите обучение дискриминатора: python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir checkpoints --checkpoint_path models/lipsync_expert.pth
6. Запустите обучение генератора: python hq_wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir checkpoints --checkpoint_path models/wav2lip_gan.pth --disc_checkpoint_path models/visual_quality_disc.pth --syncnet_checkpoint_path models/lipsync_expert.pth


Более подробная информация по обучению и настройке гиперпараметров будет представлена позже.

### Обучение дискриминатора

### Обучение генератора

## Инференс
Для инференса откройте командную строку и или ноутбук и воспользуйтесь следующими командами:
<br>python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "путь к видеофайлу или изображению" --audio "путь к аудиофайлу"
<br>Например: python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "sample_data/480p.mp4" --audio "sample_data/audio.wav"
<br>На выходе получим входное видео говорящего человека со звуком из аудиофайла. При этом движения губ говорящего человека будут соответствовать речи из аудиофайла.
