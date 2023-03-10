<h1 align = 'center'>Вступительный экзамен по программированию <br> Tinkoff Deep Learning 2023</h1>
<hr>

<!-- ![alt text]([https://kassaofd.ru/wp-content/uploads/2021/11/Tinkoff-bank.png]) -->

<h2>Структура проекта: </h2>

<ul>
  <li>	<i>train.py</i> - программа для обучения модели логистической регрессии для классификации плагиата, сохраняет модель в ьинарный файл.</li>
  <li>	<i>compare.py</i> - программа, непосредственно сравнивающая тексты, поданные ей на вход. </li>
  <li>	<i>files</i> – папка с оригинальными файлами программ.</li>
  <li>	<i>plagiat1, plagiat2</i> – в этих папках хранятся измененные версии файлов из files, являющиеся плагиатом.</li>
  <li>	<i>model.pkl</i> – файл с моделью классификатора, обученного на файлах из папок files, plagiat1, plagiat2</li>
  <li>	<i>tfidf.pkl</i> – бинарный файл со словарем всех встречающихся токенов. Необходим для представления текста программы в виде вектора.
  <li>	<i>input.txt</i> - пример входного файла, необходимого для работы compare.py
</ul>

<hr>
<h2>Идея, лежащая в основе метода распознавания: </h2>

<div> Все тексты программ были приведены к "нормальному" виду, то есть, при помощи модуля ast был произведен семантический анализ текста программ: заменены имена переменных,
функций, классов, аргументов и т.д. Также были учтены типы переменных. Удалены все комментарии и аннотации к функциям и классам. 

После нормализации был применен метод TF-IDF, который приводит уже обработанный текст в векторное представление, необходимое для применения логистической регрессии.

Пары документов будем представлять вектором модуля разности от векторов каждого документа. То есть, вычитаем два вектора друг из друга и берем от каждого элемента модуль.
Основная идея заключается в том, что похожие документы при вычитании из векторов друг из друга будут представлять около-нулевой вектор с минимальными различиями,
следовательно, классификатор сможет уловить эту свяь и показать адекватные результаты.

Помимо классификатора было использовано косинусное сходство между двумя векторами. Если два документа похожи друг на друга - то и их вектора будут находиться рядом.
</div>


<h2>Запуск программы: </h2>
<div> <i><b>train.py</i></b>: <br>
Программа train.py принимает на вход имена трех папок: одна с оригинальными документами, две другие - с измененными версиями программ из оригинальной папки, а также
флаг --model с именем файла для сохранения модели.

Функционал заклюючается в создании словаря TF-IDF на основе всех файлов python из трех папок, формировании корпуса векторов пар документов и обучении на нем модели
логистической регрессии. После этого происходит сохранение словаря tf-idf и модели в бинарный файл.

Пример запуска из командной строки:
> python train.py files plagiat1 plagiat2 --model model.pkl


<div> <i><b>compare.py</i></b>: <br>
Программа принимает на вход два обязательных аргумента: input - входной файл с парами программ python, которые необходимо проверить на плагиат. (пример входного файла находится в репозитории)
output - выходной файл, куда будут выведены результаты работы.

Флаг --model: файл с обученной моделью классификации и словарь токенов для построения tf-idf по паре текстов, поданных на вход. 

Пример запуска из командной строки:
> python compare.py input.txt output.txt --model model.pkl
