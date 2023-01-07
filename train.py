import ast
import re
import os
import random
import pickle
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def get_corpus(*folders):
    file_names = []
    corpus = []

    for folder in folders:
        folder_corpus = []

        for filename in [filename for filename in os.listdir(folder) if filename.endswith('.py')]:
            full_filename = folder + '\\' + filename
            with open(full_filename, 'r', encoding='ascii', errors='ignore') as file:
                text = file.read()
            corpus.append(text)
            file_names.append(full_filename)

    return file_names, corpus


def ast_preprocessing(text):
    """
    Функция для обработки ast дерева кода, поступающего на вход.
    Выполняется замена имен функций и классов, объявленных в коде.
    Также заменяются имена аргументов и имена переменных, участвующих в операции присвоеня.
    (В случае поступления на вход кода, котоырй невозможно скомпилировать, функция возвращает его, не обрабатывая)

    :param text: Текст программы на Python.
    :return: Обработанный текст программы.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text

    count_classes = 0
    count_functions = 0

    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and
                len(node.body) and
                isinstance(node.body[0], ast.Expr) and
                hasattr(node.body[0], 'value') and
                isinstance(node.body[0].value, ast.Str)):

            node.body = node.body[1:]


        elif isinstance(node, ast.Assign):
            var_type = re.split(r"[. ]", str(type(node.value)))[-1][:-2]

            if var_type == 'Constant':
                var_type = str(type(node.value.value)).split()[1][1:-2]

            node.targets[0].id = 'assign_var' + '_' + var_type


        elif isinstance(node, (ast.ClassDef)):
            node.name = 'class_' + str(count_classes + 1)
            count_classes += 1

            class_count_methods = 0
            for child_node in ast.iter_child_nodes(node):
                if isinstance(child_node, (ast.FunctionDef)):
                    child_node.name = node.name + '_method_' + str(class_count_methods + 1)
                    class_count_methods += 1


        elif isinstance(node, (ast.FunctionDef)) and (not node.name.startswith('class_')):
            node.name = 'function_' + str(count_functions + 1)
            count_functions += 1

            if len(node.args.args):
                node.args.args = [ast.arg(node.name + f'_arg_{i}') for i in range(len(node.args.args))]

    text = ast.unparse(tree)
    return text


def preprocessing(text):
    """
    Функция для предобработки текста, поступающего на вход.
    Удаляет все знаки, прописанные в регулярном выражении,
    приводит все к нижнему регистру, а также разбивает обработанный текст на токены.

    :param text: исходный текст
    :return: текст у удаленными знаками
    """

    return re.sub(r"[().:='\"#\n\[\]{}!/-\\^`><_,-]", ' ', text)



parser = argparse.ArgumentParser(description='Input and Output files')
parser.add_argument('orig_folder', type=str, help='Folder with original texts of python programms')
parser.add_argument('plagiat_folder_1', type=str, help='Folder with plagiated programms')
parser.add_argument('plagiat_folder_2', type=str, help='Folder with plagiated programms')
parser.add_argument('--tfidf', type=str, help='Filename for saving tfidf vocabulary')
parser.add_argument('--model', type=str, help='Filename for saving trained classifier')


args = parser.parse_args()


file_names, corpus = get_corpus(args.orig_folder, args.plagiat_folder_1, args.plagiat_folder_2)

#Выполняем предобработку ast и символьную.
for i in range(len(corpus)):
    corpus[i] = preprocessing(ast_preprocessing(corpus[i]))

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus)

X = [0] * vectors.shape[0] * 2
y = [0] * vectors.shape[0] * 2


# Организуем пары векторов, вычитая один из другого, комбинируем тексты из разных папок.
# Также какждому вектору пары текстов присваиваем метку (1 - плагиат, 0 - НЕ плагиат)
folder_size = vectors.shape[0] // 3
for i in range(0, folder_size):
    X[6 * i] = [abs(x) for x in (vectors[i] - vectors[i + folder_size]).todense().tolist()[0]]
    y[6 * i] = 1

    X[1 + i * 6] = [abs(x) for x in (vectors[i] - vectors[i + 2 * folder_size]).todense().tolist()[0]]
    y[1 + i * 6] = 1

    X[2 + i * 6] = [abs(x) for x in (vectors[i + folder_size] - vectors[i + 2 * folder_size]).todense().tolist()[0]]
    y[2 + i * 6] = 1

    X[3 + i * 6] = [abs(x) for x in (vectors[i].todense() - random.choice(vectors.todense())).tolist()[0]]
    X[4 + i * 6] = [abs(x) for x in (vectors[i].todense() - random.choice(vectors.todense())).tolist()[0]]
    X[5 + i * 6] = [abs(x) for x in (vectors[i].todense() - random.choice(vectors.todense())).tolist()[0]]


#Создаем классификатор и прогоняем через сетку для поиска наилучших параметров.
logreg = LogisticRegression()
logreg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

logreg_grid = GridSearchCV(logreg, logreg_params)
logreg_grid.fit(X, y)
best_logreg = logreg_grid.best_estimator_


# Сохраняем модель tfidf и классификатор
if args.tfidf:
    with open(args.tfidf, "wb") as tfidf_file:
        pickle.dump(vectorizer.vocabulary_, tfidf_file)

if args.model:
    with open(args.model, "wb") as model_file:
        pickle.dump(best_logreg, model_file)

