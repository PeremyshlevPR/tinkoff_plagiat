import argparse
import re
import ast
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


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
parser.add_argument('input_file', type=str, help='Input text file with names of .py files')
parser.add_argument('output_file', type=str, help='Output text file with scores of plagiarism')
parser.add_argument('--tfidf', type=str, help='Filename for loading tfidf vocabulary')
parser.add_argument('--model', type=str, help='Filename for loading trained classifier')
args = parser.parse_args()

result = []
with open(args.input_file, 'r') as f:
    for line in f:
        file1, file2 = line.split()

        text1 = ''
        text2 = ''

        with open(file1, 'r', encoding='ascii', errors='ignore') as f1:
            for line1 in f1:
                text1 += line1

        with open(file2, 'r', encoding='ascii', errors='ignore') as f2:
            for line2 in f2:
                text2 += line2

        # Удаляем комментарии
        text1 = preprocessing(ast_preprocessing(text1))
        text2 = preprocessing(ast_preprocessing(text2))

        # Разбиваем на токены и создаем корпус токенов
        transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(args.tfidf, "rb")))
        tfidf = transformer.fit_transform(loaded_vec.fit_transform([text1, text2]))

        # Сходство по косинусному расстоянию
        cos_sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]

        if args.model:
            with open('model.pkl', 'rb') as model_file:
                model = pickle.load(model_file)

            # Сходство, предсказанное моделью
            model_pred = model.predict_proba([[abs(x) for x in (tfidf[0] - tfidf[1]).todense().tolist()[0]]])[0, 1]

        else:
            model_pred = cos_sim

        result.append(str(round((cos_sim + model_pred) / 2, 3)))

result = '\n'.join(result)
with open(args.output_file, 'w') as f:
    f.write(result)
