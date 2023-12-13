
from flask import Flask, render_template, request
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

app = Flask(__name__)

stopwordsES = set(['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'fue', 'este', 'ha', 'sí', 'porque', 'esta', 'son', 'entre', 'está', 'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'están', 'estado', 'desde', 'todo', 'nos', 'durante', 'estados', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'fueron', 'ese', 'eso', 'había', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas', 'estoy', 'estás', 'está', 'estamos', 'estáis', 'están', 'esté', 'estés', 'estemos', 'estéis', 'estén', 'estaré', 'estarás', 'estará', 'estaremos', 'estaréis', 'estarán', 'estaría', 'estarías', 'estaríamos', 'estaríais', 'estarían', 'estaba', 'estabas', 'estábamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos', 'estuvisteis', 'estuvieron', 'estuviera', 'estuvieras', 'estuviéramos', 'estuvierais', 'estuvieran', 'estuviese', 'estuvieses', 'estuviésemos', 'estuvieseis', 'estuviesen', 'estando', 'estado', 'estada', 'estados', 'estadas', 'estad', 'he', 'has', 'ha', 'hemos', 'habéis', 'han', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayan', 'habré', 'habrás', 'habrá', 'habremos', 'habréis', 'habrán', 'habría', 'habrías', 'habríamos', 'habríais', 'habrían', 'había', 'habías', 'habíamos', 'habíais', 'habían', 'hube', 'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron', 'hubiera', 'hubieras', 'hubiéramos', 'hubierais', 'hubieran', 'hubiese', 'hubieses', 'hubiésemos', 'hubieseis', 'hubiesen', 'habiendo', 'habido', 'habida', 'habidos', 'habidas', 'soy', 'eres', 'es', 'somos', 'sois', 'son', 'sea', 'seas', 'seamos', 'seáis', 'sean', 'seré', 'serás', 'será', 'seremos', 'seréis', 'serán', 'sería', 'serías', 'seríamos', 'seríais', 'serían', 'era', 'eras', 'éramos', 'erais', 'eran', 'fui', 'fuiste', 'fue', 'fuimos', 'fuisteis', 'fueron', 'fuera', 'fueras', 'fuéramos', 'fuerais', 'fueran', 'fuese', 'fueses', 'fuésemos', 'fueseis', 'fuesen', 'sintiendo', 'sentido', 'sentida', 'sentidos', 'sentidas', 'siente', 'sentid', 'tengo', 'tienes', 'tiene', 'tenemos', 'tenéis', 'tienen', 'tenga', 'tengas', 'tengamos', 'tengáis', 'tengan', 'tendré', 'tendrás', 'tendrá', 'tendremos', 'tendréis', 'tendrán', 'tendría', 'tendrías', 'tendríamos', 'tendríais', 'tendrían', 'tenía', 'tenías', 'teníamos', 'teníais', 'tenían', 'tuve', 'tuviste', 'tuvo', 'tuvimos', 'tuvisteis', 'tuvieron', 'tuviera', 'tuvieras', 'tuviéramos', 'tuvierais', 'tuvieran', 'tuviese', 'tuvieses', 'tuviésemos', 'tuvieseis', 'tuviesen', 'teniendo', 'tenido', 'tenida', 'tenidos', 'tenidas', 'tened'])


lista_deseos = []
lista_insultos = ['conche', 'verga', 'vrg', 'mierda', 'chupa', 'maricon', 'care', 'chucha', 'nalga', 'puta', 'pta', 'condon', 'sexo', 'pene', 'pishco', 'puchica', 'uchica', 'tanga', 'menso', 'sonso', 'culeca', 'tonta', 'odio', 'muspa', 'gil',
                  'imbecil', 'muerto', 'shunsho', 'pelao', 'maldigo', 'maldito', 'perra', 'zorra', 'vagina', 'vanjaina', 'peluda', 'sucia', 'callate', 'huevo', 'changar', 'follar', 'lamber', 'choro', 'ladron', 'hijue', 'cachos', 'longo',
                  'cabron', 'pelmaso', 'pelamaso', 'caretuco', 'testiculos', 'ediondo', 'safado', 'culo', 'webada', 'veesijueputa', 'traga', 'grilla', 'putiada', 'regalada', 'lanzada', 'hijueputa', 'caracho', 'rechucha', 'asqueroso', 'marica',
                  'machista', 'arisco', 'chuchatumadre', 'mamahuevo', 'madafaka', 'gran flaut', 'timbrado', 'mamaverga', 'cuaji', 'chupar esa concha', 'chuchacaso', 'saque la mad', 'toma tu nota', 'chuta mad', 'gaver', 'hijuefruta', 'conchetu',
                  'puchica', 'pucha', 'pucha ma', 'cojudo', 'caracho', 'catracho', "shit", 'cancha ma', 'fumado', 'fumado', 'pucta', 'p@t4s', 'arrechar', 'meas', 'ahuevas', 'pinga', 'mrd', 'mrd per', 'mmvrg', 'ñña', 'nalgas', 'cuernos', 'cuernudo',
                  'inmundo', 'yisus', 'que te den', "mal", 'asquerosa', 'asqueroso', 'pudras', 'sufras', 'arruine', 'infierno', 'mueras', 'desastre', 'desmorones', 'caiga ray ', 'desaparecieras',
                  'ahogues', 'patetica', 'culos', 'atropelle', 'ridicula', 'llores', 'pesadilla', 'pierdas', 'desgracias', 'desilusiones', 'insulsa', 'convier gri', 'desdicha', 'insoportable', 'miserable', 'desmorones', 'fracaso', 'aniquilada',
                  'desmoronen', 'desvanezca', 'desesperacion', 'insulsa', 'aburrida nav','muerte','guerra','pandemia','asalto','morirme','mueras']

def deseos_sin_similitud(lista_deseos, lista_insultos):
    todas_las_palabras = lista_deseos + lista_insultos

    # Preprocesamiento NLP
    nlp = [" ".join([unidecode(token.lower().replace(',', '')) for token in doc.split() if unidecode(token.lower()) not in stopwordsES]) for doc in todas_las_palabras]

    lista_deseos_u = nlp

    # Crear el objeto CountVectorizer
    count_vectorizer = CountVectorizer()

    # Ajustar y transformar los documentos para obtener la bolsa de palabras
    bolsa_palabras = count_vectorizer.fit_transform(lista_deseos_u)

    # Calcular la frecuencia del término
    frecuencia_termino = bolsa_palabras.toarray()

    # Calcular la frecuencia de documento (df)
    frecuencia_documento = np.sum(bolsa_palabras.toarray() > 0, axis=0)

    # Calcular el peso IDF
    peso_idf = np.log((1 + len(lista_deseos_u)) / (1 + frecuencia_documento)) + 1

    # Calcular el Peso TF-IDF
    peso_tfidf = frecuencia_termino * peso_idf

    # Aplicar la long-normalización al Peso TF-IDF
    peso_tfidf_normalizado = normalize(peso_tfidf, norm='l2')

    # Calcular la similitud del coseno entre los documentos
    similitud_coseno = cosine_similarity(peso_tfidf_normalizado)

    palabras_clave_insultos = [insulto.lower() for insulto in lista_insultos]

    deseos_sin_similitud = []
    deseos_inapropiados = []
    # 0.378
    for i in range(len(lista_deseos)):
        es_inapropiado = any(palabra_clave in lista_deseos[i].lower().split() for palabra_clave in lista_insultos) or any(
            similitud_coseno[i, j] > 0.2 for j in range(len(lista_deseos), len(todas_las_palabras)))
        if es_inapropiado or any(palabra_clave in lista_deseos[i].lower().split() for palabra_clave in lista_insultos):
            deseos_inapropiados.append(lista_deseos[i])
        else:
            deseos_sin_similitud.append(lista_deseos[i])
    return deseos_sin_similitud, deseos_inapropiados


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     global lista_deseos

#     if request.method == 'POST':
#         nuevo_deseo = request.form['deseo']
#         lista_deseos.append(nuevo_deseo)
#         # Actualiza la lista de deseos sin similitud
#         deseos_sin_similitud_resultado, deseos_con_similitud_resultado = deseos_sin_similitud(lista_deseos, lista_insultos)
#         return render_template('index.html', deseos_sin_similitud=deseos_sin_similitud_resultado)

#     return render_template('index.html')
@app.route('/', methods=['GET', 'POST'])
def index():
    global lista_deseos

    if request.method == 'POST':
        nuevo_deseo = request.form['deseo']
        lista_deseos.append(nuevo_deseo)
        
        # Guarda la lista de deseos en un archivo JSON
        with open('lista_deseos.json', 'w') as file:
            json.dump(lista_deseos, file)

        # Actualiza la lista de deseos sin similitud
        deseos_sin_similitud_resultado, deseos_con_similitud_resultado = deseos_sin_similitud(lista_deseos, lista_insultos)
        return render_template('index.html', deseos_sin_similitud=deseos_sin_similitud_resultado)

    # Carga la lista de deseos desde el archivo JSON
    try:
        with open('lista_deseos.json', 'r') as file:
            lista_deseos = json.load(file)
    except FileNotFoundError:
        pass  # Ignora si el archivo no existe

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)