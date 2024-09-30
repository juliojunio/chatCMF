import time
import streamlit as st
import os
import base64
import nltk

# Agregar stopwords
nltk.data.path.append('./nltk_data')



#guardar en cache para solo entrenar el modelo una vez
@st.cache_resource
def generar_modelo():
    #generar carpeta de cache para descargas
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    os.environ['NLTK_DATA'] = nltk_data_dir

    from llama_index.legacy import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex
    from llama_index.legacy import LLMPredictor
    from langchain_openai import ChatOpenAI
    import textwrap
    import json
    #from dotenv import load_dotenv
    import pickle
    import openai


    ruta=os.getcwd()
    ruta_archivos = os.path.join(os.getcwd(), 'liquidez')

    ############################               ENTRENAMIENTO DEL MODELO     ###############################################

    #agregar mi key de openai desde la variable de entorno
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    api_key = st.secrets["OPENAI_API_KEY"]
    #print(api_key)
   

    #Leer los PDFs
    pdf = SimpleDirectoryReader(ruta_archivos).load_data()


    #Definir e instanciar el modelo
    modelo = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name='gpt-4-turbo-preview',openai_api_key=api_key))
    service_context = ServiceContext.from_defaults(llm_predictor=modelo)
    index = GPTVectorStoreIndex.from_documents(pdf, service_context = service_context)

    filepath = "index.pkl"
    if not os.path.exists(filepath):
        print("El archivo no existe en la ruta especificada")
    elif not os.access(filepath, os.R_OK):
        print("No se tienen permisos de lectura para el archivo")
    else:
        print("El archivo existe y se puede leer")

    '''
    with open('index.pkl', 'rb') as f:
        index = pickle.load(f)
    '''

    ##########################         GENERACÓN DE ARCHIVOS CON HISTORIAL DE CONVERSACIÓN    ########################


    # Archivo para almacenar las interacciones
    interactions_file_norm_v2 = 'interactions_norm_v2.json' #bot con conociemiento normativo que sabe leer histórico (rootprompt)

    # Definir los rootprompts
    rootprompts = [
        "Por favor, ten en cuenta que quiero respuestas detalladas y específicas.",
        "Recuerda ser claro y conciso en tus respuestas.",
        "Responde en español y asegúrate de proporcionar ejemplos cuando sea posible."
    ]



    # Función para cargar interacciones pasadas
    def load_interactions(interactions_file):
        if os.path.exists(interactions_file):
            with open(interactions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    # Función para guardar nuevas interacciones
    def save_interaction(interactions_file,mensaje, isQuestion = True):
        interactions = load_interactions(interactions_file)
        if isQuestion: interactions.append({'role':'user','content': mensaje})
        else: interactions.append({'role':'assistant','content': mensaje})
        with open(interactions_file, 'w', encoding='utf-8') as f:
            json.dump(interactions, f, ensure_ascii=False, indent=4)
        return

    #reiniciar interacciones
    def reset_interactions(interactions_file,system_message='\n'.join(rootprompts)):
        interactions = [{'role':'system','content': system_message}]
        with open(interactions_file, 'w', encoding='utf-8') as f:
            json.dump(interactions, f, ensure_ascii=False, indent=4)
        return



    ########################## GENERAR BOT CAPAZ DE LEER HISTORIALES     ##############################################

    #root prompmt
    root_prompt_normativo='''CONTEXTO: Eres un experto en normativa CMF/SBIF. Tu rol es responder preguntas de manera concisa y en español acerca de esta \
        normativa, incorporando detalles técnicos y ejemplos cuando sea posible.
        No respondas preguntas sobre cómo modificar o adaptar la normativa. Limítate a explicar el contenido de la normativa tal como está definida.
        Si una pregunta se refiere a otras normativas que no están incluidas en el documento, responde: 'Esta normativa no está relacionada con la pregunta formulada. Por favor, consulta solo la normativa proporcionada.'
        Evita responder preguntas hipotéticas o especulativas. Si la pregunta no está relacionada directamente con el contenido de la normativa, responde: 'Este chatbot no puede realizar suposiciones. Solo puedo proporcionar respuestas basadas en la normativa dada.'

    INPUT: A contnuación te proporcionaré un historial de conversación, que incluye una serie de preguntas y respuestas. El formato será una serie de líneas \
        de texto indicando <rol><contenido>. <rol> puede tener valores "usuario" o "asistente"; "usuario" representa las preguntas hechas, y "asistente" las \
            respuestas entregadas. 

    ANÁLSIS: Simular internamente toda a conversación entregada, y generar una respuesta a la última pregunta de esta.

    EJEMPLO:
        <historial 1> 
        usuario: dame un número al azar de 1 a 10
        asistente: 7
        usuario: y el doble de ese número
        <\historial 1> 
        <respuesta 1> el doble de 7 es 14 <\respuesta 1>
    '''

    reset_interactions(interactions_file_norm_v2,root_prompt_normativo)

    #convierte la lista de diccionarios de un json a un texto legible por el bot normativo v2
    def generate_histo_text(conversation_list):
        # Extraer la información del primer diccionario con la llave "system"
        system_info = ""
        for entry in conversation_list:
            if entry['role'] == 'system':
                system_info = entry['content']
                break

        # Construir el historial de conversación
        historial = []
        for entry in conversation_list:
            if entry['role'] == 'user':
                historial.append(f"usuario: {entry['content']}")
            elif entry['role'] == 'assistant':
                historial.append(f"asistente: {entry['content']}")

        # Formatear el string final
        historial_string = "\n".join(historial)
        final_string = f"[{system_info}]\n<HISTORIAL>\n{historial_string}\n<\\HISTORIAL>"

        return final_string

    def interaccion_norm_v2(pregunta):
        save_interaction(interactions_file_norm_v2,pregunta)

        #generar respuesta
        historial=load_interactions(interactions_file_norm_v2)
        historial_text=generate_histo_text(historial) #convertir a string
        respuesta=index.as_query_engine().query(historial_text)
        respuesta_texto = respuesta.response.strip()
        save_interaction(interactions_file_norm_v2,respuesta_texto,isQuestion=False)

        #escribir respuesta
        for frase in textwrap.wrap(respuesta_texto, width=150):
            print(frase)

        return respuesta_texto
    return interaccion_norm_v2
#aqui termina la definición de la función para entrenar el modelo

def mostrar_respuesta_progresivamente(texto, delay=0.015):
    # Inicializa un widget de texto vacío
    respuesta = st.empty()
    
    # Construye progresivamente la respuesta
    texto_parcial = ""
    for letra in texto:
        texto_parcial += letra
        respuesta.markdown(f"{texto_parcial}")
        time.sleep(delay) 

##############################   EJECUTA FRONT CON STREAMLIT      ######################################################


# Función para convertir una imagen local a base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Ruta a tu imagen de fondo local
image_path = os.path.join(os.path.dirname(__file__), 'fondo_opaco.png')  # Cambia esto a la ubicación de tu imagen

# Convertir imagen a base64
image_base64 = get_base64_of_image(image_path)

# Código CSS para agregar imagen de fondo
background_css = f"""
<style>
.stApp {{
    background: url(data:image/jpeg;base64,{image_base64});
    background-size: cover;
    background-attachment: fixed;
}}
</style>
"""

# Inyectar el código CSS
st.markdown(background_css, unsafe_allow_html=True)

#######################################################################



start_time = time.time()
interaccion_norm_v2 = generar_modelo()
end_time = time.time()
response_time = end_time - start_time
st.markdown(f"<small>Tiempo entrenamiento modelo: {response_time:.2f} segundos</small>", unsafe_allow_html=True)

st.title("ChatCMF")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Haz una pregunta"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        stream = interaccion_norm_v2(prompt)
        end_time = time.time()
        response_time = end_time - start_time
        mostrar_respuesta_progresivamente(stream)
        st.markdown(f"<small>Tiempo de respuesta: {response_time:.2f} segundos</small>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": stream})
