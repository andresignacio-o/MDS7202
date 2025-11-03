# hiring_functions.py

from pathlib import Path
import pandas as pd
# Módulos de Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib # Para guardar y cargar el modelo entrenado
import gradio as gr # Para la interfaz de usuario

# --------------------------------------------------------------------------------------
def create_folders(**kwargs):
    """
    Función responsable de crear una estructura de directorios basada en la fecha de ejecución del DAG.
    """
    # Se extrae la fecha de ejecución del DAG utilizando la clave 'ds' de los argumentos.
    execution_date = kwargs['ds']

    # Se establece la ruta base de la carpeta utilizando la fecha de ejecución.
    base_path = Path(execution_date)

    # Se definen los nombres de las subcarpetas requeridas.
    subfolders = ['raw', 'splits', 'models']

    # Se construye la lista de rutas completas a crear mediante una list comprehension.
    paths_to_create = [base_path / subfolder for subfolder in subfolders]

    # Se emplea la función 'map' para crear todos los directorios de forma funcional.
    # 'parents=True' garantiza la creación de la carpeta base (fecha).
    # 'exist_ok=True' previene errores si la carpeta ya existe, eludiendo la necesidad de 'if'.
    list(map(lambda p: p.mkdir(parents=True, exist_ok=True), paths_to_create))

    # Se notifica el resultado de la operación.
    return f"Estructura de carpetas creada con éxito en ./{execution_date}"

# --------------------------------------------------------------------------------------

def split_data(**kwargs):
    """
    Función para realizar un hold-out (train-test split) estratificado y guardar los resultados.
    """
    # Se segmenta el DataFrame: X (características) y Y (variable objetivo).
    # Se asume que la última columna (HiringDecision) es la variable a predecir.
    execution_date = kwargs['ds']
    raw_path = Path(execution_date) / 'raw' / 'data_1.csv'
    splits_path = Path(execution_date) / 'splits'
    data = pd.read_csv(raw_path)
    X = data.drop(columns=['HiringDecision']) # Se remueve la columna objetivo
    y = data['HiringDecision'] # Se define la columna objetivo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Se concatenan de nuevo para guardarlos
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    train_set.to_csv(splits_path / 'train.csv', index=False)
    test_set.to_csv(splits_path / 'test.csv', index=False)
    return "División de datos (entrenamiento y prueba) completada y guardada en la carpeta 'splits'."

# --------------------------------------------------------------------------------------

def preprocess_and_train(**kwargs):
    """
    Función para entrenar un Pipeline con preprocesamiento (ColumnTransformer) y un modelo
    (RandomForest), guardar el modelo y reportar las métricas de evaluación.
    """
    # Se guarda el Pipeline entrenado en la carpeta 'models' en formato joblib.
    # Se reportan los resultados de la evaluación.
    execution_date = kwargs['ds']
    splits_path = Path(execution_date) / 'splits'
    models_path = Path(execution_date) / 'models'
    
    # Se cargan los datos de entrenamiento
    train_set = pd.read_csv(splits_path / 'train.csv')
    test_set = pd.read_csv(splits_path / 'test.csv')
    
    # Se definen X e y (asumiendo que HiringDecision es la última columna o se dropea)
    X_train = train_set.drop(columns=['HiringDecision'])
    y_train = train_set['HiringDecision']
    X_test = test_set.drop(columns=['HiringDecision'])
    y_test = test_set['HiringDecision']
    
    # Se clasifican las características. Dado que Gender, EducationLevel y RecruitmentStrategy
    # están codificadas numéricamente, todas se tratan como características numéricas.
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist() # Captura todas las 10 características

    # Se define el preprocesamiento: Imputación por mediana y Escalamiento Estándar.
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    
    # Nota: No se requiere transformer categórico ya que todas las variables están pre-codificadas numéricamente.
    # No obstante, se mantiene la estructura ColumnTransformer para manejar ambos casos, aunque 'cat' esté vacío.
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features) # Lista vacía si no hay strings
        ],
        remainder='passthrough'
    )
    
    # Se construye el Pipeline completo.
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
    
    # Se realiza el entrenamiento.
    pipeline.fit(X_train, y_train)
    
    # Se realiza la evaluación.
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    # Se guarda el modelo.
    joblib.dump(pipeline, models_path / 'trained_pipeline.joblib')
    
    # Se reportan los resultados.
    print(f"\n--- Evaluación del Modelo ---")
    print(f"Accuracy en conjunto de prueba: {accuracy:.4f}")
    print(f"F1-score (Clase Positiva 'Contratado'): {f1:.4f}")
    return "Pipeline de preprocesamiento y entrenamiento completado. Modelo guardado en 'models/trained_pipeline.joblib'."
    
# --------------------------------------------------------------------------------------

def gradio_interface(**kwargs):
    """
    Función que construye la interfaz de predicción Gradio para el modelo entrenado,
    utilizando las 10 características definidas.
    """
    # Se extrae la fecha de ejecución para construir la ruta de acceso al modelo.
    execution_date = kwargs['ds']
    
    # Se define la ruta completa del archivo del modelo.
    model_path = Path(execution_date) / 'models' / 'trained_pipeline.joblib'

    # Se define la función auxiliar que realiza la predicción.
    # Acepta las 10 características en el orden definido.
    def predict_hiring(Age, Gender, EducationLevel, ExperienceYears, PreviousCompanies, 
                       DistanceFromCompany, InterviewScore, SkillScore, PersonalityScore, RecruitmentStrategy):
        
        # Se carga el Pipeline entrenado desde la ruta especificada.
        pipeline = joblib.load(model_path)
        
        # Se construye el DataFrame de entrada con los nombres de columna correctos.
        input_data = pd.DataFrame({
            'Age': [Age], 
            'Gender': [Gender], 
            'EducationLevel': [EducationLevel], 
            'ExperienceYears': [ExperienceYears], 
            'PreviousCompanies': [PreviousCompanies], 
            'DistanceFromCompany': [DistanceFromCompany], 
            'InterviewScore': [InterviewScore], 
            'SkillScore': [SkillScore], 
            'PersonalityScore': [PersonalityScore],
            'RecruitmentStrategy': [RecruitmentStrategy]
        })
        
        # Se realiza la predicción con el Pipeline.
        prediction = pipeline.predict(input_data)[0]
        
        # Se utiliza un diccionario para mapear la predicción numérica (0 o 1) a etiquetas.
        label_map = {0: "No Contratado", 1: "Contratado"}
        
        # Se retorna el resultado de la predicción.
        return label_map.get(prediction, "Error de Predicción")

    # Se definen los 10 componentes de entrada para Gradio.
    inputs = [
        gr.Number(label="Age (Edad)"),
        gr.Radio({0: 'Male', 1: 'Female'}, label="Gender (Género)"), # Mapeo 0/1
        gr.Dropdown({1: 'Licenciatura Tipo 1', 2: 'Licenciatura Tipo 2', 3: 'Maestría', 4: 'PhD'}, label="EducationLevel (Nivel Educacional)"), # Mapeo 1-4
        gr.Number(label="ExperienceYears (Años de Experiencia)"),
        gr.Number(label="PreviousCompanies (N° Compañías Anteriores)"),
        gr.Number(label="DistanceFromCompany (Distancia a la Compañía en km)"),
        gr.Number(label="InterviewScore (Puntaje Entrevista 0-100)"),
        gr.Number(label="SkillScore (Puntaje Habilidades 0-100)"),
        gr.Number(label="PersonalityScore (Puntaje Personalidad 0-100)"),
        gr.Radio({1: 'Agresiva', 2: 'Moderada', 3: 'Conservadora'}, label="RecruitmentStrategy (Estrategia Reclutamiento)"), # Mapeo 1-3
    ]
    
    # Se define el componente de salida.
    output = gr.Textbox(label="HiringDecision: Resultado de la Contratación")

    # Se crea el objeto de la interfaz Gradio.
    interface = gr.Interface(
        fn=predict_hiring,
        inputs=inputs,
        outputs=output,
        title="Modelo de Predicción de Contratación (Uso de las 10 Características)"
    )

    # Se notifica la finalización.
    return "Interfaz Gradio definida, lista para ser desplegada en un entorno de servidor."