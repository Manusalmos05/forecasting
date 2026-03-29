# Proyecto de Forecasting con Machine Learning

Proyecto de machine learning para predicción y análisis de datos.

## Estructura del Proyecto

```
.
├── data/
│   ├── raw/                 # Datos originales sin procesar
│   └── processed/           # Datos procesados y listos para ML
├── notebooks/               # Jupyter notebooks para análisis y experimentos
├── models/                  # Modelos entrenados guardados
├── src/                     # Código fuente
│   └── app.py              # Aplicación Streamlit
├── docs/                    # Documentación
├── requirements.txt         # Dependencias del proyecto
├── .gitignore              # Archivos a ignorar en Git
└── README.md               # Este archivo
```

## Requisitos

- Python 3.8+
- pip

## Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd forcasting
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar la aplicación Streamlit:
```bash
streamlit run src/app.py
```

### Ejecutar notebooks:
```bash
jupyter notebook
```

## Estructura de Datos

- **data/raw/**: Almacenar datos crudos originales
- **data/processed/**: Datos limpios y transformados

## Modelos

Los modelos entrenados se guardan en la carpeta `models/` en formato pickle o joblib.

## Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está bajo licencia MIT.
