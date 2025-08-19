# 🚀 Career Maps Bot  

Este repositorio contiene un código de ejemplo para implementar un **bot desde cero** con una **UI básica**, utilizando servicios de **Azure OpenAI** y almacenamiento en contenedores.  

El bot permite consultar archivos almacenados en un contenedor de Azure para responder preguntas basadas en su contenido. Para ello, utiliza dos agentes de **OpenAI**:  

- 🧩 **Embeddings** → para indexar y buscar en los documentos.  
- 💬 **Chat** → para generar las respuestas finales.  

---

## ⚙️ Requisitos previos  

1. Una cuenta de **Azure** con:  
   - Endpoints y Keys configurados para los servicios de OpenAI.  
   - Un contenedor de **Azure Blob Storage** creado, donde se subirán los archivos a consultar.  

2. Configurar las variables de entorno con las claves y endpoints necesarios (usando un archivo `.env`).  

3. Puedes guiarte del archivo **.env.example** incluido en este repositorio para definir tus variables.  

---

## 📦 Dependencias  

Instala los paquetes requeridos ejecutando:  

```bash
pip install -r requirements.txt
