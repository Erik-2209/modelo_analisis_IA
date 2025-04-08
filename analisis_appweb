import os
import re
import fitz
import whisper
import difflib
import uuid
import json
import datetime
import warnings
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from supabase import create_client, Client
from typing import Optional

app = FastAPI()

# Ignorar advertencias de whisper sobre FP16
warnings.filterwarnings("ignore", category=UserWarning)

# Configuración de Supabase
SUPABASE_URL = "https://qdlmwuflfniifrmeoadl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFkbG13dWZsZm5paWZybWVvYWRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIyNDg0MDQsImV4cCI6MjA1NzgyNDQwNH0.HESG9PdRiG2U_4HuKOMx4QOXi3hCy1Z5eF24JI3kMZw"
BUCKET_PDF = "pruebas"
BUCKET_AUDIO = "audios"
PDF_FILE = "temp_ref.pdf"
AUDIO_FILE = "temp_audio.wav"

# Conectar a Supabase al iniciar la aplicación
@app.on_event("startup")
async def startup_event():
    app.state.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Funciones auxiliares (iguales que en tu script original)
def descargar_archivo(supabase, bucket, archivo_remoto, archivo_local):
    try:
        response = supabase.storage.from_(bucket).download(archivo_remoto)
        with open(archivo_local, "wb") as f:
            f.write(response)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al descargar '{archivo_remoto}': {e}")

def extraer_texto_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        texto = "\n".join(page.get_text("text") for page in doc)
        return texto.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al extraer texto del PDF: {e}")

def transcribir_audio(audio_path):
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(audio_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al transcribir el audio: {e}")

def normalizar_texto(texto):
    return re.sub(r"[^\w\s]", "", texto.lower()).strip()

def analizar_diferencias(texto_ref, texto_audio):
    diferencias = []
    ref_words = texto_ref.split()
    audio_words = texto_audio.split()

    matcher = difflib.SequenceMatcher(None, ref_words, audio_words)

    correctas = 0
    incorrectas = 0

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            correctas += (i2 - i1)
        else:
            incorrectas += max(i2 - i1, j2 - j1)
            diferencias.append({
                "tipo": opcode,
                "palabra_original": " ".join(ref_words[i1:i2]),
                "palabra_dicha": " ".join(audio_words[j1:j2])
            })
    total = correctas + incorrectas
    precision = round((correctas / total) * 100, 2) if total > 0 else 0

    return diferencias, correctas, incorrectas, precision

def calcular_fluidez(transcripcion):
    palabras = transcripcion['text'].split()
    n_palabras = len(palabras)
    duracion = transcripcion['segments'][-1]['end'] if transcripcion['segments'] else 1
    palabras_por_minuto = round(n_palabras / (duracion / 60), 2)

    for i in range(1, len(transcripcion['segments'])):
        transcripcion['segments'][i]['prev_end'] = transcripcion['segments'][i-1]['end']

    pausas = [seg for seg in transcripcion['segments'] if seg['start'] - seg.get('prev_end', 0) > 1.5]
    
    return {
        "palabras_por_minuto": palabras_por_minuto,
        "numero_pausas": len(pausas),
        "duracion_total_segundos": round(duracion, 2)
    }

def subir_estadisticas(supabase, paciente_id, diferencias, correctas, incorrectas, precision, fluidez):
    now = datetime.datetime.now().isoformat()
    
    try:
        for dif in diferencias:
            supabase.table("errores_recurrentes").insert({
                "paciente_id": paciente_id,
                "fecha": now,
                "tipo_error": dif['tipo'],
                "palabra_original": dif['palabra_original'],
                "palabra_dicha": dif['palabra_dicha'],
                "frecuencia": 1,
                "contexto": "Audio vs PDF"
            }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar errores_recurrentes: {e}")

    try:
        supabase.table("precision_pronunciacion").insert({
            "paciente_id": paciente_id,
            "fecha": now,
            "palabras_correctas": correctas,
            "palabras_incorrectas": incorrectas,
            "porcentaje_precision": precision,
            "observaciones": "Audio comparado con texto PDF"
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar precisión_pronunciacion: {e}")

    try:
        supabase.table("fluidez_lectora").insert({
            "paciente_id": paciente_id,
            "fecha": now,
            "palabras_por_minuto": int(float(fluidez['palabras_por_minuto'])),
            "numero_pausas": int(fluidez['numero_pausas']),
            "duracion_total_segundos": int(fluidez['duracion_total_segundos']),
            "observaciones": "Generado automáticamente"
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar fluidez_lectora: {e}")

    return {
        "status": "success",
        "data": {
            "paciente_id": paciente_id,
            "fecha": now,
            "precision": precision,
            "fluidez": fluidez,
            "errores": len(diferencias)
        }
    }

# Endpoint para procesar desde archivos en Supabase
@app.post("/procesar-desde-supabase")
async def procesar_desde_supabase(
    paciente_id: str = Form(...),
    archivo_pdf: str = Form(...),
    archivo_audio: str = Form(...)
):
    try:
        supabase = app.state.supabase
        
        if not descargar_archivo(supabase, BUCKET_PDF, archivo_pdf, PDF_FILE):
            raise HTTPException(status_code=400, detail="Error al descargar PDF")
        if not descargar_archivo(supabase, BUCKET_AUDIO, archivo_audio, AUDIO_FILE):
            raise HTTPException(status_code=400, detail="Error al descargar audio")

        texto_ref = normalizar_texto(extraer_texto_pdf(PDF_FILE))
        transcripcion = transcribir_audio(AUDIO_FILE)
        texto_audio = normalizar_texto(transcripcion['text'])
        
        diferencias, correctas, incorrectas, precision = analizar_diferencias(texto_ref, texto_audio)
        fluidez = calcular_fluidez(transcripcion)

        # Limpiar archivos temporales
        os.remove(PDF_FILE)
        os.remove(AUDIO_FILE)

        return subir_estadisticas(supabase, paciente_id, diferencias, correctas, incorrectas, precision, fluidez)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para procesar con archivos subidos directamente
@app.post("/procesar-archivos-directos")
async def procesar_archivos_directos(
    paciente_id: str = Form(...),
    archivo_pdf: UploadFile = File(...),
    archivo_audio: UploadFile = File(...)
):
    try:
        # Guardar archivos temporalmente
        with open(PDF_FILE, "wb") as buffer:
            buffer.write(archivo_pdf.file.read())
        
        with open(AUDIO_FILE, "wb") as buffer:
            buffer.write(archivo_audio.file.read())

        texto_ref = normalizar_texto(extraer_texto_pdf(PDF_FILE))
        transcripcion = transcribir_audio(AUDIO_FILE)
        texto_audio = normalizar_texto(transcripcion['text'])
        
        diferencias, correctas, incorrectas, precision = analizar_diferencias(texto_ref, texto_audio)
        fluidez = calcular_fluidez(transcripcion)

        # Limpiar archivos temporales
        os.remove(PDF_FILE)
        os.remove(AUDIO_FILE)

        return subir_estadisticas(app.state.supabase, paciente_id, diferencias, correctas, incorrectas, precision, fluidez)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))