import os
import re
import fitz
import whisper
import difflib
import uuid
import json
import datetime
import warnings
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from supabase import create_client, Client
from typing import Optional

# Configuración básica
app = FastAPI(title="Analizador de Lectura",
              description="API para análisis de lectura usando Whisper",
              version="1.0")

warnings.filterwarnings("ignore", category=UserWarning)

# Configuración para plan gratuito
MODELO_WHISPER = "base"  # Usamos el modelo base para el plan free
MAX_DURACION_AUDIO = 120  # 2 minutos en segundos
BUCKET_PDF = "pruebas"
BUCKET_AUDIO = "audios"
PDF_TEMP = "temp_ref.pdf"
AUDIO_TEMP = "temp_audio.mp3"  # Usamos MP3 para ahorrar espacio

# Modelos de datos
class AnalisisRequest(BaseModel):
    paciente_id: str
    archivo_pdf: str
    archivo_audio: str

class HealthCheck(BaseModel):
    status: str
    modelo: str
    timestamp: str

# Conexión a Supabase
@app.on_event("startup")
async def startup_event():
    app.state.supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )
    # Cargamos el modelo al iniciar para ahorrar tiempo
    app.state.whisper_model = whisper.load_model(MODELO_WHISPER)

# Endpoints
@app.post("/programar-analisis", summary="Programa un nuevo análisis")
async def programar_analisis(
    request: AnalisisRequest,
    background_tasks: BackgroundTasks
):
    """Programa un análisis en segundo plano"""
    try:
        # Verificamos si el servicio está ocupado (limitación del plan free)
        if hasattr(app.state, 'analisis_en_curso') and app.state.analisis_en_curso:
            raise HTTPException(status_code=429, detail="El servicio está ocupado. Solo puede procesar un audio a la vez en el plan gratuito")

        app.state.analisis_en_curso = True
        
        background_tasks.add_task(
            realizar_analisis_completo,
            request.paciente_id,
            request.archivo_pdf,
            request.archivo_audio
        )
        
        return {
            "status": "success",
            "message": "Análisis programado",
            "job_id": str(uuid.uuid4()),
            "modelo": MODELO_WHISPER,
            "limitaciones": {
                "max_duracion": MAX_DURACION_AUDIO,
                "modelo": MODELO_WHISPER
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/estado-analisis/{job_id}", summary="Verifica el estado de un análisis")
async def verificar_estado(job_id: str):
    """Endpoint simplificado para el plan gratuito"""
    return {
        "status": "completed" if not hasattr(app.state, 'analisis_en_curso') or not app.state.analisis_en_curso else "processing",
        "modelo": MODELO_WHISPER
    }

@app.get("/ultimos-resultados/{paciente_id}", summary="Obtiene los últimos resultados")
async def obtener_ultimos_resultados(paciente_id: str):
    try:
        supabase = app.state.supabase
        
        # Obtenemos solo los datos esenciales para ahorrar ancho de banda
        precision = supabase.table("precision_pronunciacion") \
                   .select("porcentaje_precision,palabras_correctas,palabras_incorrectas,fecha") \
                   .eq("paciente_id", paciente_id) \
                   .order("fecha", desc=True).limit(1).execute()
        
        fluidez = supabase.table("fluidez_lectora") \
                 .select("palabras_por_minuto,numero_pausas,fecha") \
                 .eq("paciente_id", paciente_id) \
                 .order("fecha", desc=True).limit(1).execute()
        
        errores = supabase.table("errores_recurrentes") \
                 .select("palabra_original,palabra_dicha,tipo") \
                 .eq("paciente_id", paciente_id) \
                 .order("fecha", desc=True).limit(5).execute()

        return {
            "precision": precision.data[0] if precision.data else None,
            "fluidez": fluidez.data[0] if fluidez.data else None,
            "errores": errores.data if errores.data else [],
            "modelo": MODELO_WHISPER
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {
        "status": "OK",
        "modelo": MODELO_WHISPER,
        "timestamp": datetime.datetime.now().isoformat()
    }

# Funciones de análisis optimizadas
def descargar_archivo(supabase: Client, bucket: str, archivo_remoto: str, archivo_local: str) -> bool:
    try:
        with open(archivo_local, "wb") as f:
            response = supabase.storage.from_(bucket).download(archivo_remoto)
            f.write(response)
        return True
    except Exception as e:
        print(f"Error al descargar archivo: {str(e)[:200]}")  # Limitar log
        return False

def extraer_texto_pdf(pdf_path: str) -> str:
    try:
        texto = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                texto.append(page.get_text("text"))
        return " ".join(texto).strip()[:5000]  # Limitar texto para ahorrar memoria
    except Exception as e:
        print(f"Error al extraer texto PDF: {str(e)[:200]}")
        return ""

def transcribir_audio(audio_path: str):
    try:
        # Usamos el modelo ya cargado en memoria
        result = app.state.whisper_model.transcribe(
            audio_path,
            fp16=False,  # Mejor compatibilidad con CPU
            language='es'  # Especificamos español para mejor precisión
        )
        return result
    except Exception as e:
        print(f"Error en transcripción: {str(e)[:200]}")
        return None

def normalizar_texto(texto: str) -> str:
    return re.sub(r"[^\w\sáéíóúñ]", "", texto.lower()).strip()

def analizar_diferencias(texto_ref: str, texto_audio: str):
    try:
        ref_words = texto_ref.split()[:500]  # Limitar cantidad de palabras
        audio_words = texto_audio.split()[:500]
        
        matcher = difflib.SequenceMatcher(None, ref_words, audio_words)
        correctas = sum(i2-i1 for op,i1,i2,j1,j2 in matcher.get_opcodes() if op == "equal")
        total = len(ref_words)
        
        diferencias = []
        for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
            if opcode != "equal":
                diferencias.append({
                    "tipo": opcode,
                    "palabra_original": " ".join(ref_words[i1:i2]),
                    "palabra_dicha": " ".join(audio_words[j1:j2])
                })
        
        precision = round((correctas / total) * 100, 2) if total > 0 else 0
        return diferencias[:10], correctas, total - correctas, precision  # Limitar errores
    
    except Exception as e:
        print(f"Error en análisis diferencias: {str(e)[:200]}")
        return [], 0, 0, 0

def calcular_fluidez(transcripcion: dict) -> dict:
    try:
        palabras = transcripcion['text'].split()
        n_palabras = len(palabras)
        duracion = transcripcion['segments'][-1]['end'] if transcripcion['segments'] else 1
        
        palabras_por_minuto = min(round(n_palabras / (duracion / 60), 2), 300)  # Limitar máximo
        
        pausas = 0
        for i in range(1, len(transcripcion['segments'])):
            if transcripcion['segments'][i]['start'] - transcripcion['segments'][i-1]['end'] > 1.5:
                pausas += 1
        
        return {
            "palabras_por_minuto": palabras_por_minuto,
            "numero_pausas": min(pausas, 20),  # Limitar máximo
            "duracion_total_segundos": round(min(duracion, MAX_DURACION_AUDIO), 2)
        }
    except Exception as e:
        print(f"Error cálculo fluidez: {str(e)[:200]}")
        return {
            "palabras_por_minuto": 0,
            "numero_pausas": 0,
            "duracion_total_segundos": 0
        }

async def realizar_analisis_completo(paciente_id: str, archivo_pdf: str, archivo_audio: str):
    try:
        supabase = app.state.supabase
        
        # Descargamos archivos con timeout
        if not descargar_archivo(supabase, BUCKET_PDF, archivo_pdf, PDF_TEMP):
            raise Exception("Error descargando PDF")
        
        if not descargar_archivo(supabase, BUCKET_AUDIO, archivo_audio, AUDIO_TEMP):
            raise Exception("Error descargando audio")
        
        # Verificamos duración del audio
        import librosa
        duracion = librosa.get_duration(filename=AUDIO_TEMP)
        if duracion > MAX_DURACION_AUDIO:
            raise Exception(f"Audio demasiado largo ({duracion}s > {MAX_DURACION_AUDIO}s)")
        
        # Procesamiento
        texto_ref = normalizar_texto(extraer_texto_pdf(PDF_TEMP))
        transcripcion = transcribir_audio(AUDIO_TEMP)
        if not transcripcion:
            raise Exception("Error en transcripción")
        
        texto_audio = normalizar_texto(transcripcion['text'])
        diferencias, correctas, incorrectas, precision = analizar_diferencias(texto_ref, texto_audio)
        fluidez = calcular_fluidez(transcripcion)
        
        # Guardamos resultados
        await guardar_resultados(supabase, paciente_id, diferencias, correctas, incorrectas, precision, fluidez)
        
    except Exception as e:
        print(f"Error en análisis: {str(e)[:500]}")
    finally:
        # Limpieza
        for archivo in [PDF_TEMP, AUDIO_TEMP]:
            if os.path.exists(archivo):
                try:
                    os.remove(archivo)
                except:
                    pass
        app.state.analisis_en_curso = False

async def guardar_resultados(supabase: Client, paciente_id: str, diferencias: list, correctas: int, incorrectas: int, precision: float, fluidez: dict):
    now = datetime.datetime.now().isoformat()
    
    try:
        # Datos mínimos necesarios
        datos_precision = {
            "paciente_id": paciente_id,
            "fecha": now,
            "palabras_correctas": correctas,
            "palabras_incorrectas": incorrectas,
            "porcentaje_precision": precision
        }
        
        datos_fluidez = {
            "paciente_id": paciente_id,
            "fecha": now,
            "palabras_por_minuto": fluidez["palabras_por_minuto"],
            "numero_pausas": fluidez["numero_pausas"]
        }
        
        # Insertamos en lotes pequeños
        supabase.table("precision_pronunciacion").insert(datos_precision).execute()
        supabase.table("fluidez_lectora").insert(datos_fluidez).execute()
        
        # Solo guardamos los primeros 5 errores
        for error in diferencias[:5]:
            supabase.table("errores_recurrentes").insert({
                "paciente_id": paciente_id,
                "fecha": now,
                "tipo_error": error["tipo"],
                "palabra_original": error["palabra_original"],
                "palabra_dicha": error["palabra_dicha"]
            }).execute()
            
    except Exception as e:
        print(f"Error guardando resultados: {str(e)[:500]}")