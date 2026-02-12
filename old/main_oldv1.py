import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from groq import Groq

# --- 1. SETUP ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Analista de RNC Expert v2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. MODELOS ---
class Rnc(BaseModel):
    RNC: str
    ANO: str
    PRIORIDADE: str
    COD_PRODUTO: str
    CLASSIFICACAO: str
    DESCRICAO: str
    ORIGEM: str
    CLIENTE: str
    STATUS: str
    REGISTRO: str
    CONCLUSAO: Optional[str] = None
    DEPARTAMENTO_DESTINO: str

    @field_validator('*', mode='before')
    @classmethod
    def trim_string(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

class RequisicaoAnalista(BaseModel):
    dados_rnc: List[Rnc]

# --- 3. LOGICA PRINCIPAL COM FILTRO DE DADOS ---
@app.post("/analise-rnc")
async def analise_rnc(payload: RequisicaoAnalista):
    # FILTRO PREVENTIVO: Se não há dados, retornamos a resposta fixa sem chamar a IA
    if not payload.dados_rnc or len(payload.dados_rnc) == 0:
        logger.info("Sistema estável: 0 RNCs processadas.")
        return {
            "resumo_geral": "Não foram identificados registros de Não Conformidade (RNC) para os parâmetros selecionados. Os processos operam dentro da normalidade estatística.",
            "principais_causas": ["Operação estável"],
            "analise_de_risco": "baixo",
            "sugestao_plano_acao": "Manter protocolos de monitoramento preventivo.",
            "estatisticas": {
                "total_analisado": 0,
                "status_predominante": "N/A"
            }
        }

    try:
        texto_dados = json.dumps([item.model_dump() for item in payload.dados_rnc], indent=2)

        # Prompt com restrições severas de fidelidade aos dados
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "Você é um Auditor ISO 9001 rigoroso. Sua tarefa é descrever os fatos presentes nos dados. É estritamente proibido inventar códigos de produtos, nomes de clientes ou problemas que não constam na lista enviada."
                },
                {
                    "role": "user", 
                    "content": f"""
                    Analise fielmente estes dados de RNC:
                    {texto_dados}

                    DIRETRIZES:
                    - Descreva o cenário em um tom técnico e detalhado.
                    - Se um dado não estiver presente, não o presuma.
                    - Foque na relação entre os códigos de produtos e status reais.
                    - Não invente códigos de produtos, nomes de clientes ou problemas que não constam na lista enviada.
                    - Formate o texto para ficar agradavel de ler.
                    - Não utilize jargões técnicos desnecessários.
                    
                    Retorne em JSON:
                    {{
                      "resumo_geral": "Parecer técnico detalhado (mínimo 2 parágrafos)",
                      "principais_causas": ["lista de causas reais extraídas da descrição"],
                      "analise_de_risco": "baixo|medio|alto",
                      "sugestao_plano_acao": "Recomendações baseadas nas descrições",
                      "estatisticas": {{
                         "total_analisado": {len(payload.dados_rnc)},
                         "status_predominante": "string"
                      }}
                    }}
                    """
                }
            ],
            temperature=0.2, # Menor temperatura garante mais fidelidade aos dados
            response_format={"type": "json_object"}
        )

        return json.loads(completion.choices[0].message.content)

    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)