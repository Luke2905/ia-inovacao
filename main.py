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
                "content": """
                Você é um Auditor ISO 9001 Sênior (perfil analítico e independente). 
                Seu objetivo é produzir um parecer técnico robusto, rastreável e baseado estritamente nos dados de RNC fornecidos.

                Postura e regras:
                - Extraia fatos objetivos e padrões; não aceite explicações vagas.
                - Evite respostas genéricas: tudo deve estar ancorado em informações presentes nos dados.
                - Diferencie falha pontual vs. falha sistêmica (processo/controle).
                - Identifique tendências (recorrência por cliente, produto, etapa, motivo, setor, fornecedor, turno, operador, máquina, lote, data, ou qualquer marcador existente).
                - Se algum campo crítico estiver ausente/ambíguo, registre explicitamente a limitação e o impacto disso na análise (sem inventar dados).
                - Linguagem técnica e clara, sem jargões vazios.
                - Saída obrigatória: JSON puro, sem texto fora do JSON.
                """
                },
                {
                "role": "user",
                "content": f"""
                Analise detalhadamente estes dados de RNC (texto bruto abaixo). Atenha-se estritamente ao conteúdo fornecido:
                {texto_dados}

                INSTRUÇÕES OBRIGATÓRIAS PARA O PARECER:
                1) Em 'resumo_geral':
                - Escreva no mínimo 2 parágrafos.
                - Conecte fatos entre si (o que aconteceu, onde se repete, qual o padrão, qual o indício de falha de processo).
                - Cite explicitamente os códigos de produto e os clientes mencionados nos dados (nomes/códigos conforme aparecerem).
                - Aponte recorrências e padrões com base em evidências dos dados (ex.: “ocorreu X vezes”, “repetiu em datas/lotes/OPs diferentes”, “concentrado em um cliente/produto”).
                - Se os dados não permitirem afirmar recorrência, diga isso claramente e explique o que faltou.
                - Ignore RNC com status "Registrada".

                2) Em 'principais_causas':
                - Liste somente causas que apareçam nos dados (causa informada, descrição de falha, etapa do processo, evidência repetida).
                - Escreva causas como frases objetivas e auditáveis (ex.: “Falta de inspeção final registrada”, “Parâmetro de processo fora do padrão”, “Matéria-prima fora de especificação”).
                - Não invente causa raiz; se a causa estiver indefinida, registre como “Causa não determinada nos registros” e explique no resumo.

                3) Em 'analise_de_risco':
                - Classifique como "baixo", "medio" ou "alto" com base nos próprios dados.
                - Justifique no texto (dentro do campo) considerando: recorrência, impacto no cliente, possibilidade de escape, severidade do defeito, status da RNC (aberta/fechada), e repetição por produto/cliente/lote/OP, quando houver.

                4) Em 'sugestao_plano_acao':
                - Proponha passos práticos e verificáveis, derivados das falhas relatadas e das lacunas de controle percebidas nos dados.
                - Não cite requisitos da norma por número; descreva ações de forma operacional (ex.: “revisar ponto de controle X”, “reforçar critério de aceite”, “criar verificação de registro”, “bloquear lote até evidência”, etc.).
                - Se não houver dados suficientes para um plano específico, proponha ações de coleta de evidência (ex.: “levantar histórico por produto/cliente”, “estratificar por causa/status”).

                5) Em 'estatisticas':
                - total_analisado: use exatamente {len(payload.dados_rnc)}.
                - status_predominante: indique o status mais frequente encontrado nos dados (ex.: "ABERTA", "FECHADA", "EM_ANDAMENTO"). Se não existir status nos dados, retorne "NAO_INFORMADO".

                IMPORTANTE:
                - Não use bullet points fora do JSON.
                - Não retorne markdown.
                - Não inclua comentários.
                - Não inclua campos extras.

                Retorne EXCLUSIVAMENTE em JSON, exatamente com este esquema e tipos:
                {{
                "resumo_geral": "string",
                "principais_causas": ["string", "string"],
                "analise_de_risco": "baixo|medio|alto",
                "sugestao_plano_acao": "string",
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