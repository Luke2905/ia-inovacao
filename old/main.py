import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import ollama # <--- Motor Local

# 1. CONFIGURAÇÕES
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Endereço do seu servidor Ollama na rede
OLLAMA_HOST = "http://10.0.3.2:11434"

app = FastAPI(title="Analista de RNC Local v3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. MODELO DE DADOS
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

# 3. ENDPOINT USANDO QWEN-2.5:7B LOCAL
@app.post("/analise-rnc")
async def analise_rnc(payload: RequisicaoAnalista):
    logger.info(f"Analisando {len(payload.dados_rnc)} RNCs no servidor local 10.0.3.2")

    # Cálculo dinâmico simples no seu endpoint
    qtd_rnc = len(payload.dados_rnc)
    # Base de 1024 tokens + 100 tokens extras por RNC (ajuste conforme necessário)
    contexto_dinamico = max(2048, qtd_rnc * 150) 

    # Trava Anti-Alucinação: Se não houver dados, retorna resposta padrão
    if not payload.dados_rnc:
        return {
            "resumo_geral": "Nenhum registro encontrado para análise.",
            "principais_causas": [],
            "analise_de_risco": "baixo",
            "sugestao_plano_acao": "Manter monitoramento.",
            "estatisticas": {"total_analisado": 0, "status_predominante": "N/A"}
        }

    try:
        # Prepara os dados
        texto_dados = json.dumps([item.model_dump() for item in payload.dados_rnc], indent=2)

        # Configura o cliente para o IP correto
        client = ollama.Client(host=OLLAMA_HOST)

        # Chamada ao modelo local
        response = client.chat(
            model='llama3.2:latest',
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
            format='json', # Força o modo JSON do Ollama
            options={
                'temperature': 0.1,
                'num_ctx': contexto_dinamico,  # Aumentando o contexto para ele ler todas as RNCs com calma
                'num_thread':8, #força o uso de mais núcleos da CPU da VM
                'num_predict': 500, #limta o tamanho da resposta
                'top_p': 0.1 #Faz a ia usar palavras mais coerentes
            }
        )

        # O Ollama retorna a resposta dentro de ['message']['content']
        return json.loads(response['message']['content'])

    except Exception as e:
        logger.error(f"Erro no Ollama local: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no servidor local: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)