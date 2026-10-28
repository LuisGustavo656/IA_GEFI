from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any, Dict, Optional
import pandas as pd
import numpy as np
import skfuzzy as fuzzy
import math
from openai import OpenAI
import os
import json
import re  

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PayloadSchema(BaseModel):
    jovens: Optional[List[Dict[str, Any]]] = None
    membros: Optional[List[Dict[str, Any]]] = None  
    limite_por_grupo: Optional[int] = 8

def processar_grupos_ia(jovens_lista: list, limite_por_grupo: int):
    df = pd.DataFrame(jovens_lista)
    
    if df.empty:
        return {"erro": "Nenhum membro encontrado"}

    colunas_mapeamento = {
        'skill_fisico': 'fisico', 'skillFisico': 'fisico', 'fisico': 'fisico',
        'skill_tecnico': 'tecnico', 'skillTecnico': 'tecnico', 'tecnico': 'tecnico',
        'skill_experiencia': 'experiencia', 'skillExperiencia': 'experiencia', 'experiencia': 'experiencia',
        'ramo': 'ramo', 'branch_name': 'ramo', 'section': 'ramo'
    }
    
    df = df.rename(columns=colunas_mapeamento)

    skills = ['fisico', 'tecnico', 'experiencia']

    for s in skills:
        if s not in df.columns:
            df[s] = 1
        else:
            df[s] = pd.to_numeric(df[s], errors='coerce').fillna(1).astype(int)

    if 'ramo' not in df.columns:
        df['ramo'] = 'Geral'
    else:
        df['ramo'] = df['ramo'].fillna('Geral').astype(str)

    # Executa o Fuzzy C-Means
    data_fcm = df[skills].values.T
    num_clusters = min(3, len(df))

    try:
        cntr, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(
            data_fcm,
            c=num_clusters,
            m=2,
            error=0.005,
            maxiter=1000,
            init=None
        )
        probabilidades = u.T
        mapa_clusters = {i: skills[np.argmax(centro)] for i, centro in enumerate(cntr)}
        
        esp_1 = []
        esp_2 = []
        for prob in probabilidades:
            indices_ordenados = np.argsort(prob)[::-1]
            esp_1.append(mapa_clusters[indices_ordenados[0]])
            esp_2.append(mapa_clusters[indices_ordenados[1]])
            
        df['especialidade'] = esp_1
        df['segunda_especialidade'] = esp_2
        fpc_valor = float(fpc)
    except Exception as e:
        print(f"Aviso no Fuzzy: {e}. Usando distribuição padrão.")
        df['especialidade'] = 'tecnico'
        df['segunda_especialidade'] = 'fisico'
        fpc_valor = 1.0

    df['pontuacao_total'] = df['fisico'] + df['tecnico'] + df['experiencia']

    resultado_final = []
    ramos = df['ramo'].dropna().unique()

    for ramo in ramos:
        ramo_df = df[df['ramo'] == ramo].copy().sort_values('pontuacao_total', ascending=False).reset_index(drop=True)
        quantidade_grupos = math.ceil(len(ramo_df) / limite_por_grupo)
        grupos = [[] for _ in range(quantidade_grupos)]

        direcao = 1
        grupo_atual = 0

        for _, row in ramo_df.iterrows():
            dict_jovem = row.replace({np.nan: None}).to_dict()
            grupos[grupo_atual].append(dict_jovem)

            if direcao == 1:
                if grupo_atual == quantidade_grupos - 1:
                    direcao = -1
                else:
                    grupo_atual += 1
            else:
                if grupo_atual == 0:
                    direcao = 1
                else:
                    grupo_atual -= 1

        for i, grupo in enumerate(grupos):
            resultado_final.append({
                "ramo": ramo,
                "grupo": i,
                "membros": grupo
            })

    return {
        "fpc": fpc_valor,
        "grupos": resultado_final
    }

@app.post("/gerar-grupos")
def gerar_grupos(payload: PayloadSchema):


    lista_jovens = []

    if hasattr(payload, "jovens") and payload.jovens:
        lista_jovens = payload.jovens

    elif hasattr(payload, "membros") and payload.membros:
        lista_jovens = payload.membros


    if not lista_jovens:
        return {
            "erro": "Payload inválido",
            "detalhe": "Envie os dados em 'jovens' ou 'membros'"
        }


    resultado = processar_grupos_ia(
        lista_jovens,
        payload.limite_por_grupo
    )


    print("\n===================================")
    print("GRUPOS GERADOS")
    print("===================================\n")

    print(
        json.dumps(
            resultado,
            indent=4,
            ensure_ascii=False
        )
    )

    return resultado


@app.post("/atividades")
def gerar_atividades(payload: PayloadSchema):


    lista_jovens = []

    if hasattr(payload, "jovens") and payload.jovens:
        lista_jovens = payload.jovens

    elif hasattr(payload, "membros") and payload.membros:
        lista_jovens = payload.membros

    if not lista_jovens:
        return {
            "erro": "Payload inválido",
            "detalhe": "Envie os dados em 'jovens' ou 'membros'"
        }

    grupos = processar_grupos_ia(
        lista_jovens,
        payload.limite_por_grupo
    )

    # PROMPT IA

    prompt = f"""
Você é especialista em atividades escoteiras.

Analise os grupos abaixo.

As atividades devem ser:
- específicas
- práticas
- aplicáveis no mundo real
- adequadas ao perfil do grupo

Para cada atividade informe:
- nome
- dificuldade
- objetivo
- duração
- materiais
- como_executar
- habilidades_trabalhadas

Responda SOMENTE JSON válido.

Formato:

{{
  "grupos": [
    {{
      "grupo": 0,
      "perfil_do_grupo": "...",
      "atividades": [
        {{
          "nome": "...",
          "dificuldade": "...",
          "objetivo": "...",
          "duracao": "...",
          "materiais": ["..."],
          "como_executar": [
            "...",
            "...",
            "..."
          ],
          "habilidades_trabalhadas": [
            "...",
            "..."
          ]
        }}
      ]
    }}
  ]
}}

Dados:
{json.dumps(grupos, ensure_ascii=False)}
"""

    try:
        resposta = client.chat.completions.create(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        texto = resposta.choices[0].message.content.strip()

        texto_limpo = re.sub(r"```[a-zA-Z]*", "", texto).strip()

        return json.loads(texto_limpo)

    except json.JSONDecodeError as json_err:
        return {
            "erro": "A IA retornou um formato de texto inválido",
            "detalhe": str(json_err),
            "resposta_bruta_ia": texto if 'texto' in locals() else "Sem resposta"
        }
    except Exception as e:
        return {
            "erro": "Falha ao gerar atividades",
            "detalhe": str(e)
        }
from rf08_sugestoes import router as rf08_router
app.include_router(rf08_router)
