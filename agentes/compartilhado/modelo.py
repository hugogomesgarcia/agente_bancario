import os

from google import genai
from google.adk.models.google_llm import Gemini
from google.genai import types


NOME_MODELO = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
# Picos de demanda do Gemini retornam 503/429 temporários e, às vezes, deixam a
# conexão pendurada por minutos. O tempo limite por tentativa transforma esses
# casos em novas tentativas curtas; uma resposta livre revisada depende de até
# quatro chamadas ao modelo.
NOVAS_TENTATIVAS = types.HttpRetryOptions(attempts=4, initial_delay=1, max_delay=4)
OPCOES_HTTP = types.HttpOptions(timeout=20_000, retry_options=NOVAS_TENTATIVAS)


def criar_modelo() -> Gemini:
    return Gemini(model=NOME_MODELO)


def configuracao_geracao(**opcoes) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(http_options=OPCOES_HTTP, **opcoes)


def criar_cliente_genai(chave_api: str) -> genai.Client:
    return genai.Client(api_key=chave_api, http_options=OPCOES_HTTP)
