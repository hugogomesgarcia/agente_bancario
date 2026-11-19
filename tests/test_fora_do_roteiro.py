"""Conversas completas com mensagens fora do roteiro, sem chamadas reais."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from google.adk.models import LlmResponse
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

from agentes.agent import root_agent
from agentes.cambio.agent import agente_cambio
from agentes.cambio.tools import cambio
from agentes.credito.agent import agente_credito
from agentes.credito.tools import credito
from agentes.entrevista_credito.agent import agente_entrevista_credito
from agentes.entrevista_credito.tools import entrevista_credito
from agentes.triagem.tools import consultar_client


AVALIADOR = "agentes.compartilhado.guardrail._avaliar_resposta_com_modelo"


def _textos_do_cliente(llm_request) -> list[str]:
    # O ADK repassa turnos de outros agentes como "For context:" no papel user.
    textos = []
    for conteudo in llm_request.contents:
        texto = "".join(parte.text or "" for parte in conteudo.parts or [])
        if conteudo.role == "user" and texto and not texto.startswith("For context:"):
            textos.append(texto.lower())
    return textos


def _chamada(nome, **argumentos):
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name=nome, args=argumentos))],
        )
    )


def _texto(texto):
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text=texto)])
    )


class ModeloRoteirizado(BaseLlm):
    """Simula as decisões semânticas de cada agente a partir das suas ferramentas."""

    requisicoes: list = Field(default_factory=list)

    async def generate_content_async(self, llm_request, stream=False):
        self.requisicoes.append(llm_request)
        ferramentas = set(llm_request.tools_dict)
        textos = _textos_do_cliente(llm_request) or [""]
        mensagem = textos[-1]
        if "processar_solicitacao_cambio" in ferramentas:
            yield _chamada(
                "processar_solicitacao_cambio",
                status="resolvido",
                ativo_base="BRL",
                ativo_destino="GBP",
                destino_explicito=True,
                evidencia_base="meu limite",
                evidencia_destino="em libras",
                evidencia_quantidade="meu limite",
                usar_limite_credito=True,
            )
        elif "solicitar_aumento_limite" in ferramentas:
            if "libras" in mensagem:
                yield _chamada("transfer_to_agent", agent_name="triagem")
            elif any("score" in texto for texto in textos):
                yield _chamada("consultar_score_credito")
            else:
                yield _chamada("solicitar_aumento_limite", novo_limite_solicitado="8000")
        elif not ferramentas:
            yield _texto("Tudo ótimo, obrigado por perguntar!")
        elif "libras" in mensagem:
            yield _chamada("transfer_to_agent", agent_name="cambio")
        elif "obrigad" in mensagem:
            yield _texto("De nada! Fico à disposição.")
        else:
            yield _chamada("transfer_to_agent", agent_name="credito")


class ConversaForaDoRoteiroTest(unittest.TestCase):
    def setUp(self):
        self.diretorio = tempfile.TemporaryDirectory()
        raiz = Path(self.diretorio.name)
        clientes = raiz / "clientes.csv"
        clientes.write_text(
            'CPF,"Data de Nascimento",Score,"Limite de Crédito"\n'
            '"710.483.880-50","29/07/1997",720,5000.00\n',
            encoding="utf-8",
        )
        faixas = raiz / "score_limite.csv"
        faixas.write_text(
            "score_minimo,score_maximo,limite_maximo\n0,1000,10000.00\n",
            encoding="utf-8",
        )
        solicitacoes = raiz / "solicitacoes.csv"
        solicitacoes.write_text(
            ",".join(credito.COLUNAS_SOLICITACAO) + "\n", encoding="utf-8"
        )
        self.patches = [
            patch.object(credito, "CLIENTES_CSV", clientes),
            patch.object(credito, "SCORE_LIMITE_CSV", faixas),
            patch.object(credito, "SOLICITACOES_CSV", solicitacoes),
            patch.object(consultar_client, "CSV_PATH", clientes),
            patch.object(entrevista_credito, "CLIENTES_CSV", clientes),
        ]
        self.clientes = clientes
        self.solicitacoes = solicitacoes
        for item in self.patches:
            item.start()

        self.modelo = ModeloRoteirizado(model="modelo-roteirizado")
        raiz_teste = root_agent.clone(
            update={
                "model": self.modelo,
                "sub_agents": [
                    agente_credito.clone(update={"model": self.modelo}),
                    agente_entrevista_credito.clone(update={"model": self.modelo}),
                    agente_cambio.clone(update={"model": self.modelo}),
                ],
            }
        )
        self.sessoes = InMemorySessionService()
        self.runner = Runner(
            app_name="fora_do_roteiro",
            agent=raiz_teste,
            session_service=self.sessoes,
            auto_create_session=True,
        )

    def tearDown(self):
        asyncio.run(self.runner.close())
        for item in reversed(self.patches):
            item.stop()
        self.diretorio.cleanup()

    def _enviar(self, texto):
        eventos = list(
            self.runner.run(
                user_id="cliente",
                session_id="sessao",
                new_message=types.Content(
                    role="user", parts=[types.Part.from_text(text=texto)]
                ),
            )
        )
        return eventos, [
            "".join(parte.text or "" for parte in evento.content.parts or [])
            for evento in eventos
            if evento.is_final_response() and evento.content
        ]

    def _autenticar(self):
        self._enviar("Iniciar atendimento.")
        self._enviar("710.483.880-50")
        _, respostas = self._enviar("29/07/1997")
        self.assertIn("Autenticação realizada com sucesso! Como posso ajudar você hoje?", respostas)

    def test_cumprimento_antes_do_cpf_recebe_resposta_e_pedido_do_cpf(self):
        _, respostas = self._enviar("Iniciar atendimento.")
        self.assertIn("informe o seu CPF", respostas[-1])

        with patch(AVALIADOR, new=AsyncMock(return_value=(True, "ok"))):
            _, respostas = self._enviar("oi, tudo bem?")

        self.assertEqual(
            respostas,
            ["Tudo ótimo, obrigado por perguntar! Para continuarmos, informe seu CPF."],
        )
        requisicao = self.modelo.requisicoes[-1]
        self.assertEqual(requisicao.tools_dict, {})
        self.assertFalse(requisicao.config.tools)

        _, respostas = self._enviar("710.483.880-50")
        self.assertIn("CPF localizado. Informe sua data de nascimento.", respostas)

    def test_pedido_feito_com_o_cpf_e_atendido_apos_autenticacao(self):
        self._enviar("Iniciar atendimento.")
        self._enviar("Boa tarde! Meu CPF é 710.483.880-50, quero saber meu score.")

        eventos, respostas = self._enviar("29/07/1997")

        self.assertEqual(
            respostas[-1],
            "Seu score de crédito atual é 720. Posso consultar seu limite ou "
            "analisar uma solicitação de aumento.",
        )
        self.assertEqual(eventos[-1].author, "credito")
        requisicao_credito = next(
            requisicao
            for requisicao in self.modelo.requisicoes
            if "consultar_score_credito" in requisicao.tools_dict
        )
        self.assertIn(
            "quero saber meu score", " ".join(_textos_do_cliente(requisicao_credito))
        )

    def test_limite_aprovado_pode_ser_convertido_em_libras(self):
        self._autenticar()
        _, respostas = self._enviar("Quero aumentar meu limite para 8000")
        self.assertTrue(
            any("solicitação foi aprovada" in texto for texto in respostas), respostas
        )

        cotacao = {
            "sucesso": True,
            "ativo_base": "BRL",
            "ativo_destino": "GBP",
            "nome_par": "BRL/GBP",
            "compra": "0.14",
            "venda": "0.145",
            "consultado_em": "19/11/2026 às 12:00 UTC",
            "fonte": "AwesomeAPI",
            "derivada_par_inverso": True,
            "derivada_par_cruzado": False,
        }
        with patch.object(cambio, "_consultar_cotacao", return_value=cotacao) as consultar:
            eventos, respostas = self._enviar("Quanto vale meu limite em libras?")

        consultar.assert_called_once_with("BRL", "GBP")
        autores = [evento.author for evento in eventos]
        self.assertLess(autores.index("credito"), autores.index("triagem"))
        self.assertLess(autores.index("triagem"), autores.index("cambio"))
        resposta = respostas[-1]
        self.assertIn("seu limite de crédito atual de 8.000 BRL", resposta)
        self.assertIn("GBP 1.120 pela cotação de compra", resposta)
        self.assertIn("GBP 1.160 pela cotação de venda", resposta)
        self.assertNotIn("Seu limite atual é", resposta)

    def test_agradecimento_apos_autenticacao_recebe_resposta_natural(self):
        self._autenticar()

        with patch(AVALIADOR, new=AsyncMock(return_value=(True, "ok"))):
            eventos, respostas = self._enviar("obrigado!")

        self.assertEqual(respostas, ["De nada! Fico à disposição."])
        self.assertEqual(eventos[-1].author, "triagem")

    def test_valor_apos_pausa_da_entrevista_responde_a_entrevista(self):
        asyncio.run(
            self.sessoes.create_session(
                app_name="fora_do_roteiro",
                user_id="cliente",
                session_id="sessao",
                state={
                    "cliente_autenticado": True,
                    "cpf_cliente": "71048388050",
                    credito.ETAPA_CREDITO: credito.AGUARDANDO_PROXIMA_ACAO,
                    "etapa_entrevista_credito": "aguardando_renda_mensal",
                    entrevista_credito.DADOS_ENTREVISTA: {},
                },
            )
        )

        eventos, respostas = self._enviar("8000")

        self.assertEqual(
            respostas[-1],
            "Qual é o seu tipo de emprego: formal, autônomo ou desempregado?",
        )
        self.assertIn("entrevista_credito", [evento.author for evento in eventos])
        self.assertEqual(self.modelo.requisicoes, [])
        self.assertEqual(
            self.solicitacoes.read_text(encoding="utf-8").strip(),
            ",".join(credito.COLUNAS_SOLICITACAO),
        )
        self.assertIn("5000.00", self.clientes.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
