import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from google.adk.models import LlmResponse
from google.genai import types

from agentes.compartilhado.encerramento import AGUARDANDO_CONFIRMACAO_ENCERRAMENTO
from agentes.compartilhado.estado import OPCOES_RESPOSTA
from agentes.compartilhado.guardrail import (
    LIMITE_HISTORICO,
    PerfilRevisao,
    PerguntaPendente,
    criar_revisao_de_resposta,
    criar_tratamento_de_erro,
    extrair_fatos,
)


AVALIADOR = "agentes.compartilhado.guardrail._avaliar_resposta_com_modelo"
REESCRITOR = "agentes.compartilhado.guardrail._reescrever_resposta_com_modelo"


def _conteudo(texto, *, pensamento=False):
    return types.Content(
        role="model", parts=[types.Part(text=texto, thought=pensamento or None)]
    )


def _evento(autor, texto, *, pensamento=False):
    return SimpleNamespace(author=autor, content=_conteudo(texto, pensamento=pensamento))


class ContextoGuardrailTeste:
    def __init__(self, mensagem="mensagem", eventos=()):
        self.state = {}
        self.user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=mensagem)]
        )
        self.session = SimpleNamespace(events=list(eventos))


PERFIL = PerfilRevisao(
    agente="teste",
    escopo=lambda estado: "Escopo de teste.",
    resposta_segura=lambda estado: "Resposta segura.",
    fatos_verificados=lambda estado: extrair_fatos(
        estado, "resultado", ("limite_atual", "compra")
    ),
    pergunta_pendente=lambda estado: (
        PerguntaPendente("Qual é a sua renda mensal?", ("A", "B"), ("renda",))
        if estado.get("pendente")
        else None
    ),
)
revisar = criar_revisao_de_resposta(PERFIL)


class GuardrailCompartilhadoTest(unittest.TestCase):
    def _revisar(self, contexto, texto, avaliador=None, reescritor=None):
        avaliador = avaliador or AsyncMock(return_value=(True, "ok"))
        reescritor = reescritor or AsyncMock(return_value="Reescrita.")
        with patch(AVALIADOR, avaliador), patch(REESCRITOR, reescritor):
            resposta = asyncio.run(
                revisar(contexto, LlmResponse(content=_conteudo(texto)))
            )
        return resposta, avaliador, reescritor

    def test_numeros_verificados_aceitam_formatos_brasileiros(self):
        contexto = ContextoGuardrailTeste("quanto dá 250 reais?")
        contexto.state["resultado"] = {"limite_atual": "10000.00", "compra": "0.1201"}

        for texto in (
            "Seu limite é R$ 10.000,00.",
            "A compra estava em 0,1201.",
            "Você citou 250 reais.",
        ):
            with self.subTest(texto=texto):
                resposta, avaliador, _ = self._revisar(contexto, texto)
                self.assertIsNone(resposta)
                avaliador.assert_awaited_once()

    def test_numero_inventado_vai_para_reescrita_sem_consultar_revisor(self):
        contexto = ContextoGuardrailTeste("oi")
        contexto.state["resultado"] = {"limite_atual": "10000.00"}
        avaliador = AsyncMock(return_value=(True, "ok"))

        resposta, _, reescritor = self._revisar(
            contexto, "Seu limite é de 10 mil, com 3% de juros.", avaliador
        )

        motivo = reescritor.await_args.args[2]
        self.assertIn("10", motivo)
        self.assertIn("3", motivo)
        self.assertEqual(resposta.content.parts[0].text, "Reescrita.")
        avaliador.assert_awaited_once()

    def test_reescrita_com_numero_inventado_usa_resposta_segura(self):
        contexto = ContextoGuardrailTeste("oi")
        avaliador = AsyncMock(return_value=(True, "ok"))

        resposta, _, _ = self._revisar(
            contexto,
            "Seu score é 900.",
            avaliador,
            AsyncMock(return_value="Seu score é 950."),
        )

        self.assertEqual(resposta.content.parts[0].text, "Resposta segura.")
        avaliador.assert_not_awaited()

    def test_historico_recente_omite_pensamentos_e_respeita_limite(self):
        eventos = [
            _evento("user" if indice % 2 else "credito", f"mensagem {indice}")
            for indice in range(LIMITE_HISTORICO + 4)
        ]
        eventos.append(_evento("credito", "raciocínio interno", pensamento=True))
        contexto = ContextoGuardrailTeste("obrigado", eventos)

        _, avaliador, _ = self._revisar(contexto, "De nada!")

        historico = avaliador.await_args.args[0]["historico_recente"]
        self.assertEqual(len(historico), LIMITE_HISTORICO)
        self.assertEqual(historico[-1], {"autor": "cliente", "texto": "mensagem 13"})
        self.assertEqual(historico[-2], {"autor": "banco", "texto": "mensagem 12"})
        self.assertNotIn("raciocínio interno", str(historico))

    def test_pergunta_pendente_e_acrescentada_com_opcoes(self):
        contexto = ContextoGuardrailTeste("por quê?")
        contexto.state["pendente"] = True

        resposta, avaliador, _ = self._revisar(contexto, "É para recalcular o score.")

        self.assertEqual(
            resposta.content.parts[0].text,
            "É para recalcular o score. Qual é a sua renda mensal?",
        )
        self.assertEqual(contexto.state[OPCOES_RESPOSTA], ["A", "B"])
        self.assertEqual(
            avaliador.await_args.args[0]["pergunta_pendente_adicionada_pelo_sistema"],
            "Qual é a sua renda mensal?",
        )

    def test_pergunta_feita_pelo_modelo_na_ultima_frase_nao_e_repetida(self):
        contexto = ContextoGuardrailTeste("por quê?")
        contexto.state["pendente"] = True

        for texto, esperado in (
            (
                "Ela ajuda a recalcular o score. Por favor, informe sua renda.",
                "Ela ajuda a recalcular o score. Por favor, informe sua renda.",
            ),
            (
                "Ela ajuda no cálculo. Qual é a sua renda mensal?",
                "Ela ajuda no cálculo. Qual é a sua renda mensal?",
            ),
            (
                "A renda ajuda a recalcular o score.",
                "A renda ajuda a recalcular o score. Qual é a sua renda mensal?",
            ),
        ):
            with self.subTest(texto=texto):
                resposta, _, _ = self._revisar(contexto, texto)
                self.assertEqual(resposta.content.parts[0].text, esperado)
                self.assertEqual(contexto.state[OPCOES_RESPOSTA], ["A", "B"])

    def test_confirmacao_de_encerramento_suspende_pergunta_pendente(self):
        contexto = ContextoGuardrailTeste("quero sair")
        contexto.state.update(
            {"pendente": True, AGUARDANDO_CONFIRMACAO_ENCERRAMENTO: True}
        )

        resposta, _, _ = self._revisar(contexto, "Você deseja encerrar?")

        self.assertIsNone(resposta)

    def test_chamadas_de_ferramenta_dispensam_revisao_e_descartam_texto(self):
        contexto = ContextoGuardrailTeste()
        chamada = types.Part(
            function_call=types.FunctionCall(name="ferramenta", args={})
        )
        avaliador = AsyncMock()

        with patch(AVALIADOR, avaliador):
            somente = asyncio.run(
                revisar(
                    contexto,
                    LlmResponse(content=types.Content(role="model", parts=[chamada])),
                )
            )
            mista = asyncio.run(
                revisar(
                    contexto,
                    LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text="Texto."), chamada],
                        )
                    ),
                )
            )

        self.assertIsNone(somente)
        self.assertEqual(mista.content.parts, [chamada])
        avaliador.assert_not_awaited()

    def test_resposta_local_de_falha_do_modelo_nao_e_revisada(self):
        contexto = ContextoGuardrailTeste("oi")
        with self.assertLogs("agentes.compartilhado.guardrail", level="ERROR"):
            local = criar_tratamento_de_erro(PERFIL)(
                contexto, None, RuntimeError("503")
            )
        avaliador = AsyncMock()

        with patch(AVALIADOR, avaliador):
            resposta = asyncio.run(revisar(contexto, local))

        self.assertIsNone(resposta)
        self.assertEqual(local.content.parts[0].text, "Resposta segura.")
        avaliador.assert_not_awaited()

    def test_resposta_vazia_usa_resposta_segura(self):
        contexto = ContextoGuardrailTeste()
        contexto.state["pendente"] = True

        resposta, avaliador, _ = self._revisar(contexto, "")

        self.assertEqual(
            resposta.content.parts[0].text,
            "Resposta segura. Qual é a sua renda mensal?",
        )
        avaliador.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
