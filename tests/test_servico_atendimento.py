from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from google.genai import types

from aplicacao.servico_atendimento import ServicoAtendimento


class EventoTeste:
    def __init__(
        self,
        texto: str = "",
        *,
        final: bool = True,
        encerrado: bool = False,
        pensamento: str = "",
        opcoes_resposta=None,
    ) -> None:
        partes = []
        if pensamento:
            partes.append(types.Part(text=pensamento, thought=True))
        if texto:
            partes.append(types.Part.from_text(text=texto))
        self.content = types.Content(role="model", parts=partes) if partes else None
        self.actions = SimpleNamespace(
            escalate=encerrado,
            state_delta=(
                {"opcoes_resposta": opcoes_resposta}
                if opcoes_resposta is not None
                else {}
            ),
        )
        self._final = final

    def is_final_response(self) -> bool:
        return self._final


class ServicoAtendimentoTest(unittest.TestCase):
    def setUp(self):
        self.patch_runner = patch("aplicacao.servico_atendimento.Runner")
        self.runner = self.patch_runner.start().return_value
        self.servico = ServicoAtendimento(Mock())

    def tearDown(self):
        self.patch_runner.stop()

    def test_reune_respostas_finais_de_multiplos_agentes(self):
        self.runner.run.return_value = [
            EventoTeste("Resposta da triagem."),
            EventoTeste("Evento parcial.", final=False),
            EventoTeste("Resposta do crédito."),
        ]

        resultado = self.servico.enviar_mensagem("Qual é o meu limite?")

        self.assertEqual(
            resultado.mensagens,
            ("Resposta da triagem.", "Resposta do crédito."),
        )
        self.assertFalse(resultado.encerrado)

    def test_oculta_partes_de_pensamento(self):
        self.runner.run.return_value = [
            EventoTeste("Resposta visível.", pensamento="Raciocínio interno.")
        ]

        resultado = self.servico.enviar_mensagem("Olá")

        self.assertEqual(resultado.mensagens, ("Resposta visível.",))

    def test_encerra_somente_com_sinal_padrao_do_adk(self):
        self.runner.run.return_value = [
            EventoTeste("Atendimento encerrado.", encerrado=True)
        ]

        resultado = self.servico.enviar_mensagem("Quero sair")

        self.assertTrue(resultado.encerrado)

    def test_reutiliza_a_mesma_sessao_na_conversa(self):
        self.runner.run.return_value = []

        self.servico.enviar_mensagem("Primeira mensagem")
        self.servico.enviar_mensagem("Segunda mensagem")

        primeira_chamada, segunda_chamada = self.runner.run.call_args_list
        self.assertEqual(
            primeira_chamada.kwargs["user_id"],
            segunda_chamada.kwargs["user_id"],
        )
        self.assertEqual(
            primeira_chamada.kwargs["session_id"],
            segunda_chamada.kwargs["session_id"],
        )

    def test_retorna_opcoes_de_resposta_do_ultimo_evento(self):
        self.runner.run.return_value = [
            EventoTeste(
                "Qual é o seu tipo de emprego?",
                opcoes_resposta=["Formal", "Autônomo", "Desempregado"],
            )
        ]

        resultado = self.servico.enviar_mensagem("6000")

        self.assertEqual(
            resultado.opcoes_resposta,
            ("Formal", "Autônomo", "Desempregado"),
        )

    def test_opcoes_nulas_removem_respostas_rapidas_anteriores(self):
        self.runner.run.return_value = [
            EventoTeste("Processando.", opcoes_resposta=["Sim", "Não"]),
            EventoTeste("Informe suas despesas.", opcoes_resposta=[]),
        ]

        resultado = self.servico.enviar_mensagem("Formal")

        self.assertEqual(resultado.opcoes_resposta, ())

    def test_rejeita_mensagem_vazia(self):
        with self.assertRaisesRegex(ValueError, "não pode estar vazia"):
            self.servico.enviar_mensagem("   ")

        self.runner.run.assert_not_called()

    def test_fecha_o_runner(self):
        self.runner.close = AsyncMock()

        self.servico.fechar()

        self.runner.close.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
