import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from aplicacao.servico_atendimento import ResultadoAtendimento


ROOT = Path(__file__).resolve().parents[1]


class ServicoRespostaRapidaTeste:
    def __init__(self):
        self.mensagens = []

    def enviar_mensagem(self, texto):
        self.mensagens.append(texto)
        return ResultadoAtendimento(
            ("Qual é o total das suas despesas fixas mensais?",),
            False,
        )

    def fechar(self):
        pass


class ServicoOpcoesImediatasTeste:
    def enviar_mensagem(self, texto):
        return ResultadoAtendimento(
            ("Você deseja continuar com a entrevista?",),
            False,
            ("Sim", "Não"),
        )


class ServicoEncerramentoTeste:
    def __init__(self):
        self.fechado = False

    def enviar_mensagem(self, texto):
        return ResultadoAtendimento(("Atendimento encerrado.",), True)

    def fechar(self):
        self.fechado = True


class ServicoProcessamentoVisivelTeste:
    def __init__(self):
        self.mensagem_pendente_ao_enviar = None

    def enviar_mensagem(self, texto):
        import streamlit as st

        self.mensagem_pendente_ao_enviar = st.session_state.mensagem_pendente
        return ResultadoAtendimento(("Resposta concluída.",), False)

    def fechar(self):
        pass


class InterfaceTest(unittest.TestCase):
    def setUp(self):
        self.aplicacao = AppTest.from_file(
            ROOT / "interface.py", default_timeout=20
        ).run()

    def _enviar(self, texto: str) -> None:
        self.aplicacao.chat_input[0].set_value(texto).run()

    def _substituir_servico(self, servico) -> None:
        self.aplicacao.session_state.servico_atendimento.fechar()
        self.aplicacao.session_state.servico_atendimento = servico

    def test_inicia_com_saudacao_e_campo_em_portugues(self):
        self.assertEqual(len(self.aplicacao.exception), 0)
        self.assertEqual(len(self.aplicacao.chat_message), 1)
        self.assertEqual(self.aplicacao.chat_message[0].name, "Banco Ágil")
        self.assertIn(
            "informe o seu CPF",
            self.aplicacao.chat_message[0].markdown[0].value,
        )
        self.assertEqual(
            self.aplicacao.chat_input[0].placeholder, "Digite sua mensagem"
        )

    def test_distingue_mensagem_do_cliente_e_do_banco(self):
        self._enviar("529.982.247-25")

        mensagens = self.aplicacao.chat_message
        self.assertEqual(mensagens[-2].name, "Cliente")
        self.assertEqual(mensagens[-2].markdown[0].value, "529.982.247-25")
        self.assertEqual(mensagens[-1].name, "Banco Ágil")

    def test_escapa_cifrao_para_nao_interpretar_moeda_como_matematica(self):
        self.aplicacao.session_state.mensagens = [
            {"autor": "assistente", "texto": "Limite de R$ 5.000,00 para R$ 7.000,00."}
        ]

        self.aplicacao.run()

        self.assertEqual(
            self.aplicacao.chat_message[0].markdown[0].value,
            r"Limite de R\$ 5.000,00 para R\$ 7.000,00.",
        )

    def test_encerra_e_inicia_nova_conversa(self):
        servico = ServicoEncerramentoTeste()
        self._substituir_servico(servico)
        self._enviar("encerrar")

        self.assertEqual(len(self.aplicacao.exception), 0)
        self.assertEqual(len(self.aplicacao.chat_input), 0)
        self.assertEqual(
            self.aplicacao.button[0].label, "Iniciar novo atendimento"
        )
        self.assertEqual(
            self.aplicacao.info[0].value, "Este atendimento foi encerrado."
        )

        self.aplicacao.button[0].click().run()

        self.assertTrue(servico.fechado)
        self.assertEqual(len(self.aplicacao.exception), 0)
        self.assertEqual(len(self.aplicacao.chat_message), 1)
        self.assertEqual(len(self.aplicacao.chat_input), 1)
        self.assertEqual(len(self.aplicacao.button), 0)

    def test_botao_envia_rotulo_pelo_mesmo_fluxo_da_mensagem(self):
        servico = ServicoRespostaRapidaTeste()
        self._substituir_servico(servico)
        self.aplicacao.session_state.mensagens = [
            {
                "autor": "assistente",
                "texto": "Qual é o seu tipo de emprego?",
                "opcoes_resposta": [
                    "Formal",
                    "Autônomo",
                    "Desempregado",
                ],
            }
        ]
        self.aplicacao.session_state.atendimento_encerrado = False
        self.aplicacao.run()

        botoes = {botao.label: botao for botao in self.aplicacao.button}
        self.assertEqual(
            set(botoes), {"Formal", "Autônomo", "Desempregado"}
        )
        botoes["Autônomo"].click().run()

        self.assertEqual(servico.mensagens, ["Autônomo"])
        mensagens_cliente = [
            mensagem
            for mensagem in self.aplicacao.chat_message
            if mensagem.name == "Cliente"
        ]
        self.assertEqual(
            mensagens_cliente[-1].markdown[0].value, "Autônomo"
        )
        botao_selecionado = next(
            botao
            for botao in self.aplicacao.button
            if botao.label == "Autônomo"
        )
        self.assertTrue(botao_selecionado.disabled)

    def test_exibe_opcoes_junto_da_resposta_recebida(self):
        self._substituir_servico(ServicoOpcoesImediatasTeste())

        self._enviar("sim")

        self.assertEqual(
            [botao.label for botao in self.aplicacao.button], ["Sim", "Não"]
        )

    def test_registra_mensagem_pendente_antes_de_chamar_o_agente(self):
        servico = ServicoProcessamentoVisivelTeste()
        self._substituir_servico(servico)

        self._enviar("cotação do iene")

        self.assertEqual(servico.mensagem_pendente_ao_enviar, "cotação do iene")
        self.assertNotIn("mensagem_pendente", self.aplicacao.session_state)
        self.assertEqual(
            self.aplicacao.chat_message[-2].markdown[0].value,
            "cotação do iene",
        )
        self.assertEqual(
            self.aplicacao.chat_message[-1].markdown[0].value,
            "Resposta concluída.",
        )


if __name__ == "__main__":
    unittest.main()
