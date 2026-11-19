import asyncio
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from agentes.triagem.agent import (
    AGUARDANDO_CPF,
    AGUARDANDO_DATA_NASCIMENTO,
    ATENDIMENTO_ENCERRADO,
    ETAPA_AUTENTICACAO,
    agente_triagem,
    encerrar_apos_limite_de_tentativas,
    interceptar_autenticacao_local,
    revisar_resposta,
)
from agentes.credito.tools.credito import AGUARDANDO_PROXIMA_ACAO, ETAPA_CREDITO
from agentes.agent import root_agent
from agentes.compartilhado.encerramento import (
    AGUARDANDO_CONFIRMACAO_ENCERRAMENTO,
)
from agentes.triagem.tools import consultar_client


AVALIADOR = "agentes.compartilhado.guardrail._avaliar_resposta_com_modelo"
REESCRITOR = "agentes.compartilhado.guardrail._reescrever_resposta_com_modelo"


class ContextoDeTeste:
    def __init__(self):
        self.state = {}
        self.user_content = None
        self.session = SimpleNamespace(events=[])
        self.actions = SimpleNamespace(escalate=False, skip_summarization=False)


def _resposta_texto(texto):
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text=texto)])
    )


class FerramentasTriagemTest(unittest.TestCase):
    def setUp(self):
        self.diretorio_temporario = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.diretorio_temporario.name) / "clientes.csv"
        self.csv_path.write_text(
            'CPF,"Data de Nascimento"\n"710.483.880-50", "29/07/1997"\n',
            encoding="utf-8",
        )
        self.patch_csv = patch.object(consultar_client, "CSV_PATH", self.csv_path)
        self.patch_csv.start()
        self.contexto = ContextoDeTeste()

    def tearDown(self):
        self.patch_csv.stop()
        self.diretorio_temporario.cleanup()

    def _requisicao(self, texto):
        self.contexto.user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=texto)]
        )
        return LlmRequest(
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=texto)])
            ]
        )

    def test_consulta_cpf_formatado_e_numerico(self):
        for cpf in ("710.483.880-50", "71048388050", "cpf: 710.483.880-50"):
            with self.subTest(cpf=cpf):
                resultado = consultar_client.consultar_cliente(cpf, self.contexto)
                self.assertTrue(resultado["cpf_valido"])
                self.assertTrue(resultado["encontrado"])

    def test_cpf_em_frase_natural_inicia_autenticacao(self):
        interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))

        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("Meu CPF é 710.483.880-50, por favor.")
        )

        self.assertIn("data de nascimento", resposta.content.parts[0].text)
        self.assertEqual(
            self.contexto.state[ETAPA_AUTENTICACAO], AGUARDANDO_DATA_NASCIMENTO
        )

    def test_data_em_frase_natural_autentica_variacoes(self):
        entradas = (
            "Minha data de nascimento é 29/07/1997.",
            "Nasci em 29-07-1997.",
            "Minha data de nascimento é 1997-07-29.",
            "Minha data de nascimento é 29 de julho 1997.",
            "Minha data de nascimento é 29 do 7 de 1997.",
            "Minha data de nascimento é 29 de julho de 97.",
        )
        for entrada in entradas:
            with self.subTest(entrada=entrada):
                self.contexto = ContextoDeTeste()
                interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))
                interceptar_autenticacao_local(
                    self.contexto, self._requisicao("Meu CPF é 710.483.880-50.")
                )

                resposta = interceptar_autenticacao_local(
                    self.contexto, self._requisicao(entrada)
                )

                self.assertIn("Autenticação realizada", resposta.content.parts[0].text)
                self.assertTrue(self.contexto.state["cliente_autenticado"])

    def test_cpf_invalidos_nao_sao_consultados(self):
        with patch.object(consultar_client, "_buscar_cliente") as buscar_cliente:
            for cpf in ("123", "111.111.111-11", "710.483.880-51"):
                with self.subTest(cpf=cpf):
                    resultado = consultar_client.consultar_cliente(cpf, self.contexto)
                    self.assertFalse(resultado["cpf_valido"])
                    self.assertFalse(resultado["encontrado"])
            buscar_cliente.assert_not_called()
        self.assertEqual(self.contexto.state["tentativas_falhas"], 3)
        self.assertTrue(resultado["encerrar_atendimento"])

    def test_cpf_valido_nao_cadastrado_registra_falha(self):
        resultado = consultar_client.consultar_cliente("529.982.247-25", self.contexto)
        self.assertTrue(resultado["cpf_valido"])
        self.assertFalse(resultado["encontrado"])
        self.assertEqual(resultado["tentativas_restantes"], 2)

    def test_limite_de_tentativas_bloqueia_nova_consulta(self):
        for cpf in ("529.982.247-25", "123", "111.111.111-11"):
            consultar_client.consultar_cliente(cpf, self.contexto)

        resultado = consultar_client.consultar_cliente(
            "710.483.880-50", self.contexto
        )

        self.assertFalse(resultado["encontrado"])
        self.assertTrue(resultado["encerrar_atendimento"])
        self.assertEqual(self.contexto.state["tentativas_falhas"], 3)

    def test_limite_de_tentativas_bloqueia_nova_autenticacao(self):
        self.contexto.state["tentativas_falhas"] = 3

        resultado = consultar_client.autenticar_cliente(
            "710.483.880-50", "29/07/1997", self.contexto
        )

        self.assertFalse(resultado["autenticado"])
        self.assertTrue(resultado["encerrar_atendimento"])
        self.assertNotIn("cliente_autenticado", self.contexto.state)

    def test_callback_encerra_loop_apos_terceira_falha(self):
        self.assertIsNone(
            encerrar_apos_limite_de_tentativas(
                callback_context=self.contexto, llm_request=None
            )
        )

        self.contexto.state["tentativas_falhas"] = 3
        resposta = encerrar_apos_limite_de_tentativas(
            callback_context=self.contexto, llm_request=None
        )

        self.assertIn("após três tentativas", resposta.content.parts[0].text)
        self.assertTrue(self.contexto.actions.escalate)
        self.assertTrue(self.contexto.actions.skip_summarization)

    def test_autenticacao_local_so_processa_campos_apos_pergunta(self):
        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("710.483.880-50")
        )
        self.assertIn("informe o seu cpf", resposta.content.parts[0].text.lower())
        self.assertEqual(self.contexto.state[ETAPA_AUTENTICACAO], AGUARDANDO_CPF)
        self.assertNotIn("tentativas_falhas", self.contexto.state)

        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("710.483.880-50")
        )
        self.assertIn("data de nascimento", resposta.content.parts[0].text)
        self.assertEqual(
            self.contexto.state[ETAPA_AUTENTICACAO], AGUARDANDO_DATA_NASCIMENTO
        )

        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("29 de julho de 1997")
        )
        self.assertIn("Autenticação realizada", resposta.content.parts[0].text)
        self.assertTrue(self.contexto.state["cliente_autenticado"])
        self.assertIsNone(self.contexto.state[ETAPA_AUTENTICACAO])

    def test_callback_processa_mensagem_atual_em_vez_do_historico(self):
        interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))
        self.contexto.user_content = types.Content(
            role="user", parts=[types.Part.from_text(text="710.483.880-50")]
        )
        requisicao_com_historico_atrasado = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part.from_text(text="oi")]
                )
            ]
        )

        resposta = interceptar_autenticacao_local(
            self.contexto, requisicao_com_historico_atrasado
        )

        self.assertIn("data de nascimento", resposta.content.parts[0].text)
        self.assertEqual(
            self.contexto.state[ETAPA_AUTENTICACAO], AGUARDANDO_DATA_NASCIMENTO
        )

    def test_terceira_falha_no_callback_chama_exit_loop(self):
        interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))

        for cpf in ("529.982.247-25", "168.995.350-09", "111.444.777-35"):
            resposta = interceptar_autenticacao_local(
                self.contexto, self._requisicao(cpf)
            )

        self.assertIn("após três tentativas", resposta.content.parts[0].text)
        self.assertTrue(self.contexto.state[ATENDIMENTO_ENCERRADO])
        self.assertTrue(self.contexto.actions.escalate)
        self.assertTrue(self.contexto.actions.skip_summarization)

    def test_pedido_explicito_de_encerramento_na_autenticacao_e_confirmado_localmente(self):
        casos = (
            (
                AGUARDANDO_CPF,
                "Meu CPF é 710.483.880-50, mas pode encerrar o atendimento.",
            ),
            (
                AGUARDANDO_DATA_NASCIMENTO,
                "Nasci em 29/07/1997 e não quero continuar.",
            ),
        )
        for etapa, mensagem in casos:
            with self.subTest(etapa=etapa):
                self.contexto = ContextoDeTeste()
                interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))
                if etapa == AGUARDANDO_DATA_NASCIMENTO:
                    interceptar_autenticacao_local(
                        self.contexto, self._requisicao("710.483.880-50")
                    )
                estado_antes = dict(self.contexto.state)

                resposta = interceptar_autenticacao_local(
                    self.contexto, self._requisicao(mensagem)
                )

                self.assertIn("encerrar", resposta.content.parts[0].text)
                self.assertTrue(
                    self.contexto.state[AGUARDANDO_CONFIRMACAO_ENCERRAMENTO]
                )
                self.assertNotIn(ATENDIMENTO_ENCERRADO, self.contexto.state)
                self.assertEqual(self.contexto.state[ETAPA_AUTENTICACAO], etapa)
                for chave in (
                    "cpf_em_validacao",
                    "cliente_autenticado",
                    "tentativas_falhas",
                ):
                    self.assertEqual(
                        self.contexto.state.get(chave), estado_antes.get(chave)
                    )
                self.assertNotIn("710.483.880-50", resposta.content.parts[0].text)
                self.assertNotIn("29/07/1997", resposta.content.parts[0].text)

                continuacao = interceptar_autenticacao_local(
                    self.contexto, self._requisicao("Não, continuar")
                )
                self.assertIn("Vamos continuar", continuacao.content.parts[0].text)
                self.assertEqual(self.contexto.state[ETAPA_AUTENTICACAO], etapa)
                self.assertIsNone(
                    self.contexto.state[AGUARDANDO_CONFIRMACAO_ENCERRAMENTO]
                )

    def test_cancelamento_de_produto_nao_e_tratado_como_encerramento(self):
        interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))

        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("quero cancelar meu cartão")
        )

        self.assertIsNone(resposta)
        self.assertNotIn(AGUARDANDO_CONFIRMACAO_ENCERRAMENTO, self.contexto.state)

    def test_atendimento_encerrado_nao_reabre(self):
        self.contexto.state[ATENDIMENTO_ENCERRADO] = True

        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("710.483.880-50")
        )

        self.assertIn("já foi encerrado", resposta.content.parts[0].text)
        self.assertNotIn("cpf_em_validacao", self.contexto.state)

    def test_mensagem_livre_antes_da_autenticacao_vai_ao_modelo_sem_ferramentas(self):
        interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))
        for etapa, mensagem in (
            (AGUARDANDO_CPF, "oi, tudo bem? o que vocês fazem?"),
            (AGUARDANDO_DATA_NASCIMENTO, "por que precisam da minha data?"),
        ):
            with self.subTest(etapa=etapa):
                if etapa == AGUARDANDO_DATA_NASCIMENTO:
                    interceptar_autenticacao_local(
                        self.contexto, self._requisicao("710.483.880-50")
                    )
                requisicao = self._requisicao(mensagem)
                requisicao.config.tools = [types.Tool()]
                requisicao.tools_dict["transfer_to_agent"] = object()

                resposta = interceptar_autenticacao_local(self.contexto, requisicao)

                self.assertIsNone(resposta)
                self.assertFalse(requisicao.config.tools)
                self.assertEqual(requisicao.tools_dict, {})
                self.assertIn(
                    "Não peça CPF, data de nascimento",
                    requisicao.config.system_instruction,
                )
                self.assertEqual(self.contexto.state[ETAPA_AUTENTICACAO], etapa)
                self.assertNotIn("tentativas_falhas", self.contexto.state)
                self.assertNotIn("cliente_autenticado", self.contexto.state)

    def test_data_por_extenso_nao_reconhecida_pede_numeros_sem_falha(self):
        interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))
        interceptar_autenticacao_local(self.contexto, self._requisicao("710.483.880-50"))

        resposta = interceptar_autenticacao_local(
            self.contexto,
            self._requisicao(
                "Nasci em vinte e nove de julho de mil novecentos e noventa e sete."
            ),
        )

        self.assertIn("com números, por exemplo 29/07/1997", resposta.content.parts[0].text)
        self.assertEqual(
            self.contexto.state[ETAPA_AUTENTICACAO], AGUARDANDO_DATA_NASCIMENTO
        )
        self.assertNotIn("tentativas_falhas", self.contexto.state)

    def test_resposta_livre_antes_da_autenticacao_recebe_pedido_do_campo(self):
        self.contexto.state[ETAPA_AUTENTICACAO] = AGUARDANDO_CPF
        self.contexto.user_content = types.Content(
            role="user", parts=[types.Part.from_text(text="oi, tudo bem?")]
        )

        with patch(AVALIADOR, new=AsyncMock(return_value=(True, "ok"))):
            resposta = asyncio.run(
                revisar_resposta(self.contexto, _resposta_texto("Tudo ótimo, obrigado!"))
            )

        self.assertEqual(
            resposta.content.parts[0].text,
            "Tudo ótimo, obrigado! Para continuarmos, informe seu CPF.",
        )

        self.contexto.state[ETAPA_AUTENTICACAO] = AGUARDANDO_DATA_NASCIMENTO
        with patch(AVALIADOR, new=AsyncMock(side_effect=RuntimeError("falha"))), \
                self.assertLogs("agentes.compartilhado.guardrail", level="ERROR"):
            resposta = asyncio.run(
                revisar_resposta(self.contexto, _resposta_texto("Claro!"))
            )
        self.assertEqual(
            resposta.content.parts[0].text,
            "Para continuarmos, informe sua data de nascimento.",
        )

    def test_chamada_de_ferramenta_antes_da_autenticacao_e_descartada(self):
        self.contexto.state[ETAPA_AUTENTICACAO] = AGUARDANDO_CPF
        chamada = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="transfer_to_agent", args={"agent_name": "credito"}
                        )
                    )
                ],
            )
        )

        resposta = asyncio.run(revisar_resposta(self.contexto, chamada))

        self.assertEqual(
            resposta.content.parts[0].text, "Olá! Para continuarmos, informe seu CPF."
        )
        self.assertIsNone(getattr(self.contexto.actions, "transfer_to_agent", None))

    def test_solicitacao_autenticada_livre_e_enviada_ao_modelo(self):
        self.contexto.state["cliente_autenticado"] = True

        requisicao = self._requisicao("queria aumentar meu limite")
        resposta = interceptar_autenticacao_local(self.contexto, requisicao)

        self.assertIsNone(resposta)
        self.assertIsNone(getattr(self.contexto.actions, "transfer_to_agent", None))
        instrucoes = requisicao.config.system_instruction
        self.assertIn("quanto vale meu", instrucoes)
        self.assertIn("transfira para cambio", instrucoes)
        self.assertIn("cumprimentos", instrucoes)

    def test_valor_isolado_apos_credito_retorna_ao_especialista(self):
        self.contexto.state.update(
            {
                "cliente_autenticado": True,
                ETAPA_CREDITO: AGUARDANDO_PROXIMA_ACAO,
            }
        )

        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("20 mil")
        )

        self.assertEqual(resposta.content.parts[0].text, "")
        self.assertEqual(self.contexto.actions.transfer_to_agent, "credito")

    def test_pedido_junto_da_credencial_e_retomado_apos_autenticacao(self):
        interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))
        interceptar_autenticacao_local(
            self.contexto,
            self._requisicao(
                "Boa tarde! Meu CPF é 710.483.880-50. Quero consultar meu score."
            ),
        )
        self.assertTrue(self.contexto.state["pedido_antes_da_autenticacao"])

        requisicao = self._requisicao("29/07/1997")
        resposta = interceptar_autenticacao_local(self.contexto, requisicao)

        self.assertIsNone(resposta)
        self.assertTrue(self.contexto.state["cliente_autenticado"])
        self.assertIsNone(self.contexto.state["pedido_antes_da_autenticacao"])
        self.assertIn("pedido pendente no histórico", requisicao.config.system_instruction)
        self.assertIn("transfira para credito", requisicao.config.system_instruction)

        with self.assertLogs("agentes.compartilhado.guardrail", level="ERROR"):
            falha = agente_triagem.on_model_error_callback(
                self.contexto, requisicao, error=RuntimeError("503")
            )
        self.assertEqual(
            falha.content.parts[0].text,
            "Autenticação realizada com sucesso! Como posso ajudar você hoje?",
        )

        interceptar_autenticacao_local(self.contexto, self._requisicao("obrigado"))
        self.assertIsNone(self.contexto.state["autenticacao_recem_concluida"])

    def test_credenciais_sem_pedido_mantem_confirmacao_local(self):
        interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))
        for mensagem in (
            "bom dia, tudo certo?",
            "oi, tudo bem?",
            "Fala aí, meu CPF é 710.483.880-50.",
            "Nasci em 29/07/1997, mano.",
        ):
            resposta = interceptar_autenticacao_local(
                self.contexto, self._requisicao(mensagem)
            )

        self.assertEqual(
            resposta.content.parts[0].text,
            "Autenticação realizada com sucesso! Como posso ajudar você hoje?",
        )
        self.assertNotIn("pedido_antes_da_autenticacao", self.contexto.state)

    def test_valor_isolado_com_entrevista_pausada_retorna_a_entrevista(self):
        self.contexto.state.update(
            {
                "cliente_autenticado": True,
                ETAPA_CREDITO: AGUARDANDO_PROXIMA_ACAO,
                "etapa_entrevista_credito": "aguardando_renda_mensal",
            }
        )

        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("8000")
        )

        self.assertEqual(resposta.content.parts[0].text, "")
        self.assertEqual(
            self.contexto.actions.transfer_to_agent, "entrevista_credito"
        )

    def test_guardrail_devolve_resposta_insegura_para_reescrita(self):
        self.contexto.state["cliente_autenticado"] = True
        candidata = _resposta_texto(
            "Informe o valor desejado e sua renda mensal para eu processar o "
            "aumento de limite."
        )

        avaliador = AsyncMock(
            side_effect=[
                (False, "Solicita dados para uma operação indisponível."),
                (True, "Resposta corrigida."),
            ]
        )
        reescritor = AsyncMock(
            return_value="Posso direcionar seu pedido ao atendimento de crédito."
        )
        with patch(AVALIADOR, avaliador), patch(REESCRITOR, reescritor):
            resposta = asyncio.run(revisar_resposta(self.contexto, candidata))

        texto = resposta.content.parts[0].text.lower()
        self.assertIn("atendimento de crédito", texto)
        self.assertNotIn("renda mensal", texto)
        reescritor.assert_awaited_once()
        self.assertEqual(avaliador.await_count, 2)

    def test_guardrail_bloqueia_apos_segunda_reprovacao(self):
        self.contexto.state["cliente_autenticado"] = True
        self.contexto.user_content = types.Content(
            role="user", parts=[types.Part.from_text(text="quero melhorar meu score")]
        )

        with patch(
            AVALIADOR,
            new=AsyncMock(return_value=(False, "A transferência não foi executada.")),
        ), patch(REESCRITOR, new=AsyncMock(return_value="Vou verificar.")):
            resposta = asyncio.run(
                revisar_resposta(self.contexto, _resposta_texto("Vou verificar para você."))
            )

        texto = resposta.content.parts[0].text
        self.assertIn("não consegui responder isso com segurança", texto)
        self.assertIn("cotações de moedas", texto)
        self.assertIsNone(getattr(self.contexto.actions, "transfer_to_agent", None))

    def test_guardrail_libera_resposta_aprovada_sem_altera_la(self):
        self.contexto.state["cliente_autenticado"] = True

        with patch(AVALIADOR, new=AsyncMock(return_value=(True, "Resposta segura."))):
            resposta = asyncio.run(
                revisar_resposta(self.contexto, _resposta_texto("De nada! Até logo."))
            )

        self.assertIsNone(resposta)

    def test_falha_do_modelo_principal_retorna_resposta_segura(self):
        self.contexto.state["cliente_autenticado"] = True

        with self.assertLogs("agentes.compartilhado.guardrail", level="ERROR"):
            resposta = agente_triagem.on_model_error_callback(
                self.contexto,
                self._requisicao("quero aumentar meu limite"),
                error=RuntimeError("indisponível"),
            )

        self.assertIn(
            "não consegui responder isso com segurança",
            resposta.content.parts[0].text,
        )

    def test_autenticacao_por_data_de_nascimento(self):
        resultado = consultar_client.autenticar_cliente(
            "710.483.880-50", "29/07/1997", self.contexto
        )
        self.assertTrue(resultado["autenticado"])
        self.assertTrue(self.contexto.state["cliente_autenticado"])
        self.assertEqual(self.contexto.state["cpf_cliente"], "71048388050")

    def test_data_invalida_registra_falha(self):
        for data_nascimento in ("30/07/1997", "29/07/1997x"):
            with self.subTest(data_nascimento=data_nascimento):
                self.contexto = ContextoDeTeste()
                resultado = consultar_client.autenticar_cliente(
                    "71048388050", data_nascimento, self.contexto
                )
                self.assertFalse(resultado["autenticado"])
                self.assertEqual(resultado["tentativas_restantes"], 2)

    def test_autenticacao_bem_sucedida_reinicia_tentativas(self):
        consultar_client.consultar_cliente("123", self.contexto)
        resultado = consultar_client.autenticar_cliente(
            "71048388050", "29/07/1997", self.contexto
        )
        self.assertTrue(resultado["autenticado"])
        self.assertEqual(self.contexto.state["tentativas_falhas"], 0)

    def test_base_malformada_retorna_erro_controlado(self):
        self.csv_path.write_text("identificador\n71048388050\n", encoding="utf-8")
        resultado = consultar_client.consultar_cliente("71048388050", self.contexto)
        self.assertIn("erro", resultado)

    def test_base_ausente_retorna_erro_controlado(self):
        self.csv_path.unlink()
        resultado = consultar_client.consultar_cliente("71048388050", self.contexto)
        self.assertIn("erro", resultado)

    def test_triagem_expoe_confirmacao_e_tem_especialistas_como_subagentes(self):
        ferramentas = asyncio.run(root_agent.canonical_tools())
        nomes = {ferramenta.name for ferramenta in ferramentas}
        self.assertEqual(nomes, {"solicitar_confirmacao_encerramento"})
        self.assertEqual(
            [agente.name for agente in root_agent.sub_agents],
            ["credito", "entrevista_credito", "cambio"],
        )
        for subagente in root_agent.sub_agents:
            self.assertIs(subagente.parent_agent, root_agent)
        for agente in (root_agent, *root_agent.sub_agents):
            opcoes_http = agente.generate_content_config.http_options
            self.assertEqual(opcoes_http.timeout, 20_000)
            self.assertEqual(opcoes_http.retry_options.attempts, 4)
            self.assertEqual(agente.mode, "chat")


if __name__ == "__main__":
    unittest.main()
