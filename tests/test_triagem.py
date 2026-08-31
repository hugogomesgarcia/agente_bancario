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
    encerrar_apos_limite_de_tentativas,
    interceptar_autenticacao_local,
)
from agentes.triagem.guardrail import (
    CONTEXTO_GUARDRAIL,
    POS_AUTENTICACAO,
    aplicar_guardrail_de_resposta,
    tratar_erro_do_modelo,
)
from agentes.credito.tools.credito import AGUARDANDO_PROXIMA_ACAO, ETAPA_CREDITO
from agentes.agent import root_agent
from agentes.compartilhado.encerramento import (
    AGUARDANDO_CONFIRMACAO_ENCERRAMENTO,
)
from agentes.triagem.tools import consultar_client


class ContextoDeTeste:
    def __init__(self):
        self.state = {}
        self.user_content = None
        self.actions = SimpleNamespace(escalate=False, skip_summarization=False)


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

        self.assertIn("informe seu CPF", resposta.content.parts[0].text)
        self.assertNotIn(AGUARDANDO_CONFIRMACAO_ENCERRAMENTO, self.contexto.state)

    def test_atendimento_encerrado_nao_reabre(self):
        self.contexto.state[ATENDIMENTO_ENCERRADO] = True

        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("710.483.880-50")
        )

        self.assertIn("já foi encerrado", resposta.content.parts[0].text)
        self.assertNotIn("cpf_em_validacao", self.contexto.state)

    def test_entrada_sem_cpf_recebe_lembrete_local_sem_contar_tentativa(self):
        interceptar_autenticacao_local(self.contexto, self._requisicao("oi"))

        resposta = interceptar_autenticacao_local(
            self.contexto, self._requisicao("meu telefone é 11999999999")
        )

        self.assertEqual(
            resposta.content.parts[0].text,
            "Olá! Para continuarmos, informe seu CPF.",
        )
        self.assertEqual(self.contexto.state[ETAPA_AUTENTICACAO], AGUARDANDO_CPF)
        self.assertNotIn("tentativas_falhas", self.contexto.state)

    def test_solicitacao_autenticada_livre_e_enviada_ao_modelo(self):
        self.contexto.state["cliente_autenticado"] = True

        requisicao = self._requisicao("queria aumentar meu limite")
        resposta = interceptar_autenticacao_local(self.contexto, requisicao)

        self.assertIsNone(resposta)
        self.assertIsNone(getattr(self.contexto.actions, "transfer_to_agent", None))
        self.assertIsNone(self.contexto.state["classificacao_registrada"])
        self.assertEqual(
            self.contexto.state[CONTEXTO_GUARDRAIL], POS_AUTENTICACAO
        )

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

    def test_guardrail_devolve_resposta_insegura_para_reescrita(self):
        self.contexto.state.update(
            {
                CONTEXTO_GUARDRAIL: POS_AUTENTICACAO,
                "classificacao_registrada": "credito",
            }
        )
        candidata = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Informe o valor desejado e sua renda mensal para eu "
                            "processar o aumento de limite."
                        )
                    )
                ],
            )
        )

        avaliador = AsyncMock(
            side_effect=[
                (False, "Solicita dados para uma operação indisponível."),
                (True, "Resposta corrigida."),
            ]
        )
        reescritor = AsyncMock(
            return_value=(
                "Entendo seu pedido, mas o atendimento de crédito ainda não "
                "está disponível."
            )
        )
        with patch(
            "agentes.triagem.guardrail._avaliar_resposta_com_modelo", avaliador
        ), patch(
            "agentes.triagem.guardrail._reescrever_resposta_com_modelo", reescritor
        ):
            resposta = asyncio.run(
                aplicar_guardrail_de_resposta(self.contexto, candidata)
            )

        texto = resposta.content.parts[0].text.lower()
        self.assertIn("ainda não está disponível", texto)
        self.assertNotIn("informe o valor", texto)
        self.assertNotIn("renda mensal", texto)
        reescritor.assert_awaited_once()
        self.assertEqual(avaliador.await_count, 2)

    def test_guardrail_nao_infere_transferencia_nao_executada(self):
        mensagens = (
            "quero aumentar meu limite para 8000",
            "quero melhorar meu score",
        )
        for mensagem in mensagens:
            with self.subTest(mensagem=mensagem):
                self.contexto = ContextoDeTeste()
                self.contexto.state.update(
                    {
                        "cliente_autenticado": True,
                        CONTEXTO_GUARDRAIL: POS_AUTENTICACAO,
                    }
                )
                self.contexto.user_content = types.Content(
                    role="user", parts=[types.Part.from_text(text=mensagem)]
                )
                candidata = LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="Vou verificar para você.")],
                    )
                )

                with patch(
                    "agentes.triagem.guardrail._avaliar_resposta_com_modelo",
                    new=AsyncMock(
                        return_value=(False, "A transferência não foi executada.")
                    ),
                ), patch(
                    "agentes.triagem.guardrail._reescrever_resposta_com_modelo",
                    new=AsyncMock(return_value="Não posso realizar essa operação."),
                ):
                    resposta = asyncio.run(
                        aplicar_guardrail_de_resposta(self.contexto, candidata)
                    )

                self.assertIn("Não tenho uma fonte", resposta.content.parts[0].text)
                self.assertIsNone(
                    getattr(self.contexto.actions, "transfer_to_agent", None)
                )

    def test_guardrail_libera_resposta_aprovada_sem_altera_la(self):
        self.contexto.state[CONTEXTO_GUARDRAIL] = POS_AUTENTICACAO
        candidata = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Entendo sua solicitação.")],
            )
        )

        with patch(
            "agentes.triagem.guardrail._avaliar_resposta_com_modelo",
            new=AsyncMock(return_value=(True, "Resposta segura.")),
        ):
            resposta = asyncio.run(
                aplicar_guardrail_de_resposta(self.contexto, candidata)
            )

        self.assertIsNone(resposta)

    def test_guardrail_falha_fechado_se_revisor_ficar_indisponivel(self):
        self.contexto.state[CONTEXTO_GUARDRAIL] = POS_AUTENTICACAO
        candidata = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Resposta não verificada.")],
            )
        )

        with patch(
            "agentes.triagem.guardrail._avaliar_resposta_com_modelo",
            new=AsyncMock(side_effect=RuntimeError("indisponível")),
        ), self.assertLogs("agentes.triagem.guardrail", level="ERROR"):
            resposta = asyncio.run(
                aplicar_guardrail_de_resposta(self.contexto, candidata)
            )

        self.assertIn(
            "Não tenho uma fonte ou ferramenta disponível",
            resposta.content.parts[0].text,
        )

    def test_falha_do_modelo_principal_retorna_resposta_segura(self):
        self.contexto.state[CONTEXTO_GUARDRAIL] = POS_AUTENTICACAO

        with self.assertLogs("agentes.triagem.guardrail", level="ERROR"):
            resposta = tratar_erro_do_modelo(
                self.contexto,
                self._requisicao("quero aumentar meu limite"),
                error=RuntimeError("indisponível"),
            )

        self.assertIn(
            "Não tenho uma fonte ou ferramenta disponível",
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

    def test_triagem_expoe_classificacao_e_tem_especialistas_como_subagentes(self):
        ferramentas = asyncio.run(root_agent.canonical_tools())
        nomes = {ferramenta.name for ferramenta in ferramentas}
        self.assertEqual(
            nomes,
            {"registrar_classificacao", "solicitar_confirmacao_encerramento"},
        )
        self.assertEqual(
            [agente.name for agente in root_agent.sub_agents],
            ["credito", "entrevista_credito"],
        )
        for subagente in root_agent.sub_agents:
            self.assertIs(subagente.parent_agent, root_agent)

    def test_registro_de_classificacao_preserva_semantica_de_placeholder(self):
        resultado = consultar_client.registrar_classificacao("cambio", self.contexto)

        self.assertTrue(resultado["registrado"])
        self.assertFalse(resultado["encaminhamento_executado"])
        self.assertEqual(self.contexto.state["classificacao_registrada"], "cambio")

    def test_registro_de_classificacao_rejeita_destinos_nao_suportados(self):
        for destino in ("credito", "entrevista_credito", "inexistente"):
            with self.subTest(destino=destino):
                self.contexto = ContextoDeTeste()
                resultado = consultar_client.registrar_classificacao(
                    destino, self.contexto
                )

                self.assertFalse(resultado["registrado"])
                self.assertIn("erro", resultado)
                self.assertNotIn("classificacao_registrada", self.contexto.state)


if __name__ == "__main__":
    unittest.main()
