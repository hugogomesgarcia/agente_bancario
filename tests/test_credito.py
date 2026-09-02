import csv
import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agentes.credito.agent import (
    _solicitou_aumento_limite,
    _solicitou_aumento_sem_valor,
    _solicitou_consulta_limite,
    _solicitou_consulta_score,
    _solicitou_entrevista_credito,
    agente_credito,
    interceptar_fluxo_credito,
    restringir_resposta_livre,
)
from agentes.credito.tools import credito
from agentes.agent import migrar_faixas_limite, root_agent
from agentes.compartilhado.estado import OPCOES_RESPOSTA
from agentes.compartilhado.encerramento import solicitar_confirmacao_encerramento
from agentes.entrevista_credito.tools import entrevista_credito


class ModeloTransferenciaTeste(BaseLlm):
    chamadas_credito: int = 0

    async def generate_content_async(self, llm_request, stream=False):
        nomes = set(llm_request.tools_dict)
        texto_usuario = ""
        for conteudo in reversed(llm_request.contents):
            if conteudo.role == "user":
                texto_atual = "".join(
                    parte.text or "" for parte in conteudo.parts or []
                )
                if texto_atual:
                    texto_usuario = texto_atual
                    break
        if "outro assunto" in texto_usuario.lower():
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Como posso ajudar?")],
                )
            )
            return
        if "solicitar_aumento_limite" not in nomes:
            chamada = types.FunctionCall(
                name="transfer_to_agent", args={"agent_name": "credito"}
            )
        else:
            self.chamadas_credito += 1
            chamada = types.FunctionCall(
                name="solicitar_aumento_limite",
                args={"novo_limite_solicitado": "8000"},
            )
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(function_call=chamada)]
            )
        )


class ContextoCreditoTeste:
    def __init__(self, *, autenticado=True, invocation_id="invocacao-1"):
        self.state = {}
        if autenticado:
            self.state.update(
                {"cliente_autenticado": True, "cpf_cliente": "71048388050"}
            )
        self.invocation_id = invocation_id
        self.user_content = None
        self.actions = SimpleNamespace(
            escalate=False,
            skip_summarization=False,
            transfer_to_agent=None,
        )


class FerramentasCreditoTest(unittest.TestCase):
    def setUp(self):
        self.diretorio = tempfile.TemporaryDirectory()
        raiz = Path(self.diretorio.name)
        self.clientes = raiz / "clientes.csv"
        self.faixas = raiz / "score_limite.csv"
        self.solicitacoes = raiz / "solicitacoes_aumento_limite.csv"
        self.clientes.write_text(
            'CPF,"Data de Nascimento",Score,"Limite de Crédito"\n'
            '"710.483.880-50","29/07/1997",780,5000.00\n',
            encoding="utf-8",
        )
        self.faixas.write_text(
            "score_minimo,score_maximo,limite_maximo\n"
            "0,699,5000.00\n"
            "700,749,10000.00\n"
            "750,849,15000.00\n"
            "850,1000,20000.00\n",
            encoding="utf-8",
        )
        self.solicitacoes.write_text(
            ",".join(credito.COLUNAS_SOLICITACAO) + "\n", encoding="utf-8"
        )
        self.patches = [
            patch.object(credito, "CLIENTES_CSV", self.clientes),
            patch.object(credito, "SCORE_LIMITE_CSV", self.faixas),
            patch.object(credito, "SOLICITACOES_CSV", self.solicitacoes),
        ]
        for patch_atual in self.patches:
            patch_atual.start()
        self.contexto = ContextoCreditoTeste()

    def tearDown(self):
        for patch_atual in reversed(self.patches):
            patch_atual.stop()
        self.diretorio.cleanup()

    def _linhas(self, caminho):
        with caminho.open("r", encoding="utf-8", newline="") as arquivo:
            return list(csv.DictReader(arquivo))

    def test_consulta_limite_usa_cpf_do_estado_autenticado(self):
        resultado = credito.consultar_limite_credito(self.contexto)

        self.assertTrue(resultado["sucesso"])
        self.assertEqual(resultado["limite_atual"], "5000.00")
        self.assertNotIn("cpf_cliente", resultado)

    def test_consulta_score_usa_cpf_do_estado_autenticado(self):
        resultado = credito.consultar_score_credito(self.contexto)

        self.assertTrue(resultado["sucesso"])
        self.assertEqual(resultado["score_atual"], 780)

    def test_normaliza_valores_monetarios_em_linguagem_usual(self):
        self.assertEqual(
            credito.normalizar_valor_monetario("9 mil reais"),
            Decimal("9000.00"),
        )
        self.assertEqual(
            credito.normalizar_valor_monetario("9,5 mil"),
            Decimal("9500.00"),
        )
        self.assertEqual(
            credito.normalizar_valor_monetario("12k"),
            Decimal("12000.00"),
        )

    def test_ferramentas_recusam_cliente_nao_autenticado(self):
        contexto = ContextoCreditoTeste(autenticado=False)

        consulta = credito.consultar_limite_credito(contexto)
        aumento = credito.solicitar_aumento_limite("8000", contexto)

        self.assertEqual(consulta["erro"], "cliente_nao_autenticado")
        self.assertEqual(aumento["erro"], "cliente_nao_autenticado")
        self.assertEqual(self._linhas(self.solicitacoes), [])

    def test_aprova_registra_e_atualiza_limite(self):
        resultado = credito.solicitar_aumento_limite(
            "R$ 15.000,00", self.contexto
        )

        self.assertEqual(resultado["status"], "aprovado")
        self.assertEqual(resultado["score_atual"], 780)
        self.assertEqual(resultado["limite_maximo_score"], "15000.00")
        cliente = self._linhas(self.clientes)[0]
        self.assertEqual(cliente["Limite de Crédito"], "15000.00")
        solicitacao = self._linhas(self.solicitacoes)[0]
        self.assertEqual(solicitacao["status_pedido"], "aprovado")
        self.assertEqual(solicitacao["limite_atual"], "5000.00")
        self.assertEqual(solicitacao["novo_limite_solicitado"], "15000.00")
        datetime.fromisoformat(solicitacao["data_hora_solicitacao"])

    def test_rejeita_e_preserva_limite_atual(self):
        resultado = credito.solicitar_aumento_limite(
            "15000.01", self.contexto
        )

        self.assertEqual(resultado["status"], "rejeitado")
        self.assertEqual(resultado["score_atual"], 780)
        self.assertEqual(resultado["limite_maximo_score"], "15000.00")
        self.assertEqual(
            self._linhas(self.clientes)[0]["Limite de Crédito"], "5000.00"
        )
        self.assertEqual(
            self._linhas(self.solicitacoes)[0]["status_pedido"], "rejeitado"
        )
        self.assertEqual(
            self.contexto.state[credito.ETAPA_CREDITO],
            credito.AGUARDANDO_DECISAO_ENTREVISTA,
        )
        self.assertEqual(
            self.contexto.state[credito.PENDENCIA_REANALISE][
                "novo_limite_solicitado"
            ],
            "15000.01",
        )

    def test_perfil_de_referencia_aprova_limite_de_treze_mil(self):
        score = entrevista_credito.calcular_score_credito(
            "12000", "formal", "2000", 0, False
        )
        self.assertEqual(score, 680)
        self.clientes.write_text(
            'CPF,"Data de Nascimento",Score,"Limite de Crédito"\n'
            f'"710.483.880-50","29/07/1997",{score},5000.00\n',
            encoding="utf-8",
        )
        faixas_padrao = (
            Path(__file__).resolve().parents[1]
            / "csv"
            / "default"
            / "score_limite.csv"
        )

        with patch.object(credito, "SCORE_LIMITE_CSV", faixas_padrao):
            resultado = credito.solicitar_aumento_limite("13000", self.contexto)

        self.assertEqual(resultado["status"], "aprovado")
        self.assertEqual(resultado["score_atual"], 680)
        self.assertEqual(resultado["limite_maximo_score"], "15000.00")

    def test_valor_invalido_ou_sem_aumento_nao_e_registrado(self):
        for valor, erro in (
            ("infinito", "valor_invalido"),
            ("NaN", "valor_invalido"),
            ("1e100", "valor_invalido"),
            ("8000.001", "valor_invalido"),
            ("5000", "valor_nao_representa_aumento"),
        ):
            with self.subTest(valor=valor):
                resultado = credito.solicitar_aumento_limite(
                    valor, ContextoCreditoTeste(invocation_id=valor)
                )
                self.assertEqual(resultado["erro"], erro)
        self.assertEqual(self._linhas(self.solicitacoes), [])

    def test_perfil_sem_score_retorna_erro_especifico(self):
        self.clientes.write_text(
            'CPF,"Data de Nascimento",Score,"Limite de Crédito"\n'
            '"710.483.880-50","29/07/1997",,5000.00\n',
            encoding="utf-8",
        )

        resultado = credito.consultar_limite_credito(self.contexto)

        self.assertEqual(resultado["erro"], "perfil_credito_indisponivel")

    def test_mesma_invocacao_nao_duplica_solicitacao(self):
        primeiro = credito.solicitar_aumento_limite("8000", self.contexto)
        segundo = credito.solicitar_aumento_limite("8000", self.contexto)

        self.assertEqual(segundo, primeiro)
        self.assertEqual(len(self._linhas(self.solicitacoes)), 1)

    def test_faixas_sobrepostas_falham_de_forma_controlada(self):
        self.faixas.write_text(
            "score_minimo,score_maximo,limite_maximo\n"
            "0,800,10000\n700,1000,20000\n",
            encoding="utf-8",
        )

        with self.assertLogs(level="ERROR"):
            resultado = credito.solicitar_aumento_limite("8000", self.contexto)

        self.assertEqual(resultado["erro"], "base_credito_indisponivel")
        self.assertEqual(self._linhas(self.solicitacoes), [])

    def test_falha_no_registro_reverte_atualizacao_do_cliente(self):
        substituir_original = credito._substituir_csv
        chamadas = 0

        def falhar_na_solicitacao(temporario, destino):
            nonlocal chamadas
            chamadas += 1
            if chamadas == 2:
                raise credito.BaseCreditoIndisponivel
            substituir_original(temporario, destino)

        with patch.object(
            credito, "_substituir_csv", side_effect=falhar_na_solicitacao
        ), self.assertLogs(level="ERROR"):
            resultado = credito.solicitar_aumento_limite("8000", self.contexto)

        self.assertEqual(resultado["erro"], "base_credito_indisponivel")
        self.assertEqual(
            self._linhas(self.clientes)[0]["Limite de Crédito"], "5000.00"
        )
        self.assertEqual(self._linhas(self.solicitacoes), [])

    def test_journal_recupera_quando_o_rollback_tambem_falha(self):
        substituir_original = credito._substituir_csv
        chamadas = 0

        def falhar_no_registro_e_no_primeiro_rollback(temporario, destino):
            nonlocal chamadas
            chamadas += 1
            if chamadas in {2, 3}:
                raise credito.BaseCreditoIndisponivel
            substituir_original(temporario, destino)

        with patch.object(
            credito,
            "_substituir_csv",
            side_effect=falhar_no_registro_e_no_primeiro_rollback,
        ), self.assertLogs(level="CRITICAL"):
            resultado = credito.solicitar_aumento_limite("8000", self.contexto)

        self.assertEqual(resultado["erro"], "base_credito_indisponivel")
        self.assertTrue((self.clientes.parent / ".transacao_credito.json").exists())
        self.assertTrue(credito.recuperar_transacao_pendente())
        self.assertEqual(
            self._linhas(self.clientes)[0]["Limite de Crédito"], "5000.00"
        )
        self.assertEqual(self._linhas(self.solicitacoes), [])
        self.assertFalse(
            (self.clientes.parent / ".transacao_credito.json").exists()
        )

    def test_journal_malformado_falha_sem_interromper_a_recuperacao(self):
        caminho = self.clientes.parent / ".transacao_credito.json"
        caminho.write_text(
            '{"campos_clientes": ["CPF"], "clientes": [1], '
            '"campos_solicitacoes": ["cpf_cliente"], "solicitacoes": []}',
            encoding="utf-8",
        )

        with self.assertLogs(level="ERROR"):
            recuperado = credito.recuperar_transacao_pendente()

        self.assertFalse(recuperado)
        self.assertTrue(caminho.exists())

    def test_migracao_preserva_cliente_e_adiciona_dados_do_seed(self):
        local = Path(self.diretorio.name) / "clientes_antigos.csv"
        padrao = Path(self.diretorio.name) / "clientes_padrao.csv"
        local.write_text(
            'CPF,"Data de Nascimento"\n"710.483.880-50","29/07/1997"\n',
            encoding="utf-8",
        )
        padrao.write_text(
            'CPF,"Data de Nascimento",Score,"Limite de Crédito"\n'
            '"710.483.880-50","29/07/1997",720,5000.00\n',
            encoding="utf-8",
        )

        credito.migrar_base_clientes(padrao, local)

        cliente = self._linhas(local)[0]
        self.assertEqual(cliente["Data de Nascimento"], "29/07/1997")
        self.assertEqual(cliente["Score"], "720")
        self.assertEqual(cliente["Limite de Crédito"], "5000.00")


class PoliticaLimitePadraoTest(unittest.TestCase):
    FAIXAS = (
        (0, 299, "1000.00"),
        (300, 399, "2500.00"),
        (400, 499, "5000.00"),
        (500, 549, "7500.00"),
        (550, 574, "10000.00"),
        (575, 599, "11000.00"),
        (600, 624, "12000.00"),
        (625, 649, "13000.00"),
        (650, 674, "14000.00"),
        (675, 699, "15000.00"),
        (700, 724, "16000.00"),
        (725, 749, "17000.00"),
        (750, 774, "18000.00"),
        (775, 799, "19000.00"),
        (800, 1000, "20000.00"),
    )

    def setUp(self):
        self.faixas_padrao = (
            Path(__file__).resolve().parents[1]
            / "csv"
            / "default"
            / "score_limite.csv"
        )

    def test_limites_inferior_e_superior_de_todas_as_faixas(self):
        with patch.object(credito, "SCORE_LIMITE_CSV", self.faixas_padrao):
            for minimo, maximo, limite in self.FAIXAS:
                for score in (minimo, maximo):
                    with self.subTest(score=score):
                        self.assertEqual(
                            credito._limite_permitido(score), Decimal(limite)
                        )

    def test_perfis_representativos_usam_formula_inalterada(self):
        casos = (
            (("12000", "formal", "2000", 0, False), 680, "15000.00"),
            (("12000", "formal", "2000", 0, True), 480, "5000.00"),
            (("12000", "formal", "2000", 2, False), 640, "13000.00"),
            (("12000", "formal", "2000", 3, False), 610, "12000.00"),
            (("12000", "formal", "11000", 0, False), 533, "7500.00"),
        )
        with patch.object(credito, "SCORE_LIMITE_CSV", self.faixas_padrao):
            for argumentos, score_esperado, limite_esperado in casos:
                with self.subTest(argumentos=argumentos):
                    score = entrevista_credito.calcular_score_credito(*argumentos)
                    self.assertEqual(score, score_esperado)
                    self.assertEqual(
                        credito._limite_permitido(score), Decimal(limite_esperado)
                    )

    def test_migracao_substitui_politica_anterior_e_preserva_customizada(self):
        with tempfile.TemporaryDirectory() as diretorio:
            raiz = Path(diretorio)
            padrao = raiz / "padrao.csv"
            local = raiz / "local.csv"
            nova_politica = self.faixas_padrao.read_text(encoding="utf-8")
            padrao.write_text(nova_politica, encoding="utf-8")
            local.write_text(
                "score_minimo,score_maximo,limite_maximo\n"
                "0,299,1000.00\n"
                "300,499,2500.00\n"
                "500,699,5000.00\n"
                "700,749,10000.00\n"
                "750,849,15000.00\n"
                "850,1000,20000.00\n",
                encoding="utf-8",
            )

            migrar_faixas_limite(padrao, local)
            self.assertEqual(local.read_text(encoding="utf-8"), nova_politica)

            politica_customizada = (
                "score_minimo,score_maximo,limite_maximo\n0,1000,9999.00\n"
            )
            local.write_text(politica_customizada, encoding="utf-8")
            migrar_faixas_limite(padrao, local)
            self.assertEqual(
                local.read_text(encoding="utf-8"), politica_customizada
            )


class IntencoesCreditoTest(unittest.TestCase):
    def test_distingue_consulta_de_pedido_para_melhorar_score(self):
        self.assertTrue(_solicitou_consulta_score("Qual é meu score?"))
        self.assertFalse(_solicitou_consulta_score("Quero aumentar meu score"))
        self.assertTrue(_solicitou_entrevista_credito("Quero aumentar meu score"))

    def test_distingue_consulta_de_aumento_de_limite(self):
        self.assertTrue(_solicitou_consulta_limite("Qual é meu limite atual?"))
        self.assertFalse(_solicitou_consulta_limite("Quero um limite maior"))
        self.assertTrue(_solicitou_aumento_limite("Quero um limite maior"))

    def test_reconhece_sinonimos_e_respeita_negativa(self):
        for mensagem in (
            "Quero elevar meu limite",
            "Pode ampliar meu limite?",
            "Quero subir o limite",
            "Preciso de mais limite",
        ):
            with self.subTest(mensagem=mensagem):
                self.assertTrue(_solicitou_aumento_limite(mensagem))

        self.assertFalse(_solicitou_aumento_limite("Não quero aumentar meu limite"))

    def test_mil_e_k_contam_como_valor_informado(self):
        self.assertFalse(
            _solicitou_aumento_sem_valor("Quero aumentar meu limite para 9 mil")
        )
        self.assertFalse(
            _solicitou_aumento_sem_valor("Quero aumentar meu limite para 12k")
        )


class FluxoCreditoTest(unittest.TestCase):
    def _requisicao(self, contexto, texto, *, retorno_ferramenta=False):
        contexto.user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=texto)]
        )
        partes = [types.Part.from_text(text=texto)]
        if retorno_ferramenta:
            partes = [
                types.Part(
                    function_response=types.FunctionResponse(
                        name="solicitar_aumento_limite", response={"sucesso": True}
                    )
                )
            ]
        return LlmRequest(contents=[types.Content(role="user", parts=partes)])

    def test_resposta_de_aprovacao_vem_da_evidencia_da_ferramenta(self):
        contexto = ContextoCreditoTeste()
        contexto.state[credito.RESULTADO_CREDITO] = {
            "sucesso": True,
            "tipo": "aumento_limite",
            "status": "aprovado",
            "limite_atual": "5000.00",
            "novo_limite_solicitado": "8000.00",
        }

        resposta = interceptar_fluxo_credito(
            contexto,
            self._requisicao(contexto, "8000", retorno_ferramenta=True),
        )

        texto = resposta.content.parts[0].text
        self.assertIn("solicitação foi aprovada", texto)
        self.assertIn("R$ 8.000,00", texto)

    def test_aumento_claro_sem_valor_mantem_etapa_aguardando_novo_limite(self):
        contexto = ContextoCreditoTeste()

        resposta = interceptar_fluxo_credito(
            contexto,
            self._requisicao(
                contexto, "queria aumentar o crédito do meu cartão"
            ),
        )

        self.assertIn("novo limite total", resposta.content.parts[0].text)
        self.assertEqual(
            contexto.state[credito.ETAPA_CREDITO],
            credito.AGUARDANDO_NOVO_LIMITE,
        )
        self.assertIsNone(contexto.actions.transfer_to_agent)

        interceptar_fluxo_credito(
            contexto,
            self._requisicao(contexto, "ainda não decidi o valor"),
        )
        self.assertEqual(
            contexto.state[credito.ETAPA_CREDITO],
            credito.AGUARDANDO_NOVO_LIMITE,
        )

    def test_valor_apos_pedido_de_aumento_e_processado_sem_modelo(self):
        contexto = ContextoCreditoTeste()
        contexto.state[credito.ETAPA_CREDITO] = credito.AGUARDANDO_NOVO_LIMITE
        resultado = {
            "sucesso": True,
            "tipo": "aumento_limite",
            "status": "aprovado",
            "limite_atual": "5000.00",
            "novo_limite_solicitado": "8000.00",
        }

        with patch(
            "agentes.credito.agent.solicitar_aumento_limite",
            return_value=resultado,
        ) as solicitar:
            resposta = interceptar_fluxo_credito(
                contexto, self._requisicao(contexto, "R$ 8.000,00")
            )

        solicitar.assert_called_once_with("R$ 8.000,00", contexto)
        self.assertIn("solicitação foi aprovada", resposta.content.parts[0].text)

    def test_aceite_conversacional_transfere_para_entrevista(self):
        contexto = ContextoCreditoTeste()
        contexto.state[credito.ETAPA_CREDITO] = credito.AGUARDANDO_DECISAO_ENTREVISTA

        interceptar_fluxo_credito(
            contexto, self._requisicao(contexto, "Sim, eu quero fazer a entrevista.")
        )

        self.assertEqual(contexto.actions.transfer_to_agent, "entrevista_credito")
        self.assertIsNone(contexto.state[credito.ETAPA_CREDITO])

    def test_pedido_direto_de_entrevista_ou_melhora_do_score_transfere(self):
        for mensagem in (
            "Quero fazer uma entrevista de crédito.",
            "Como posso melhorar meu score?",
            "Quero aumentar meu score.",
        ):
            with self.subTest(mensagem=mensagem):
                contexto = ContextoCreditoTeste()

                resposta = interceptar_fluxo_credito(
                    contexto, self._requisicao(contexto, mensagem)
                )

                self.assertEqual(
                    contexto.actions.transfer_to_agent, "entrevista_credito"
                )
                self.assertEqual(resposta.content.parts[0].text, "")

    def test_recusa_conversacional_com_consulta_de_limite_usa_ferramenta(self):
        contexto = ContextoCreditoTeste()
        contexto.state.update(
            {
                credito.ETAPA_CREDITO: credito.AGUARDANDO_DECISAO_ENTREVISTA,
                credito.PENDENCIA_REANALISE: {
                    "novo_limite_solicitado": "12000.00"
                },
            }
        )
        resultado = {
            "sucesso": True,
            "tipo": "consulta_limite",
            "limite_atual": "5000.00",
        }

        with patch(
            "agentes.credito.agent.consultar_limite_credito",
            return_value=resultado,
        ) as consultar:
            resposta = interceptar_fluxo_credito(
                contexto,
                self._requisicao(
                    contexto,
                    "Não quero fazer entrevista, mas qual é meu limite atual?",
                ),
            )

        consultar.assert_called_once_with(contexto)
        self.assertIn("R$ 5.000,00", resposta.content.parts[0].text)
        self.assertIsNone(contexto.state[credito.PENDENCIA_REANALISE])
        self.assertEqual(
            contexto.state[credito.ETAPA_CREDITO],
            credito.AGUARDANDO_PROXIMA_ACAO,
        )
        self.assertIsNone(contexto.actions.transfer_to_agent)

    def test_recusa_conversacional_com_consulta_de_score_usa_ferramenta(self):
        contexto = ContextoCreditoTeste()
        contexto.state.update(
            {
                credito.ETAPA_CREDITO: credito.AGUARDANDO_DECISAO_ENTREVISTA,
                credito.PENDENCIA_REANALISE: {
                    "novo_limite_solicitado": "12000.00"
                },
            }
        )
        resultado = {
            "sucesso": True,
            "tipo": "consulta_score",
            "score_atual": 780,
        }

        with patch(
            "agentes.credito.agent.consultar_score_credito",
            return_value=resultado,
        ) as consultar:
            resposta = interceptar_fluxo_credito(
                contexto,
                self._requisicao(
                    contexto,
                    "Prefiro não fazer a entrevista; qual é meu score?",
                ),
            )

        consultar.assert_called_once_with(contexto)
        self.assertIn("780", resposta.content.parts[0].text)
        self.assertIsNone(contexto.state[credito.PENDENCIA_REANALISE])
        self.assertEqual(
            contexto.state[credito.ETAPA_CREDITO],
            credito.AGUARDANDO_PROXIMA_ACAO,
        )
        self.assertIsNone(contexto.actions.transfer_to_agent)

    def test_recusa_conversacional_com_novo_aumento_preserva_nova_etapa(self):
        contexto = ContextoCreditoTeste()
        contexto.state.update(
            {
                credito.ETAPA_CREDITO: credito.AGUARDANDO_DECISAO_ENTREVISTA,
                credito.PENDENCIA_REANALISE: {
                    "novo_limite_solicitado": "12000.00"
                },
            }
        )

        resposta = interceptar_fluxo_credito(
            contexto,
            self._requisicao(
                contexto,
                "Não quero entrevista; quero aumentar meu limite para 9000.",
            ),
        )

        self.assertIsNone(resposta)
        self.assertIsNone(contexto.state[credito.PENDENCIA_REANALISE])
        self.assertEqual(
            contexto.state[credito.ETAPA_CREDITO],
            credito.AGUARDANDO_NOVO_LIMITE,
        )

    def test_correcao_apos_consulta_pede_novo_limite_e_preserva_etapa(self):
        contexto = ContextoCreditoTeste()
        contexto.state[credito.ETAPA_CREDITO] = credito.AGUARDANDO_PROXIMA_ACAO

        resposta = interceptar_fluxo_credito(
            contexto, self._requisicao(contexto, "eu falei que quero aumentar")
        )

        self.assertIn("novo limite total", resposta.content.parts[0].text)
        self.assertEqual(
            contexto.state[credito.ETAPA_CREDITO],
            credito.AGUARDANDO_NOVO_LIMITE,
        )

    def test_valor_isolado_apos_decisao_e_novo_pedido_de_aumento(self):
        contexto = ContextoCreditoTeste()
        contexto.state[credito.ETAPA_CREDITO] = credito.AGUARDANDO_PROXIMA_ACAO
        resultado = {
            "sucesso": True,
            "tipo": "aumento_limite",
            "status": "rejeitado",
            "score_atual": 780,
            "limite_atual": "9000.00",
            "novo_limite_solicitado": "20000.00",
            "limite_maximo_score": "18000.00",
        }

        with patch(
            "agentes.credito.agent.solicitar_aumento_limite",
            return_value=resultado,
        ) as solicitar:
            resposta = interceptar_fluxo_credito(
                contexto, self._requisicao(contexto, "20 mil")
            )

        solicitar.assert_called_once_with("20 mil", contexto)
        self.assertIn("entrevista de crédito", resposta.content.parts[0].text)
        self.assertEqual(contexto.state[OPCOES_RESPOSTA], ["Sim", "Não"])

    def test_rejeicao_oferece_entrevista_disponivel(self):
        contexto = ContextoCreditoTeste()
        contexto.state[credito.RESULTADO_CREDITO] = {
            "sucesso": True,
            "tipo": "aumento_limite",
            "status": "rejeitado",
            "score_atual": 599,
            "limite_atual": "5000.00",
            "novo_limite_solicitado": "12000.00",
            "limite_maximo_score": "11000.00",
        }

        resposta = interceptar_fluxo_credito(
            contexto,
            self._requisicao(contexto, "12000", retorno_ferramenta=True),
        )

        self.assertIn("entrevista de crédito", resposta.content.parts[0].text)
        self.assertEqual(contexto.state[OPCOES_RESPOSTA], ["Sim", "Não"])

    def test_aceita_entrevista_e_transfere_sem_apagar_pendencia(self):
        contexto = ContextoCreditoTeste()
        contexto.state.update(
            {
                credito.ETAPA_CREDITO: credito.AGUARDANDO_DECISAO_ENTREVISTA,
                credito.PENDENCIA_REANALISE: {
                    "novo_limite_solicitado": "8000.00"
                },
            }
        )

        resposta = interceptar_fluxo_credito(
            contexto, self._requisicao(contexto, "sim")
        )

        self.assertEqual(resposta.content.parts[0].text, "")
        self.assertEqual(
            contexto.actions.transfer_to_agent, "entrevista_credito"
        )
        self.assertIsNone(contexto.state[OPCOES_RESPOSTA])
        self.assertEqual(
            contexto.state[credito.PENDENCIA_REANALISE][
                "novo_limite_solicitado"
            ],
            "8000.00",
        )

    def test_recusa_entrevista_sem_encerrar_atendimento(self):
        contexto = ContextoCreditoTeste()
        contexto.state.update(
            {
                credito.ETAPA_CREDITO: credito.AGUARDANDO_DECISAO_ENTREVISTA,
                credito.PENDENCIA_REANALISE: {
                    "novo_limite_solicitado": "8000.00"
                },
            }
        )

        resposta = interceptar_fluxo_credito(
            contexto, self._requisicao(contexto, "não")
        )

        self.assertIn("outro assunto", resposta.content.parts[0].text)
        self.assertFalse(contexto.actions.escalate)
        self.assertIsNone(contexto.state[credito.PENDENCIA_REANALISE])

    def test_retorno_de_recusa_e_consulta_preserva_resultado_da_consulta(self):
        contexto = ContextoCreditoTeste()
        contexto.state.update(
            {
                credito.ETAPA_CREDITO: credito.AGUARDANDO_DECISAO_ENTREVISTA,
                credito.PENDENCIA_REANALISE: {
                    "novo_limite_solicitado": "8000.00"
                },
                "acao_entrevista_credito": "recusada",
                credito.RESULTADO_CREDITO: {
                    "sucesso": True,
                    "tipo": "consulta_limite",
                    "limite_atual": "5000.00",
                },
            }
        )
        contexto.user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text="Não, mas qual é meu limite?")],
        )
        requisicao = LlmRequest(
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="recusar_entrevista_credito",
                                response={"sucesso": True},
                            )
                        ),
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="consultar_limite_credito",
                                response={"sucesso": True},
                            )
                        ),
                    ],
                )
            ]
        )

        resposta = interceptar_fluxo_credito(contexto, requisicao)

        self.assertIn("R$ 5.000,00", resposta.content.parts[0].text)
        self.assertIsNone(contexto.state[credito.PENDENCIA_REANALISE])

    def test_retorno_da_entrevista_reanalisa_mesmo_limite(self):
        contexto = ContextoCreditoTeste()
        contexto.state.update(
            {
                entrevista_credito.RETORNO_ENTREVISTA: {
                    "score_atualizado": 700
                },
                credito.PENDENCIA_REANALISE: {
                    "novo_limite_solicitado": "8000.00"
                },
            }
        )
        resultado = {
            "sucesso": True,
            "tipo": "aumento_limite",
            "status": "aprovado",
            "limite_atual": "5000.00",
            "novo_limite_solicitado": "8000.00",
        }

        with patch(
            "agentes.credito.agent.solicitar_aumento_limite",
            return_value=resultado,
        ) as solicitar:
            resposta = interceptar_fluxo_credito(
                contexto, self._requisicao(contexto, "não")
            )

        solicitar.assert_called_once_with("8000.00", contexto)
        self.assertIn("solicitação foi aprovada", resposta.content.parts[0].text)
        self.assertIsNone(contexto.state[credito.PENDENCIA_REANALISE])
        self.assertIsNone(
            contexto.state[entrevista_credito.RETORNO_ENTREVISTA]
        )

    def test_retorno_de_entrevista_sem_pedido_pendente_explica_ausencia_de_reanalise(self):
        contexto = ContextoCreditoTeste()
        contexto.state[entrevista_credito.RETORNO_ENTREVISTA] = {
            "score_atualizado": 620
        }

        resposta = interceptar_fluxo_credito(
            contexto, self._requisicao(contexto, "não")
        )

        self.assertIn("não há solicitação de aumento pendente", resposta.content.parts[0].text)
        self.assertIsNone(contexto.state[entrevista_credito.RETORNO_ENTREVISTA])

    def test_reanalise_rejeitada_nao_oferece_nova_entrevista(self):
        contexto = ContextoCreditoTeste()
        contexto.state.update(
            {
                entrevista_credito.RETORNO_ENTREVISTA: {
                    "score_atualizado": 599
                },
                credito.PENDENCIA_REANALISE: {
                    "novo_limite_solicitado": "12000.00"
                },
            }
        )
        resultado = {
            "sucesso": True,
            "tipo": "aumento_limite",
            "status": "rejeitado",
            "score_atual": 599,
            "limite_atual": "5000.00",
            "novo_limite_solicitado": "12000.00",
            "limite_maximo_score": "11000.00",
        }

        with patch(
            "agentes.credito.agent.solicitar_aumento_limite",
            return_value=resultado,
        ):
            resposta = interceptar_fluxo_credito(
                contexto, self._requisicao(contexto, "não")
            )

        texto = resposta.content.parts[0].text
        self.assertIn("score é 599", texto)
        self.assertIn("R$ 11.000,00", texto)
        self.assertNotIn("Você deseja continuar com a entrevista", texto)
        self.assertEqual(
            contexto.state[credito.ETAPA_CREDITO],
            credito.AGUARDANDO_PROXIMA_ACAO,
        )

    def test_falha_na_reanalise_preserva_pendencia_para_tentar_novamente(self):
        contexto = ContextoCreditoTeste()
        contexto.state.update(
            {
                entrevista_credito.RETORNO_ENTREVISTA: {
                    "score_atualizado": 700
                },
                credito.PENDENCIA_REANALISE: {
                    "novo_limite_solicitado": "8000.00"
                },
            }
        )
        falha = {"sucesso": False, "erro": "base_credito_indisponivel"}
        sucesso = {
            "sucesso": True,
            "tipo": "aumento_limite",
            "status": "aprovado",
            "limite_atual": "5000.00",
            "novo_limite_solicitado": "8000.00",
        }

        with patch(
            "agentes.credito.agent.solicitar_aumento_limite",
            side_effect=[falha, sucesso],
        ) as solicitar:
            primeira = interceptar_fluxo_credito(
                contexto, self._requisicao(contexto, "não")
            )
            segunda = interceptar_fluxo_credito(
                contexto, self._requisicao(contexto, "tentar novamente")
            )

        self.assertIn("tentar novamente", primeira.content.parts[0].text)
        self.assertIn("solicitação foi aprovada", segunda.content.parts[0].text)
        self.assertEqual(solicitar.call_count, 2)
        self.assertIsNone(contexto.state[credito.PENDENCIA_REANALISE])

    def test_assunto_sem_suporte_retorna_triagem_sem_apagar_autenticacao(self):
        contexto = ContextoCreditoTeste()
        contexto.state[credito.ETAPA_CREDITO] = credito.AGUARDANDO_PROXIMA_ACAO

        interceptar_fluxo_credito(
            contexto, self._requisicao(contexto, "quero consultar câmbio")
        )

        self.assertEqual(contexto.actions.transfer_to_agent, "triagem")
        self.assertTrue(contexto.state["cliente_autenticado"])
        self.assertEqual(contexto.state["cpf_cliente"], "71048388050")

    def test_encerramento_no_credito_exige_confirmacao_explicita(self):
        contexto = ContextoCreditoTeste()
        solicitar_confirmacao_encerramento(contexto)

        resposta = interceptar_fluxo_credito(
            contexto, self._requisicao(contexto, "Sim, encerrar")
        )

        self.assertIn("Atendimento encerrado", resposta.content.parts[0].text)
        self.assertTrue(contexto.actions.escalate)
        self.assertTrue(contexto.state["atendimento_encerrado"])

    def test_atendimento_encerrado_nao_reabre_no_credito(self):
        contexto = ContextoCreditoTeste()
        contexto.state["atendimento_encerrado"] = True

        resposta = interceptar_fluxo_credito(
            contexto, self._requisicao(contexto, "qual é o meu limite?")
        )

        self.assertIn("já foi encerrado", resposta.content.parts[0].text)
        self.assertTrue(contexto.actions.escalate)

    def test_texto_financeiro_livre_do_modelo_e_substituido(self):
        contexto = ContextoCreditoTeste()
        candidata = LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text="Seu limite foi aprovado em 9000")],
            )
        )

        resposta = restringir_resposta_livre(contexto, candidata)

        self.assertNotIn("aprovado em 9000", resposta.content.parts[0].text)
        self.assertIn("novo limite total", resposta.content.parts[0].text)

    def test_agente_expoe_somente_ferramentas_deterministicas(self):
        ferramentas = asyncio.run(agente_credito.canonical_tools())
        self.assertEqual(
            {ferramenta.name for ferramenta in ferramentas},
            {
                "consultar_limite_credito",
                "consultar_score_credito",
                "solicitar_aumento_limite",
                "aceitar_entrevista_credito",
                "recusar_entrevista_credito",
                "solicitar_confirmacao_encerramento",
            },
        )

    def test_runner_transfere_estado_autenticado_da_triagem_ao_credito(self):
        with tempfile.TemporaryDirectory() as diretorio:
            raiz = Path(diretorio)
            clientes = raiz / "clientes.csv"
            faixas = raiz / "score_limite.csv"
            solicitacoes = raiz / "solicitacoes.csv"
            clientes.write_text(
                'CPF,"Data de Nascimento",Score,"Limite de Crédito"\n'
                '"710.483.880-50","29/07/1997",720,5000.00\n',
                encoding="utf-8",
            )
            faixas.write_text(
                "score_minimo,score_maximo,limite_maximo\n0,1000,10000.00\n",
                encoding="utf-8",
            )
            solicitacoes.write_text(
                ",".join(credito.COLUNAS_SOLICITACAO) + "\n",
                encoding="utf-8",
            )
            modelo = ModeloTransferenciaTeste(model="modelo-transferencia-teste")
            credito_teste = agente_credito.clone(update={"model": modelo})
            triagem_teste = root_agent.clone(
                update={"model": modelo, "sub_agents": [credito_teste]}
            )
            servico_sessao = InMemorySessionService()
            sessao = asyncio.run(
                servico_sessao.create_session(
                    app_name="teste",
                    user_id="cliente",
                    state={
                        "cliente_autenticado": True,
                        "cpf_cliente": "71048388050",
                    },
                )
            )
            runner = Runner(
                app_name="teste",
                agent=triagem_teste,
                session_service=servico_sessao,
            )
            mensagem = types.Content(
                role="user",
                parts=[types.Part.from_text(text="Qual é o meu limite?")],
            )

            with patch.object(credito, "CLIENTES_CSV", clientes), patch.object(
                credito, "SCORE_LIMITE_CSV", faixas
            ), patch.object(credito, "SOLICITACOES_CSV", solicitacoes):
                eventos = list(
                    runner.run(
                        user_id=sessao.user_id,
                        session_id=sessao.id,
                        new_message=mensagem,
                    )
                )
                with patch(
                    "agentes.triagem.guardrail._avaliar_resposta_com_modelo",
                    new=AsyncMock(return_value=(True, "Resposta segura.")),
                ):
                    eventos_retorno = list(
                        runner.run(
                            user_id=sessao.user_id,
                            session_id=sessao.id,
                            new_message=types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_text(
                                        text="Preciso de ajuda com outro assunto"
                                    )
                                ],
                            ),
                        )
                    )
                eventos_aumento = list(
                    runner.run(
                        user_id=sessao.user_id,
                        session_id=sessao.id,
                        new_message=types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(
                                    text="Quero aumentar para 8000"
                                )
                            ],
                        ),
                    )
                )
                with clientes.open(
                    "r", encoding="utf-8", newline=""
                ) as arquivo:
                    limite_final = next(csv.DictReader(arquivo))[
                        "Limite de Crédito"
                    ]
            sessao_final = asyncio.run(
                servico_sessao.get_session(
                    app_name="teste",
                    user_id=sessao.user_id,
                    session_id=sessao.id,
                )
            )
            asyncio.run(runner.close())

        respostas = [
            "".join(parte.text or "" for parte in evento.content.parts or [])
            for evento in eventos
            if evento.is_final_response() and evento.content
        ]
        self.assertTrue(
            any("Seu limite atual é R$ 5.000,00" in texto for texto in respostas),
            respostas,
        )
        self.assertTrue(
            any(evento.author == "credito" for evento in eventos), eventos
        )
        self.assertTrue(
            any(
                evento.author == "credito" and evento.is_final_response()
                for evento in eventos_retorno
            ),
            eventos_retorno,
        )
        self.assertTrue(sessao_final.state["cliente_autenticado"])
        self.assertEqual(sessao_final.state["cpf_cliente"], "71048388050")
        self.assertTrue(
            any(
                "solicitação foi aprovada" in "".join(
                    parte.text or "" for parte in evento.content.parts or []
                )
                for evento in eventos_aumento
                if evento.content
            ),
            eventos_aumento,
        )
        self.assertEqual(limite_final, "8000.00")
        self.assertGreaterEqual(modelo.chamadas_credito, 1)


if __name__ == "__main__":
    unittest.main()
