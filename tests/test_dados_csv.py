from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Thread
import time
import unittest
from unittest.mock import patch

import portalocker

from agentes.compartilhado import dados_csv


ROOT = Path(__file__).resolve().parents[1]


class BloqueioCsvTest(unittest.TestCase):
    def setUp(self):
        self.diretorio_temporario = tempfile.TemporaryDirectory()
        self.diretorio = Path(self.diretorio_temporario.name)

    def tearDown(self):
        self.diretorio_temporario.cleanup()

    def test_serializa_threads_no_mesmo_diretorio(self):
        primeira_entrou = Event()
        liberar_primeira = Event()
        segunda_entrou = Event()

        def primeira():
            with dados_csv.bloquear_csv(self.diretorio):
                primeira_entrou.set()
                liberar_primeira.wait(timeout=2)

        def segunda():
            primeira_entrou.wait(timeout=2)
            with dados_csv.bloquear_csv(self.diretorio):
                segunda_entrou.set()

        thread_primeira = Thread(target=primeira)
        thread_segunda = Thread(target=segunda)
        thread_primeira.start()
        thread_segunda.start()

        self.assertTrue(primeira_entrou.wait(timeout=2))
        self.assertFalse(segunda_entrou.wait(timeout=0.1))
        liberar_primeira.set()
        self.assertTrue(segunda_entrou.wait(timeout=2))

        thread_primeira.join(timeout=2)
        thread_segunda.join(timeout=2)
        self.assertFalse(thread_primeira.is_alive())
        self.assertFalse(thread_segunda.is_alive())

    def test_serializa_processos_no_mesmo_diretorio(self):
        marcador = self.diretorio / "estado.txt"
        codigo = """
from pathlib import Path
import sys

from agentes.compartilhado.dados_csv import bloquear_csv

diretorio = Path(sys.argv[1])
marcador = diretorio / "estado.txt"
marcador.write_text("tentando", encoding="utf-8")
with bloquear_csv(diretorio):
    marcador.write_text("adquirido", encoding="utf-8")
"""

        with dados_csv.bloquear_csv(self.diretorio):
            processo = subprocess.Popen(
                [sys.executable, "-c", codigo, str(self.diretorio)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            limite = time.monotonic() + 2
            while not marcador.exists() and time.monotonic() < limite:
                time.sleep(0.01)

            self.assertTrue(marcador.exists())
            self.assertEqual(marcador.read_text(encoding="utf-8"), "tentando")
            self.assertIsNone(processo.poll())

        stdout, stderr = processo.communicate(timeout=5)
        self.assertEqual(processo.returncode, 0, stderr or stdout)
        self.assertEqual(marcador.read_text(encoding="utf-8"), "adquirido")

    def test_timeout_vira_erro_de_dominio(self):
        caminho_lock = self.diretorio / ".credito.lock"
        with portalocker.Lock(caminho_lock, mode="a", timeout=0):
            with patch.object(dados_csv, "TEMPO_LIMITE_BLOQUEIO_CSV", 0.1):
                with self.assertRaises(dados_csv.DadosCsvIndisponiveis) as erro:
                    with dados_csv.bloquear_csv(self.diretorio):
                        self.fail("O segundo lock não deveria ser adquirido")

        self.assertIsInstance(
            erro.exception.__cause__, portalocker.exceptions.LockException
        )

    def test_libera_lock_quando_corpo_falha(self):
        with self.assertRaisesRegex(ValueError, "falha controlada"):
            with dados_csv.bloquear_csv(self.diretorio):
                raise ValueError("falha controlada")

        with dados_csv.bloquear_csv(self.diretorio):
            adquirido_novamente = True

        self.assertTrue(adquirido_novamente)

    def test_diretorios_distintos_nao_se_bloqueiam(self):
        outro_diretorio = self.diretorio / "outro"

        with dados_csv.bloquear_csv(self.diretorio):
            with dados_csv.bloquear_csv(outro_diretorio):
                adquirido = True

        self.assertTrue(adquirido)


if __name__ == "__main__":
    unittest.main()
