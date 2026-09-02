import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CarregamentoAdkTest(unittest.TestCase):
    def test_agent_loader_carrega_composicao_multiagente(self):
        codigo = """
import sys
from google.adk.cli.utils.agent_loader import AgentLoader

agente = AgentLoader(sys.argv[1]).load_agent("agentes")
assert agente.name == "triagem"
assert [subagente.name for subagente in agente.sub_agents] == [
    "credito", "entrevista_credito", "cambio"
]
assert all(subagente.parent_agent is agente for subagente in agente.sub_agents)
"""

        resultado = subprocess.run(
            [sys.executable, "-I", "-c", codigo, str(ROOT / "agentes")],
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(resultado.returncode, 0, resultado.stderr)


if __name__ == "__main__":
    unittest.main()
