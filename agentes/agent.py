from pathlib import Path
import logging
import shutil

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DEFAULT_DIR = ROOT / "csv" / "default"
LOCAL_DIR = ROOT / "csv" / "local"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
POLITICAS_LIMITE_ANTERIORES = {
    """score_minimo,score_maximo,limite_maximo
0,299,1000.00
300,499,2500.00
500,699,5000.00
700,849,10000.00
850,1000,20000.00
""",
    """score_minimo,score_maximo,limite_maximo
0,299,1000.00
300,499,2500.00
500,699,5000.00
700,749,10000.00
750,849,15000.00
850,1000,20000.00
""",
}


def migrar_faixas_limite(caminho_padrao: Path, caminho_local: Path) -> None:
    try:
        politica_local = caminho_local.read_text(encoding="utf-8")
        if politica_local in POLITICAS_LIMITE_ANTERIORES:
            shutil.copy2(caminho_padrao, caminho_local)
    except (OSError, UnicodeDecodeError):
        logger.warning("Não foi possível verificar a política local de crédito")


for arquivo in DEFAULT_DIR.iterdir():
    destino = LOCAL_DIR / arquivo.name
    if not destino.exists():
        shutil.copy2(arquivo, destino)

migrar_faixas_limite(
    DEFAULT_DIR / "score_limite.csv", LOCAL_DIR / "score_limite.csv"
)

from .credito.agent import agente_credito
from .credito.tools.credito import (
    migrar_base_clientes,
    recuperar_transacao_pendente,
)
from .entrevista_credito.agent import agente_entrevista_credito
from .triagem.agent import agente_triagem


migrar_base_clientes(DEFAULT_DIR / "clientes.csv", LOCAL_DIR / "clientes.csv")
recuperar_transacao_pendente()

root_agent = agente_triagem.clone(
    update={"sub_agents": [agente_credito, agente_entrevista_credito]}
)
