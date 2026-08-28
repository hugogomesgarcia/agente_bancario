from google.adk.agents import Agent
from pathlib import Path
import shutil

# Caminhos
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DIR = ROOT / "csv" / "default"
LOCAL_DIR = ROOT / "csv" / "local"


# Garante que os CSVs locais existam
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

for arquivo in DEFAULT_DIR.iterdir():
    destino = LOCAL_DIR / arquivo.name

    if not destino.exists():
        shutil.copy2(arquivo, destino)

root_agent = Agent(
    name="triagem",
    model="gemini-3.5-flash-lite",
    description="Agente responsável pela triagem de clientes do banco.",
    instruction="Você é um agente de triagem de um banco. Seja educado e objetivo.",
)
