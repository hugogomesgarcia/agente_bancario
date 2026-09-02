import csv
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

import portalocker


class DadosCsvIndisponiveis(Exception):
    pass


TEMPO_LIMITE_BLOQUEIO_CSV = 5
INTERVALO_BLOQUEIO_CSV = 0.05


@contextmanager
def bloquear_csv(diretorio: Path):
    try:
        diretorio.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(
            diretorio / ".credito.lock",
            mode="a",
            timeout=TEMPO_LIMITE_BLOQUEIO_CSV,
            check_interval=INTERVALO_BLOQUEIO_CSV,
            raise_on_release_error=True,
        ):
            yield
    except (OSError, portalocker.exceptions.LockException) as erro:
        raise DadosCsvIndisponiveis from erro


def ler_csv(
    caminho: Path, colunas_obrigatorias: set[str]
) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with caminho.open("r", encoding="utf-8", newline="") as arquivo:
            leitor = csv.DictReader(arquivo, skipinitialspace=True)
            campos = list(leitor.fieldnames or [])
            if not colunas_obrigatorias.issubset(campos):
                raise DadosCsvIndisponiveis
            linhas = [dict(linha) for linha in leitor]
    except DadosCsvIndisponiveis:
        raise
    except (OSError, UnicodeDecodeError, csv.Error) as erro:
        raise DadosCsvIndisponiveis from erro
    return campos, linhas


def preparar_csv(
    caminho: Path, campos: list[str], linhas: list[dict[str, object]]
) -> Path:
    temporario: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=caminho.parent,
            prefix=f".{caminho.name}.",
            suffix=".tmp",
            delete=False,
        ) as arquivo:
            temporario = Path(arquivo.name)
            escritor = csv.DictWriter(
                arquivo, fieldnames=campos, extrasaction="ignore"
            )
            escritor.writeheader()
            escritor.writerows(linhas)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        return temporario
    except (OSError, csv.Error) as erro:
        if temporario is not None:
            temporario.unlink(missing_ok=True)
        raise DadosCsvIndisponiveis from erro


def substituir_csv(temporario: Path, destino: Path) -> None:
    try:
        os.replace(temporario, destino)
    except OSError as erro:
        raise DadosCsvIndisponiveis from erro
