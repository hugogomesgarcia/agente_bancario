from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types


def transferir_silenciosamente(
    callback_context: CallbackContext, nome_agente: str
) -> LlmResponse:
    """Emite o evento vazio exigido pelo ADK para transferir sem texto visível."""
    callback_context.actions.transfer_to_agent = nome_agente
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text="")]
        )
    )
