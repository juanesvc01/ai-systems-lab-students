"""Cliente del LLM: la única parte de la aplicación que conoce al proveedor.

Recibe una lista de mensajes y parámetros; devuelve un LLMResponse.
El resto de la aplicación no importa `openai` ni sabe qué proveedor se usa.
"""

from dataclasses import dataclass

import openai

from config import Settings

# Un mensaje es un diccionario {"role": "system" | "user" | "assistant", "content": "..."}
Message = dict[str, str]


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: str  # "stop" = terminó normalmente, "length" = se agotó max_tokens
    prompt_tokens: int
    completion_tokens: int


class LLMError(Exception):
    """Error al comunicarse con el proveedor, expresado sin detalles del SDK."""


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = openai.OpenAI(base_url=settings.base_url, api_key=settings.api_key)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        try:
            completion = self._client.chat.completions.create(
                        model=self.settings.model,
                        messages=messages,
                        temperature=temperature if temperature is not None else self.settings.temperature,
                        max_tokens=max_tokens if max_tokens is not None else self.settings.max_tokens,
                        response_format={"type": "json_object"} if json_mode is True else {"type": "text"}
                    )
            return LLMResponse(
                text=completion.choices[0].message.content,
                model=completion.model,
                finish_reason=completion.choices[0].finish_reason,
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens
            )
        except openai.APIError as e:
            raise LLMError(f"Error al comunicarse con el LLM: {e}") from e

if __name__ == "__main__":
    # Prueba de humo: una sola llamada, sin interfaz ni historial.
    from config import load_settings

    client = LLMClient(load_settings())
    response = client.chat(
        [
            {"role": "system", "content": "Responde en una sola frase, en español."},
            {"role": "user", "content": "¿Qué es un Transformer en IA?"},
        ]
    )
    print(response)
